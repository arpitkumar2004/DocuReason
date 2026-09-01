"""statistics.py — uncertainty and significance for evaluation metrics.

Every number the harness reported was a bare mean over a handful of queries, with
no interval and no test.  On a 3-case smoke suite the difference between
``recall@5 = 0.67`` and ``recall@5 = 1.00`` is one query, so a mean alone could not
distinguish a real improvement from resampling noise - and the ablation table in
``Documentations/eval_method.md`` quoted deltas to one decimal place with nothing
behind them.

Three procedures, all non-parametric because metric distributions here are bounded,
discrete and badly skewed (recall@5 over 5 slots takes six possible values), which
rules out the normal approximation a t-interval assumes:

* :func:`bootstrap_ci` - percentile bootstrap interval for the mean of one metric.
* :func:`paired_bootstrap` - interval for the *difference* between two systems
  scored on the same queries.  Paired, because the query-to-query variance dwarfs
  the between-system variance: an unpaired test on the same data is far less
  sensitive and would call real ablation effects insignificant.
* :func:`paired_permutation_test` - exact-when-small sign-flip test for the same
  difference, giving a p-value that makes no distributional assumption at all.

Implemented in pure Python against :mod:`random` rather than NumPy: this module is
imported by the CI metric gate, where a deterministic seed and no array dependency
matter more than speed. A 10,000-iteration bootstrap over a few hundred queries runs
in about a second.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: Default resample count. 10,000 puts the Monte-Carlo error on a 95% percentile
#: bound at roughly +/-0.005 of a metric point, an order of magnitude below the
#: 0.01 tolerance the CI regression gate uses.
DEFAULT_ITERATIONS = 10_000

#: Below this many observations, :func:`paired_permutation_test` enumerates all
#: 2**n sign assignments and returns an exact p-value instead of sampling.
EXACT_PERMUTATION_LIMIT = 14


def mean(values: Sequence[float]) -> Optional[float]:
    """Arithmetic mean, or ``None`` for an empty sample.

    ``None`` rather than ``0.0``: a metric with no observations is *unmeasured*,
    and reporting it as zero is what let a disabled attribution gate publish a
    perfect faithfulness score (audit P0-2).
    """
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


@dataclass
class ConfidenceInterval:
    """A percentile bootstrap interval for the mean of one metric."""

    point: Optional[float]
    low: Optional[float]
    high: Optional[float]
    n: int
    confidence: float = 0.95
    iterations: int = DEFAULT_ITERATIONS

    @property
    def width(self) -> Optional[float]:
        if self.low is None or self.high is None:
            return None
        return self.high - self.low

    def as_dict(self) -> Dict[str, Any]:
        return {
            "point": _round(self.point),
            "ci_low": _round(self.low),
            "ci_high": _round(self.high),
            "ci_width": _round(self.width),
            "n": self.n,
            "confidence": self.confidence,
            "bootstrap_iterations": self.iterations,
        }


@dataclass
class PairedComparison:
    """A paired comparison of two systems scored on the same queries."""

    n: int
    mean_a: Optional[float]
    mean_b: Optional[float]
    delta: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    p_value: Optional[float]
    alpha: float = 0.05
    iterations: int = DEFAULT_ITERATIONS
    exact: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def significant(self) -> Optional[bool]:
        """True when the paired test rejects at :attr:`alpha`.

        Reported alongside the interval, never instead of it: with n this small a
        non-significant result usually means "not enough queries", not "no effect".
        """
        if self.p_value is None:
            return None
        return self.p_value < self.alpha

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_paired": self.n,
            "mean_a": _round(self.mean_a),
            "mean_b": _round(self.mean_b),
            "delta": _round(self.delta),
            "delta_ci_low": _round(self.ci_low),
            "delta_ci_high": _round(self.ci_high),
            "p_value": _round(self.p_value, 5),
            "alpha": self.alpha,
            "significant": self.significant,
            "test": "exact paired sign-flip" if self.exact else "sampled paired sign-flip",
            "iterations": self.iterations,
            "notes": list(self.notes),
        }


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def bootstrap_ci(
    values: Sequence[float],
    confidence: float = 0.95,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap interval for the mean of *values*.

    Resamples the observations with replacement *iterations* times and takes the
    empirical quantiles of the resampled means.  A single observation yields a
    degenerate interval (both bounds equal the point estimate) rather than an error:
    that is the honest answer for n=1, and it makes the width visibly useless.
    """
    clean = [float(v) for v in values if v is not None]
    n = len(clean)
    point = mean(clean)
    if n == 0:
        return ConfidenceInterval(None, None, None, 0, confidence, iterations)
    if n == 1:
        return ConfidenceInterval(point, point, point, 1, confidence, 0)

    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += clean[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    alpha = 1.0 - confidence
    return ConfidenceInterval(
        point=point,
        low=_quantile(means, alpha / 2.0),
        high=_quantile(means, 1.0 - alpha / 2.0),
        n=n,
        confidence=confidence,
        iterations=iterations,
    )


def _quantile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted sample."""
    if not sorted_values:
        raise ValueError("quantile of an empty sample")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _pair(a: Sequence[float], b: Sequence[float]) -> List[tuple[float, float]]:
    """Observations present in both samples, positionally aligned.

    Callers align by query id before calling; this drops any pair where either side
    is missing so a metric that one system could not produce cannot silently shift
    the other system's mean.
    """
    pairs: List[tuple[float, float]] = []
    for left, right in zip(a, b):
        if left is None or right is None:
            continue
        pairs.append((float(left), float(right)))
    return pairs


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bootstrap interval for the mean paired difference ``a - b``.

    Resamples *queries*, not scores, so each resample keeps both systems' results
    for the same query together.  An interval that excludes zero is the same
    conclusion a significance test reaches, stated in metric units.
    """
    pairs = _pair(a, b)
    deltas = [left - right for left, right in pairs]
    return bootstrap_ci(deltas, confidence=confidence, iterations=iterations, seed=seed)


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
) -> tuple[Optional[float], bool]:
    """Two-sided paired sign-flip test on ``a - b``.

    Under the null hypothesis that the two systems are interchangeable, the sign of
    each query's difference is arbitrary.  The test enumerates (or samples) sign
    assignments and reports how often ``|mean delta|`` is at least as extreme as
    observed.  Returns ``(p_value, exact)``; ``exact`` is True when all ``2**n``
    assignments were enumerated.

    The p-value uses the add-one correction ``(hits + 1) / (iterations + 1)`` in the
    sampled case, so it can never be reported as exactly 0 - a sampled test cannot
    justify that claim.
    """
    pairs = _pair(a, b)
    deltas = [left - right for left, right in pairs]
    n = len(deltas)
    if n == 0:
        return None, False

    observed = abs(sum(deltas) / n)
    # Ties contribute nothing under sign flipping; with every delta zero the two
    # systems produced identical scores, which is p = 1 by definition.
    if all(delta == 0.0 for delta in deltas):
        return 1.0, True

    if n <= EXACT_PERMUTATION_LIMIT:
        hits = 0
        total = 1 << n
        for mask in range(total):
            flipped = sum(-delta if (mask >> i) & 1 else delta for i, delta in enumerate(deltas))
            if abs(flipped / n) >= observed - 1e-12:
                hits += 1
        return hits / total, True

    rng = random.Random(seed)
    hits = 0
    for _ in range(iterations):
        flipped = sum(delta if rng.random() < 0.5 else -delta for delta in deltas)
        if abs(flipped / n) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (iterations + 1), False


def compare_paired(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
    iterations: int = DEFAULT_ITERATIONS,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedComparison:
    """Full paired comparison of *a* against *b*: delta, bootstrap CI and p-value."""
    pairs = _pair(a, b)
    n = len(pairs)
    notes: List[str] = []
    if n == 0:
        return PairedComparison(0, None, None, None, None, None, None, alpha, iterations, False, ["no paired queries"])

    dropped = max(len(a), len(b)) - n
    if dropped > 0:
        notes.append(f"{dropped} query(ies) dropped: metric missing on one side")
    if n < 30:
        notes.append(f"n={n} is small; a non-significant result here is weak evidence of no effect")

    interval = paired_bootstrap(a, b, confidence=confidence, iterations=iterations, seed=seed)
    p_value, exact = paired_permutation_test(a, b, iterations=iterations, seed=seed)

    return PairedComparison(
        n=n,
        mean_a=mean([left for left, _ in pairs]),
        mean_b=mean([right for _, right in pairs]),
        delta=interval.point,
        ci_low=interval.low,
        ci_high=interval.high,
        p_value=p_value,
        alpha=alpha,
        iterations=iterations,
        exact=exact,
        notes=notes,
    )


__all__ = [
    "ConfidenceInterval",
    "PairedComparison",
    "bootstrap_ci",
    "compare_paired",
    "mean",
    "paired_bootstrap",
    "paired_permutation_test",
]
