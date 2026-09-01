"""test_statistics.py — the uncertainty machinery, checked against known answers.

Every number in the ablation report and the suite summary now carries an interval and
a p-value from this module, so an error here would be invisible and everywhere.
"""

from __future__ import annotations

import pytest

from docureason.tripath.evaluation.statistics import (
    bootstrap_ci,
    compare_paired,
    mean,
    paired_bootstrap,
    paired_permutation_test,
)

pytestmark = pytest.mark.unit


def test_mean_of_nothing_is_none_not_zero() -> None:
    """Unmeasured must not read as zero - that is how a disabled gate scored 1.0."""
    assert mean([]) is None
    assert mean([None, None]) is None
    assert mean([1.0, 2.0]) == 1.5


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    values = [0.2, 0.4, 0.4, 0.6, 0.8, 1.0, 0.6, 0.4]
    interval = bootstrap_ci(values, seed=1)
    assert interval.low <= interval.point <= interval.high
    assert interval.n == len(values)


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    """A CI gate cannot tolerate a metric that moves between identical runs.

    Only determinism is asserted, not seed sensitivity: on a small discrete sample the
    resampled means take few distinct values, so two seeds can legitimately land on the
    same rounded percentile bounds.
    """
    values = [0.1, 0.9, 0.5, 0.5, 0.7]
    assert bootstrap_ci(values, seed=7).as_dict() == bootstrap_ci(values, seed=7).as_dict()
    assert (
        bootstrap_ci(values, seed=7, iterations=250).as_dict() == bootstrap_ci(values, seed=7, iterations=250).as_dict()
    )


def test_a_constant_sample_has_a_zero_width_interval() -> None:
    interval = bootstrap_ci([0.5] * 12, seed=0)
    assert interval.low == interval.high == 0.5
    assert interval.width == 0.0


def test_single_observation_is_reported_as_degenerate() -> None:
    interval = bootstrap_ci([0.42])
    assert (interval.low, interval.point, interval.high) == (0.42, 0.42, 0.42)
    assert interval.iterations == 0, "no resampling is possible from one observation"


def test_more_data_narrows_the_interval() -> None:
    narrow = bootstrap_ci([0.4, 0.6] * 60, seed=3)
    wide = bootstrap_ci([0.4, 0.6] * 3, seed=3)
    assert narrow.width < wide.width


def test_identical_systems_produce_zero_delta_and_p_of_one() -> None:
    result = compare_paired([0.5] * 8, [0.5] * 8)
    assert result.delta == 0.0
    assert result.p_value == 1.0
    assert result.significant is False


def test_a_uniform_improvement_is_exact_and_significant() -> None:
    """With n<=14 the test enumerates all 2**n sign flips, so p is exact."""
    result = compare_paired([0.9] * 10, [0.4] * 10)
    assert result.exact is True
    assert result.delta == pytest.approx(0.5)
    # Both the all-flip and no-flip assignments are as extreme as observed.
    assert result.p_value == pytest.approx(2 / 2**10)
    assert result.significant is True


def test_large_samples_switch_to_sampling_with_an_add_one_correction() -> None:
    result = compare_paired([0.9] * 40, [0.1] * 40, iterations=500)
    assert result.exact is False
    assert result.p_value == pytest.approx(1 / 501)
    assert result.p_value > 0.0, "a sampled test cannot justify claiming p == 0"


def test_pairing_drops_unmatched_observations_and_says_so() -> None:
    result = compare_paired([1.0, None, 0.5], [0.2, 0.4, 0.1])
    assert result.n == 2
    assert any("dropped" in note for note in result.notes)


def test_small_samples_carry_a_warning() -> None:
    assert any("n=5" in note for note in compare_paired([0.4] * 5, [0.5] * 5).notes)


def test_empty_comparison_reports_nothing_measurable() -> None:
    result = compare_paired([], [])
    assert result.delta is None and result.p_value is None
    assert result.significant is None
    assert result.as_dict()["notes"] == ["no paired queries"]


def test_paired_delta_interval_matches_the_difference_sample() -> None:
    a = [0.8, 0.6, 0.9, 0.7]
    b = [0.5, 0.5, 0.6, 0.4]
    paired = paired_bootstrap(a, b, seed=2)
    direct = bootstrap_ci([x - y for x, y in zip(a, b)], seed=2)
    assert paired.as_dict() == direct.as_dict()


def test_permutation_test_is_two_sided() -> None:
    """Direction must not change the p-value, only the sign of the delta."""
    forward, _ = paired_permutation_test([0.8] * 9, [0.3] * 9)
    reverse, _ = paired_permutation_test([0.3] * 9, [0.8] * 9)
    assert forward == reverse


def test_noise_is_not_called_significant() -> None:
    a = [0.5, 0.4, 0.6, 0.5, 0.55, 0.45, 0.5, 0.52]
    b = [0.5, 0.45, 0.55, 0.52, 0.5, 0.48, 0.51, 0.49]
    assert compare_paired(a, b, seed=11).significant is False
