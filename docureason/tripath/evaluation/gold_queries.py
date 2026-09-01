"""gold_queries.py — build a stratified query set whose gold labels are derivable.

The previous evaluation set was ``benchmark_dataset.py``: hand-written questions with
invented answers ("2.4M" against a table containing 100M and 125M) and gold ids naming
fixture stems that no longer resolved.  Audit P0-4 and §10 both come back to the same
thing - if the labels are arbitrary, a retrieval score measures the labels.

The approach here is different: **derive** each query from the corpus so its gold
region is known by construction, and refuse any query whose anchor is not unique.

* ``tabular_lookup`` - a row label and a column header from a parsed table.  The gold
  region is the table; the answer is the cell where they intersect.
* ``numeric`` - the same, restricted to cells that parse as numbers, phrased as a
  value question.  Separate stratum because exact-figure retrieval is the claim the
  SQL path exists to support.
* ``prose`` - a distinctive phrase from a body region, verified to appear in exactly
  one region corpus-wide.
* ``figure`` - a figure caption, with the same uniqueness requirement.

Uniqueness is the whole mechanism.  A query is emitted only if its anchor identifies
one region across the entire corpus; otherwise a retriever could return a *correct*
region and be scored wrong, which measures the label format rather than retrieval.
Rejected candidates are counted by reason, so the yield is visible instead of the
generator quietly emitting whatever it managed.

These labels are **programmatically derived, not human-annotated**, and every record
says so in ``provenance``.  That is a real limitation: derivable queries over-represent
lookup and under-represent multi-hop reasoning, which no rule can generate.  It is not
a substitute for annotation - it is what makes the numbers falsifiable without it.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from docureason.tripath.utils import get_logger

logger = get_logger(__name__)

STRATA = ("tabular_lookup", "numeric", "prose", "figure")

#: Region types that count as prose for the prose stratum.
_PROSE_TYPES = frozenset({"body", "paragraph", "text"})

#: A phrase this short is unlikely to be distinctive; this long stops being a query
#: and starts being a copy of the region.
_MIN_ANCHOR_WORDS = 4
_MAX_ANCHOR_WORDS = 9

#: Cell/label text this short carries no signal ("-", "1", "$").
_MIN_LABEL_CHARS = 4

_NUMERIC = re.compile(r"^\(?-?[$€£]?\s?\d[\d,]*\.?\d*\s?[%)]?$")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+")

#: Words too common to make a phrase distinctive, so an anchor must not be built
#: entirely from them.
_STOPWORDS = frozenset(
    """the a an and or of to in for on at by with from as is are was were be been being
    this that these those it its our we their his her they you your not no if then than
    which who whom whose what when where why how all any both each few more most other
    some such only own same so too very can will just should now""".split()
)


@dataclass
class GoldQuery:
    """One evaluation case with a gold region derived from the corpus."""

    id: str
    stratum: str
    query: str
    relevant_ids: List[str]
    document_id: str
    source: str
    derivation: str
    anchor: Dict[str, Any]
    ground_truth_answer: Optional[str] = None
    provenance: str = "programmatic"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "stratum": self.stratum,
            "query": self.query,
            "relevant_ids": list(self.relevant_ids),
            "document_id": self.document_id,
            "source": self.source,
            "ground_truth_answer": self.ground_truth_answer,
            "derivation": self.derivation,
            "anchor": dict(self.anchor),
            "provenance": self.provenance,
        }


@dataclass
class GenerationReport:
    """What the generator produced and, more usefully, what it threw away."""

    queries: List[GoldQuery] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)
    per_stratum: Counter = field(default_factory=Counter)
    regions_scanned: int = 0
    documents: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query_count": len(self.queries),
            "documents": self.documents,
            "regions_scanned": self.regions_scanned,
            "per_stratum": dict(self.per_stratum),
            "rejected": dict(self.rejected),
        }


def _regions(corpus: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """``(document, region)`` pairs from a ``corpus.json`` payload."""
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for document in corpus.get("documents") or []:
        for region in document.get("regions") or []:
            pairs.append((document, region))
    return pairs


def _region_id(region: Dict[str, Any]) -> str:
    return str(region.get("id") or (region.get("metadata") or {}).get("region_id") or "")


def _normalise(text: str) -> str:
    return " ".join(str(text).lower().split())


def build_phrase_index(corpus: Dict[str, Any], anchor_words: int) -> Dict[str, List[str]]:
    """Map every *anchor_words*-length word shingle to the regions containing it.

    Built once for the whole corpus: uniqueness has to be checked globally, and doing
    it with a substring scan per candidate would be quadratic in corpus size.
    """
    index: Dict[str, List[str]] = {}
    for _, region in _regions(corpus):
        region_id = _region_id(region)
        if not region_id:
            continue
        words = _WORD.findall(_normalise(region.get("text", "")))
        seen_here = set()
        for start in range(0, max(0, len(words) - anchor_words + 1)):
            shingle = " ".join(words[start : start + anchor_words])
            if shingle in seen_here:
                continue
            seen_here.add(shingle)
            index.setdefault(shingle, []).append(region_id)
    return index


def _is_distinctive(shingle: str) -> bool:
    """A phrase built only from stopwords cannot anchor a query."""
    words = shingle.split()
    content = [word for word in words if word not in _STOPWORDS]
    return len(content) >= max(2, len(words) // 2)


def _table_payload(region: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = region.get("table_json")
    if isinstance(raw, str):
        try:
            import json

            raw = json.loads(raw)
        except Exception:
            return None
    return raw if isinstance(raw, dict) else None


def _is_numeric(value: Any) -> bool:
    return bool(_NUMERIC.match(str(value).strip()))


def generate_queries(
    corpus: Dict[str, Any],
    per_stratum: int = 50,
    seed: int = 0,
    strata: Sequence[str] = STRATA,
) -> GenerationReport:
    """Derive up to *per_stratum* queries for each stratum from *corpus*.

    Deterministic for a given corpus and seed, because an evaluation set that changes
    between runs makes every comparison against it meaningless.
    """
    rng = random.Random(seed)
    report = GenerationReport()
    pairs = _regions(corpus)
    report.regions_scanned = len(pairs)
    report.documents = len(corpus.get("documents") or [])

    phrase_indexes = {
        words: build_phrase_index(corpus, words) for words in range(_MIN_ANCHOR_WORDS, _MAX_ANCHOR_WORDS + 1)
    }

    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"q-{counter:04d}"

    table_pairs = [(d, r) for d, r in pairs if r.get("type") == "table" and _table_payload(r)]
    prose_pairs = [(d, r) for d, r in pairs if r.get("type") in _PROSE_TYPES]
    figure_pairs = [(d, r) for d, r in pairs if r.get("type") == "figure"]
    rng.shuffle(table_pairs)
    rng.shuffle(prose_pairs)
    rng.shuffle(figure_pairs)

    if "tabular_lookup" in strata:
        for query in _table_queries(table_pairs, per_stratum, rng, report, numeric_only=False, next_id=next_id):
            report.queries.append(query)
            report.per_stratum["tabular_lookup"] += 1

    if "numeric" in strata:
        rng.shuffle(table_pairs)
        for query in _table_queries(table_pairs, per_stratum, rng, report, numeric_only=True, next_id=next_id):
            report.queries.append(query)
            report.per_stratum["numeric"] += 1

    if "prose" in strata:
        for query in _phrase_queries(prose_pairs, phrase_indexes, per_stratum, rng, report, "prose", next_id):
            report.queries.append(query)
            report.per_stratum["prose"] += 1

    if "figure" in strata:
        if not figure_pairs:
            # Distinguish "no figure regions exist in this corpus" from "the generator
            # found none it could use": HTML filings carry images that Docling's HTML
            # pipeline does not emit as figure regions, so this stratum is empty by
            # corpus composition, not by rejection.
            report.rejected["figure:no_regions_in_corpus"] += 1
        for query in _phrase_queries(figure_pairs, phrase_indexes, per_stratum, rng, report, "figure", next_id):
            report.queries.append(query)
            report.per_stratum["figure"] += 1

    logger.info(
        "gold_queries: %d query(ies) from %d region(s) across %d document(s); rejected %s",
        len(report.queries),
        report.regions_scanned,
        report.documents,
        dict(report.rejected),
    )
    return report


def _table_queries(
    table_pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    limit: int,
    rng: random.Random,
    report: GenerationReport,
    numeric_only: bool,
    next_id: Any,
) -> Iterable[GoldQuery]:
    """Row-label + column-header lookups, one per table at most.

    One per table keeps the set from being fifty questions about the same balance
    sheet, which would make a per-stratum mean a statement about one table.
    """
    stratum = "numeric" if numeric_only else "tabular_lookup"
    produced = 0

    for document, region in table_pairs:
        if produced >= limit:
            break
        payload = _table_payload(region)
        region_id = _region_id(region)
        if not payload or not region_id:
            report.rejected[f"{stratum}:no_payload"] += 1
            continue

        columns = [str(c).strip() for c in payload.get("columns") or []]
        rows = payload.get("rows") or []
        if len(columns) < 2 or not rows:
            report.rejected[f"{stratum}:too_small"] += 1
            continue

        candidates: List[Tuple[str, str, str]] = []  # (row_label, column, value)
        for row in rows:
            if not row:
                continue
            label = str(row[0]).strip()
            if len(label) < _MIN_LABEL_CHARS or _is_numeric(label):
                continue
            for position in range(1, min(len(row), len(columns))):
                value = str(row[position]).strip()
                column = columns[position]
                if not value or len(column) < 2:
                    continue
                if numeric_only and not _is_numeric(value):
                    continue
                if not numeric_only and _is_numeric(value) and len(value) < _MIN_LABEL_CHARS:
                    continue
                candidates.append((label, column, value))

        if not candidates:
            report.rejected[f"{stratum}:no_cell"] += 1
            continue

        label, column, value = rng.choice(candidates)
        if numeric_only:
            text = f"What is the {label} figure reported under {column}?"
        else:
            text = f"In the table reporting {label}, what value is shown for {column}?"

        yield GoldQuery(
            id=next_id(),
            stratum=stratum,
            query=text,
            relevant_ids=[region_id],
            document_id=str(document.get("id", "")),
            source=str(document.get("source", "")),
            derivation="table_cell_numeric" if numeric_only else "table_cell",
            anchor={"row_label": label, "column": column, "cell_value": value},
            ground_truth_answer=value,
        )
        produced += 1


def _phrase_queries(
    pairs: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    phrase_indexes: Dict[int, Dict[str, List[str]]],
    limit: int,
    rng: random.Random,
    report: GenerationReport,
    stratum: str,
    next_id: Any,
) -> Iterable[GoldQuery]:
    """Queries anchored on a phrase that occurs in exactly one region corpus-wide."""
    produced = 0

    for document, region in pairs:
        if produced >= limit:
            break
        region_id = _region_id(region)
        text = _normalise(region.get("text", ""))
        words = _WORD.findall(text)
        if not region_id or len(words) < _MIN_ANCHOR_WORDS + 2:
            report.rejected[f"{stratum}:too_short"] += 1
            continue

        anchor = _unique_phrase(words, region_id, phrase_indexes, rng)
        if anchor is None:
            # Boilerplate regions (risk-factor headers, legal footers) repeat verbatim
            # across filings and have no unique phrase; they cannot be gold-labelled.
            report.rejected[f"{stratum}:no_unique_phrase"] += 1
            continue

        if stratum == "figure":
            query_text = f"Which figure describes {anchor}?"
        else:
            query_text = f"What does the filing say about {anchor}?"

        yield GoldQuery(
            id=next_id(),
            stratum=stratum,
            query=query_text,
            relevant_ids=[region_id],
            document_id=str(document.get("id", "")),
            source=str(document.get("source", "")),
            derivation="unique_phrase",
            anchor={"phrase": anchor, "phrase_words": len(anchor.split())},
            ground_truth_answer=None,
        )
        produced += 1


def _unique_phrase(
    words: List[str],
    region_id: str,
    phrase_indexes: Dict[int, Dict[str, List[str]]],
    rng: random.Random,
) -> Optional[str]:
    """A shingle from *words* that appears only in *region_id*, or None."""
    lengths = sorted(phrase_indexes)
    for length in lengths:
        index = phrase_indexes[length]
        starts = list(range(0, max(0, len(words) - length + 1)))
        rng.shuffle(starts)
        for start in starts[:40]:
            shingle = " ".join(words[start : start + length])
            if not _is_distinctive(shingle):
                continue
            owners = index.get(shingle) or []
            if len(owners) == 1 and owners[0] == region_id:
                return shingle
    return None


def validate_queries(queries: Sequence[Dict[str, Any]], corpus: Dict[str, Any]) -> Dict[str, Any]:
    """Check every gold id resolves to a region, and every anchor is still unique.

    Run after generation and again before any evaluation: a corpus rebuilt with
    different chunk sizes or a different digest invalidates the gold ids, and silently
    scoring against dangling labels is how a suite comes to report zero for reasons
    that have nothing to do with retrieval.

    Uniqueness is checked over the **same word-token shingles** the generator used, not
    by raw substring search.  Those are not the same test: tokenizing drops digits and
    punctuation, so ``"revenue increased compared to"`` is a valid token shingle of
    ``"revenue increased 4% compared to"`` and is not a substring of it.  Validating
    with the stricter test reported 21 sound anchors as broken - a disagreement between
    two definitions of the same property, not a defect in the data.
    """
    known = {_region_id(region) for _, region in _regions(corpus) if _region_id(region)}

    dangling: List[str] = []
    non_unique: List[str] = []

    phrase_lengths = {
        len(str((record.get("anchor") or {}).get("phrase", "")).split())
        for record in queries
        if (record.get("anchor") or {}).get("phrase")
    }
    indexes = {length: build_phrase_index(corpus, length) for length in phrase_lengths if length > 0}

    for record in queries:
        for gold in record.get("relevant_ids") or []:
            if gold not in known:
                dangling.append(f"{record.get('id')}:{gold}")

        phrase = (record.get("anchor") or {}).get("phrase")
        if not phrase:
            continue
        owners = indexes.get(len(str(phrase).split()), {}).get(str(phrase)) or []
        expected = set(record.get("relevant_ids") or [])
        if len(owners) != 1 or set(owners) != expected:
            non_unique.append(f"{record.get('id')}:{len(owners)} owner(s)")

    return {
        "queries": len(queries),
        "regions_in_corpus": len(known),
        "dangling_gold_ids": dangling,
        "non_unique_anchors": non_unique,
        "valid": not dangling and not non_unique,
    }


def load_eval_cases(path: Union[str, Path], limit: int = 0, stratified: bool = False) -> List[Dict[str, Any]]:
    """Read a generated query file back as harness cases.

    Lives here rather than in a script because both the ablation sweep and the eval suite
    need it and must agree: two loaders that sample differently would report metrics over
    two different query sets under one corpus name.

    *stratified* samples ``limit`` evenly across strata. It is not a nicety - the
    generator emits stratum by stratum, so a plain prefix ``limit`` runs the whole
    evaluation against tabular lookup alone and calls the result a corpus-wide number.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = payload.get("queries") or []
    if not queries:
        raise ValueError(f"{path} contains no queries")

    if limit and stratified:
        by_stratum: Dict[str, List[Dict[str, Any]]] = {}
        for record in queries:
            by_stratum.setdefault(str(record.get("stratum", "unknown")), []).append(record)
        per = max(1, limit // max(1, len(by_stratum)))
        picked: List[Dict[str, Any]] = []
        for stratum in sorted(by_stratum):
            picked.extend(by_stratum[stratum][:per])
        queries = picked[:limit]
    elif limit:
        queries = queries[:limit]

    return [
        {
            "query": record["query"],
            "relevant_ids": record.get("relevant_ids", []),
            "ground_truth_answer": record.get("ground_truth_answer"),
            "stratum": record.get("stratum"),
        }
        for record in queries
    ]


__all__ = [
    "STRATA",
    "GenerationReport",
    "GoldQuery",
    "build_phrase_index",
    "generate_queries",
    "load_eval_cases",
    "validate_queries",
]
