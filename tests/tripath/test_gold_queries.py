"""test_gold_queries.py — gold labels that are derivable, or not emitted at all.

The evaluation set this replaces had invented answers and gold ids naming fixtures that
no longer resolved (audit P0-4). The property that makes the replacement trustworthy is
narrow and testable: a query is emitted only when its anchor identifies exactly one
region in the whole corpus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from docureason.tripath.evaluation.gold_queries import (
    build_phrase_index,
    generate_queries,
    validate_queries,
)

pytestmark = pytest.mark.unit


def _region(
    region_id: str,
    region_type: str,
    text: str,
    table: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"id": region_id, "type": region_type, "text": text, "metadata": {}}
    if table is not None:
        payload["table_json"] = table
    return payload


def _corpus(regions: List[Dict[str, Any]], doc_id: str = "doc-a") -> Dict[str, Any]:
    return {"documents": [{"id": doc_id, "source": "a.html", "regions": regions}]}


UNIQUE_PROSE = (
    "The Americas segment recorded an unusual goodwill impairment tied to the Helios "
    "acquisition during the fourth quarter of the reporting period."
)
BOILERPLATE = "We are subject to risks and uncertainties that could affect our business and results of operations."


def test_a_table_cell_query_carries_the_cell_as_its_answer() -> None:
    table = {"columns": ["segment", "fiscal 2024"], "rows": [["Americas revenue", "15000"]]}
    report = generate_queries(_corpus([_region("doc-a-region-0", "table", "table", table)]), per_stratum=5)

    lookups = [q for q in report.queries if q.stratum == "tabular_lookup"]
    assert lookups, "a two-column table with one labelled row is enough to derive a lookup"
    query = lookups[0]
    assert query.relevant_ids == ["doc-a-region-0"]
    assert query.ground_truth_answer == "15000"
    assert query.anchor["row_label"] == "Americas revenue"
    assert query.anchor["column"] == "fiscal 2024"
    assert query.provenance == "programmatic"


def test_the_numeric_stratum_only_uses_cells_that_parse_as_numbers() -> None:
    table = {"columns": ["metric", "value"], "rows": [["Total assets", "not disclosed"], ["Total equity", "91000"]]}
    report = generate_queries(_corpus([_region("doc-a-region-0", "table", "t", table)]), per_stratum=5)

    numeric = [q for q in report.queries if q.stratum == "numeric"]
    assert numeric
    assert numeric[0].ground_truth_answer == "91000"


def test_a_table_with_no_usable_cell_is_rejected_with_a_reason() -> None:
    table = {"columns": ["a", "b"], "rows": [["1", "2"]]}
    report = generate_queries(_corpus([_region("doc-a-region-0", "table", "t", table)]), per_stratum=5)

    assert not [q for q in report.queries if q.stratum == "tabular_lookup"]
    assert report.rejected["tabular_lookup:no_cell"] == 1


def test_at_most_one_query_per_table() -> None:
    """Fifty questions about one balance sheet would make the stratum mean meaningless."""
    # Values wide enough to qualify: the lookup stratum skips 1-3 character numerics,
    # which carry no signal as an answer.
    rows = [[f"Line item {i}", "10,500", "20,750", "30,900"] for i in range(30)]
    table = {"columns": ["metric", "v1", "v2", "v3"], "rows": rows}
    report = generate_queries(_corpus([_region("doc-a-region-0", "table", "t", table)]), per_stratum=20)

    assert len([q for q in report.queries if q.stratum == "tabular_lookup"]) == 1


def test_a_prose_query_is_anchored_on_a_unique_phrase() -> None:
    report = generate_queries(_corpus([_region("doc-a-region-0", "body", UNIQUE_PROSE)]), per_stratum=5)

    prose = [q for q in report.queries if q.stratum == "prose"]
    assert prose
    assert prose[0].relevant_ids == ["doc-a-region-0"]
    assert prose[0].anchor["phrase"] in prose[0].query
    assert prose[0].ground_truth_answer is None, "retrieval gold, not an answer key"


def test_boilerplate_repeated_across_regions_cannot_be_gold_labelled() -> None:
    """Risk-factor and legal boilerplate repeats verbatim; no phrase identifies one region."""
    corpus = _corpus(
        [
            _region("doc-a-region-0", "body", BOILERPLATE),
            _region("doc-a-region-1", "body", BOILERPLATE),
            _region("doc-a-region-2", "body", BOILERPLATE),
        ]
    )
    report = generate_queries(corpus, per_stratum=5)

    assert not [q for q in report.queries if q.stratum == "prose"]
    assert report.rejected["prose:no_unique_phrase"] == 3


def test_a_region_too_short_to_anchor_is_rejected() -> None:
    report = generate_queries(_corpus([_region("doc-a-region-0", "body", "Revenue rose.")]), per_stratum=5)
    assert report.rejected["prose:too_short"] == 1


def test_an_empty_stratum_says_why() -> None:
    """Zero-by-composition must be distinguishable from zero-by-rejection."""
    report = generate_queries(_corpus([_region("doc-a-region-0", "body", UNIQUE_PROSE)]), per_stratum=5)
    assert report.rejected["figure:no_regions_in_corpus"] == 1


def test_generation_is_deterministic_for_a_seed() -> None:
    corpus = _corpus(
        [
            _region("doc-a-region-0", "body", UNIQUE_PROSE),
            _region("doc-a-region-1", "body", UNIQUE_PROSE.replace("Helios", "Aurora")),
        ]
    )
    first = [q.as_dict() for q in generate_queries(corpus, per_stratum=5, seed=3).queries]
    second = [q.as_dict() for q in generate_queries(corpus, per_stratum=5, seed=3).queries]
    assert first == second


def test_the_phrase_index_counts_each_region_once_per_shingle() -> None:
    corpus = _corpus([_region("doc-a-region-0", "body", "alpha beta gamma delta " * 3)])
    index = build_phrase_index(corpus, anchor_words=4)
    assert index["alpha beta gamma delta"] == ["doc-a-region-0"]


def test_validation_passes_on_a_freshly_generated_set() -> None:
    corpus = _corpus(
        [
            _region("doc-a-region-0", "body", UNIQUE_PROSE),
            _region("doc-a-region-1", "table", "t", {"columns": ["k", "v"], "rows": [["Total assets", "9100"]]}),
        ]
    )
    report = generate_queries(corpus, per_stratum=5, seed=0)
    result = validate_queries([q.as_dict() for q in report.queries], corpus)

    assert result["valid"] is True
    assert result["dangling_gold_ids"] == []


def test_validation_catches_a_gold_id_that_no_longer_resolves() -> None:
    """A re-ingest with a different digest invalidates every gold id; that must be loud."""
    corpus = _corpus([_region("doc-a-region-0", "body", UNIQUE_PROSE)])
    stale = [{"id": "q-0001", "relevant_ids": ["doc-a-region-999"], "anchor": {}}]
    result = validate_queries(stale, corpus)

    assert result["valid"] is False
    assert result["dangling_gold_ids"] == ["q-0001:doc-a-region-999"]


def test_validation_catches_an_anchor_that_stopped_being_unique() -> None:
    corpus = _corpus(
        [
            _region("doc-a-region-0", "body", UNIQUE_PROSE),
            _region("doc-a-region-1", "body", UNIQUE_PROSE),
        ]
    )
    duplicated = [
        {
            "id": "q-0001",
            "relevant_ids": ["doc-a-region-0"],
            "anchor": {"phrase": "americas segment recorded an unusual goodwill impairment"},
        }
    ]
    result = validate_queries(duplicated, corpus)

    assert result["valid"] is False
    assert result["non_unique_anchors"] == ["q-0001:2 owner(s)"]
