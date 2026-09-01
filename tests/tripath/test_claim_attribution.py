"""test_claim_attribution.py — support checking must reject numeric substitution.

The attributor is lexical, so token overlap alone marked ``"Q2 revenue was $8M"``
as supported by evidence reading ``"$12M"`` — the exact failure the component
exists to catch.  A numeric-consistency check now runs alongside the overlap test.
"""

from __future__ import annotations

import pytest

from docureason.tripath.attribution.claim_attributor import ClaimSupportAttributor, NLIFaithfulnessAttributor
from docureason.tripath.config import AttributionConfig

pytestmark = pytest.mark.unit


def test_numeric_substitution_is_not_supported():
    attributor = ClaimSupportAttributor()
    report = attributor.attribute(
        "Q2 revenue was $8M",
        [{"text": "Quarter Q2 Region Europe Revenue $12M"}],
    )
    claim = report["claims"][0]
    assert claim["supported"] is False, "a contradicted figure must not count as supported"
    assert claim["numeric_conflict"] is True
    assert report["attribution_precision"] == 0.0


def test_matching_figure_is_supported():
    attributor = ClaimSupportAttributor()
    report = attributor.attribute(
        "Q2 revenue was $12M",
        [{"text": "Quarter Q2 Region Europe Revenue $12M"}],
    )
    assert report["claims"][0]["supported"] is True
    assert report["attribution_precision"] == 1.0


def test_claim_without_numbers_falls_back_to_lexical_support():
    attributor = ClaimSupportAttributor()
    report = attributor.attribute(
        "Europe led all regions",
        [{"text": "Quarter Q2 Region Europe Revenue 12"}],
    )
    # No numbers in the claim, so only the overlap test applies.
    assert report["claims"][0]["numeric_conflict"] is False


def test_disabled_check_reports_no_measurement_not_perfection():
    """A switched-off gate must not be able to satisfy an SLA threshold."""
    config = AttributionConfig(enable_nli=False)
    report = ClaimSupportAttributor(config=config).attribute("anything", [{"text": "unrelated"}])
    assert report["attribution_precision"] is None
    assert report["status"] == "bypassed_via_config"


def test_method_is_declared():
    report = ClaimSupportAttributor().attribute("Revenue grew", [{"text": "Revenue grew"}])
    assert report["method"] == "lexical_containment+numeric_consistency"


def test_deprecated_alias_still_importable():
    assert NLIFaithfulnessAttributor is ClaimSupportAttributor
