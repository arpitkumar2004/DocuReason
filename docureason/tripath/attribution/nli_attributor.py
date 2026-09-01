"""nli_attributor.py — DEPRECATED shim.

The implementation moved to :mod:`docureason.tripath.attribution.claim_attributor` and the
class was renamed from ``NLIFaithfulnessAttributor`` to
:class:`~docureason.tripath.attribution.claim_attributor.ClaimSupportAttributor`, because
it performs lexical support checking plus numeric-consistency verification — not
natural-language inference.  The configured ``nli-deberta-v3-small`` model was
never loaded.

Both names remain importable from here for backward compatibility.
"""

from __future__ import annotations

from .claim_attributor import ClaimSupportAttributor, NLIFaithfulnessAttributor

__all__ = ["ClaimSupportAttributor", "NLIFaithfulnessAttributor"]
