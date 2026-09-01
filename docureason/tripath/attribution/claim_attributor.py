"""claim_attributor.py — sentence-level claim support checking.

.. warning::
   This module performs **lexical support checking**, not natural-language
   inference.  A claim is marked supported when its token set is sufficiently
   contained in an evidence item's token set.  That is fast and deterministic,
   but it cannot detect numeric substitution or negation on its own, so a
   numeric-consistency check is applied alongside it (see
   :meth:`ClaimSupportAttributor._numeric_conflict`).

   The class was previously named ``NLIFaithfulnessAttributor`` and configured
   with ``cross-encoder/nli-deberta-v3-small``, which it never loaded.  The name
   now matches the mechanism.  Loading a real NLI model is tracked as future
   work; when it lands, ``AttributionConfig.entailment_threshold`` becomes live.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Union

from docureason.tripath.config import AttributionConfig, DocuReasonConfig

# Numbers with optional sign, thousands separators, decimals and a magnitude suffix.
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*\s*(?:%|k|m|bn|b|thousand|million|billion)?", re.IGNORECASE)


class ClaimSupportAttributor:
    """Verify each sentence of an answer against the retrieved evidence.

    Two independent checks must both pass for a claim to count as supported:

    1. **Lexical containment** — at least ``overlap_threshold`` of the claim's
       tokens appear in the evidence item.
    2. **Numeric consistency** — every number in the claim also appears in the
       evidence item.  Without this, ``"Q2 revenue was $8M"`` was marked
       supported by evidence reading ``"$12M"`` because the words overlapped.
    """

    #: Fraction of claim tokens that must appear in an evidence item.
    overlap_threshold: float = 0.2

    def __init__(self, config: Optional[Union[AttributionConfig, DocuReasonConfig]] = None) -> None:
        if isinstance(config, DocuReasonConfig):
            a_cfg: Optional[AttributionConfig] = config.attribution
        elif isinstance(config, AttributionConfig):
            a_cfg = config
        else:
            a_cfg = None

        if a_cfg:
            self.model_name = a_cfg.nli_model_name
            self.entailment_threshold = a_cfg.entailment_threshold
            self.flag_threshold_precision = a_cfg.flag_threshold_precision
            self.enable_nli = a_cfg.enable_nli
        else:
            self.model_name = "cross-encoder/nli-deberta-v3-small"
            self.entailment_threshold = 0.5
            self.flag_threshold_precision = 0.5
            self.enable_nli = True

    def attribute(self, answer: str | Dict[str, Any], evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.enable_nli:
            # Report *no measurement* rather than perfect faithfulness.  Returning
            # 1.0 here previously let a disabled safety gate pass an SLA check.
            return {
                "claims": [],
                "supported_claims": 0,
                "total_claims": 0,
                "attribution_precision": None,
                "status": "bypassed_via_config",
                "method": "disabled",
            }

        if isinstance(answer, dict):
            answer = answer.get("answer", "")
        answer_str = str(answer or "").strip()
        claims = self._split_claims(answer_str)

        verified_claims: List[Dict[str, Any]] = []
        for claim in claims:
            supporting_evidence: List[Dict[str, Any]] = []
            conflicting = False
            for item in evidence:
                text = str(item.get("text", ""))
                if not self._is_lexically_supported(claim, text):
                    continue
                if self._numeric_conflict(claim, text):
                    conflicting = True
                    continue
                supporting_evidence.append(item)

            verified_claims.append(
                {
                    "claim": claim,
                    "supported": bool(supporting_evidence),
                    "numeric_conflict": conflicting and not supporting_evidence,
                    "supporting_evidence": supporting_evidence[:2],
                }
            )

        supported_count = sum(1 for item in verified_claims if item["supported"])
        precision = round(supported_count / max(1, len(verified_claims)), 3)
        return {
            "claims": verified_claims,
            "supported_claims": supported_count,
            "total_claims": len(verified_claims),
            "attribution_precision": precision,
            "status": "verified" if precision >= self.flag_threshold_precision else "needs_review",
            "method": "lexical_containment+numeric_consistency",
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _split_claims(self, answer: str) -> List[str]:
        cleaned = re.split(r"(?<!\w)\.(?!\w)", answer)
        claims = [segment.strip() for segment in cleaned if segment.strip()]
        return claims or [answer.strip()]

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {token.lower() for token in re.findall(r"[a-zA-Z0-9]+", text) if token}

    def _is_lexically_supported(self, claim: str, evidence: str) -> bool:
        claim_terms = self._tokens(claim)
        if not claim_terms:
            return False
        overlap = claim_terms & self._tokens(evidence)
        return bool(overlap) and (len(overlap) / len(claim_terms)) >= self.overlap_threshold

    @staticmethod
    def _numbers(text: str) -> Set[str]:
        """Normalized numeric literals found in *text* (separators and case stripped)."""
        found: Set[str] = set()
        for match in _NUMBER_RE.finditer(text):
            token = match.group(0).replace(",", "").replace(" ", "").lower()
            if any(ch.isdigit() for ch in token):
                found.add(token)
        return found

    def _numeric_conflict(self, claim: str, evidence: str) -> bool:
        """True when the claim asserts a number the evidence does not contain."""
        claim_numbers = self._numbers(claim)
        if not claim_numbers:
            return False
        return not claim_numbers.issubset(self._numbers(evidence))


#: Deprecated alias.  The original name asserted an NLI model that was never loaded.
NLIFaithfulnessAttributor = ClaimSupportAttributor

__all__ = ["ClaimSupportAttributor", "NLIFaithfulnessAttributor"]
