from __future__ import annotations

import re
from typing import List, Dict, Any


class NLIFaithfulnessAttributor:
    """A lightweight attribution engine that uses heuristic entailment checks over evidence."""

    def attribute(self, answer: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        claims = self._split_claims(answer)
        verified_claims: List[Dict[str, Any]] = []
        for claim in claims:
            supported = False
            supporting_evidence = []
            for item in evidence:
                text = str(item.get("text", ""))
                if self._is_entailed(claim, text):
                    supported = True
                    supporting_evidence.append(item)
            verified_claims.append({
                "claim": claim,
                "supported": supported,
                "supporting_evidence": supporting_evidence[:2],
            })

        supported_count = sum(1 for item in verified_claims if item["supported"])
        precision = round(supported_count / max(1, len(verified_claims)), 3)
        return {
            "claims": verified_claims,
            "supported_claims": supported_count,
            "total_claims": len(verified_claims),
            "attribution_precision": precision,
            "status": "verified" if precision >= 0.5 else "needs_review",
        }

    def _split_claims(self, answer: str) -> List[str]:
        cleaned = re.split(r"(?<!\w)\.(?!\w)", answer)
        claims = [segment.strip() for segment in cleaned if segment.strip()]
        return claims or [answer.strip()]

    def _is_entailed(self, claim: str, evidence: str) -> bool:
        claim_terms = {token.lower() for token in re.findall(r"[a-zA-Z0-9]+", claim) if token}
        evidence_terms = {token.lower() for token in re.findall(r"[a-zA-Z0-9]+", evidence) if token}
        if not claim_terms:
            return False
        overlap = claim_terms & evidence_terms
        return len(overlap) >= 1 and len(overlap) / len(claim_terms) >= 0.2
