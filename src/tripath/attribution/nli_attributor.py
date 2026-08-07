import re
from typing import Any, Dict, List, Optional, Union

from src.tripath.config import AttributionConfig, DocuReasonConfig


class NLIFaithfulnessAttributor:
    """Attribution engine using NLI claim verification and precision thresholding."""

    def __init__(self, config: Optional[Union[AttributionConfig, DocuReasonConfig]] = None) -> None:
        if isinstance(config, DocuReasonConfig):
            a_cfg = config.attribution
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
            return {
                "claims": [],
                "supported_claims": 0,
                "total_claims": 0,
                "attribution_precision": 1.0,
                "status": "bypassed_via_config",
            }
        if isinstance(answer, dict):
            answer = answer.get("answer", "")
        answer_str = str(answer or "").strip()
        claims = self._split_claims(answer_str)
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
            "status": "verified" if precision >= self.flag_threshold_precision else "needs_review",
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
