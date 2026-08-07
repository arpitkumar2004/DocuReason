from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from src.tripath.config import DocuReasonConfig, RerankerConfig
from src.tripath.utils import get_logger, trace_execution

logger = get_logger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_GLOBAL_CROSS_ENCODER_CACHE: Dict[str, Any] = {}


class Ranker:
    """Cross-Encoder reranker with breadcrumb evaluation and short-heading penalty."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[Union[RerankerConfig, DocuReasonConfig]] = None,
    ) -> None:
        if isinstance(config, DocuReasonConfig):
            r_cfg = config.reranker
        elif isinstance(config, RerankerConfig):
            r_cfg = config
        else:
            r_cfg = None

        if r_cfg:
            self.model_name = model_name or r_cfg.model_name
            self.short_heading_threshold = r_cfg.short_heading_char_threshold
            self.heading_penalty_multiplier = r_cfg.heading_penalty_multiplier
        else:
            self.model_name = model_name or _CROSS_ENCODER_MODEL
            self.short_heading_threshold = 45
            self.heading_penalty_multiplier = 0.5

    def _get_cross_encoder(self) -> Optional[Any]:
        if self.model_name in _GLOBAL_CROSS_ENCODER_CACHE:
            return _GLOBAL_CROSS_ENCODER_CACHE[self.model_name]

        try:
            from sentence_transformers import CrossEncoder
            encoder = CrossEncoder(self.model_name)
            _GLOBAL_CROSS_ENCODER_CACHE[self.model_name] = encoder
            logger.info("Loaded and cached process-wide CrossEncoder model: %s", self.model_name)
            return encoder
        except Exception as exc:
            logger.warning("CrossEncoder model unavailable (%s) — using token similarity reranker fallback", exc)
            return None

    @trace_execution(logger=logger)
    def rank(
        self,
        query_or_results: str | List[Dict[str, Any]],
        results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate results against query using CrossEncoder or fallback."""
        if isinstance(query_or_results, list):
            query = ""
            target_results = query_or_results
        else:
            query = query_or_results
            target_results = results or []

        if not target_results:
            return []

        encoder = self._get_cross_encoder()
        ranked: List[Dict[str, Any]] = []

        if encoder is not None:
            try:
                # Use enriched text (including section breadcrumbs) for CrossEncoder prediction
                pairs = [[query, item.get("text", "")] for item in target_results]
                scores = encoder.predict(pairs)

                for idx, item in enumerate(target_results):
                    cross_score = float(scores[idx])

                    # Apply penalty to standalone short heading chunks (< 45 body chars)
                    body_text = item.get("text", "")
                    if "[Context:" in body_text:
                        body_text = body_text.split("]", 1)[-1].strip()

                    if len(body_text) < self.short_heading_threshold and item.get("modality") == "text":
                        cross_score *= self.heading_penalty_multiplier  # Penalize short standalone headings

                    ranked_item = dict(item)
                    ranked_item["cross_encoder_score"] = round(cross_score, 4)
                    ranked_item["rank_score"] = round(cross_score, 4)
                    ranked.append(ranked_item)

                return sorted(ranked, key=lambda entry: entry["rank_score"], reverse=True)
            except Exception as exc:
                logger.warning("CrossEncoder prediction error: %s — falling back", exc)

        # Fallback Reranker: Token Jaccard + Heading Penalty + Modality Weighted Score
        query_tokens = set(query.lower().split()) if query else set()
        for item in target_results:
            text = item.get("text", "").lower()
            text_tokens = set(text.split())
            if query_tokens and text_tokens:
                intersection = query_tokens.intersection(text_tokens)
                union = query_tokens.union(text_tokens)
                jaccard = len(intersection) / max(1, len(union))
            else:
                jaccard = 0.0

            modality_weight = {"text": 1.0, "table": 1.3, "vision": 1.1}.get(item.get("modality"), 1.0)
            base_score = float(item.get("score", 0.0))

            body_text = item.get("text", "")
            if "[Context:" in body_text:
                body_text = body_text.split("]", 1)[-1].strip()
            heading_penalty = self.heading_penalty_multiplier if (len(body_text) < self.short_heading_threshold and item.get("modality") == "text") else 1.0

            fused_rank_score = ((base_score * 0.7) + (jaccard * 100.0 * 0.3 * modality_weight)) * heading_penalty

            ranked_item = dict(item)
            ranked_item["rank_score"] = round(fused_rank_score, 3)
            ranked.append(ranked_item)

        return sorted(ranked, key=lambda entry: entry["rank_score"], reverse=True)
