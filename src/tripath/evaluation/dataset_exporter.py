"""src/tripath/evaluation/dataset_exporter.py — Export DocuReason corpora & evaluations into model training datasets.

Exports:
1. Retriever Fine-Tuning Triplets: (query, positive_doc, negative_doc) for SentenceTransformers / ColPali.
2. Cross-Encoder Reranker Training Pairs: (query, document, relevance_score).
3. LLM SFT & DPO Training Datasets: (prompt, grounded_response, citations).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union
from src.tripath.ingestion.schema import Document


class DatasetExporter:
    """Exports indexed documents and evaluation queries into standard ML model training formats."""

    def __init__(self, output_dir: Union[str, Path] = "artifacts/training_data") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_retriever_triplets(
        self, eval_cases: List[Dict[str, Any]], filename: str = "train_retriever_triplets.jsonl"
    ) -> Path:
        """Exports (query, positive_doc, negative_doc) for fine-tuning Dense / ColPali embedding models."""
        out_path = self.output_dir / filename
        records = []

        for case in eval_cases:
            query = case.get("query", "")
            results = case.get("results", [])
            relevant_ids = case.get("relevant_ids", [])

            positives = [r for r in results if any(rel in (r.get("document_id") or r.get("id") or "") for rel in relevant_ids)]
            negatives = [r for r in results if not any(rel in (r.get("document_id") or r.get("id") or "") for rel in relevant_ids)]

            if positives and negatives:
                records.append({
                    "query": query,
                    "positive": positives[0].get("text", ""),
                    "negative": negatives[0].get("text", ""),
                    "positive_id": positives[0].get("document_id") or positives[0].get("id"),
                    "negative_id": negatives[0].get("document_id") or negatives[0].get("id"),
                })

        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        return out_path

    def export_reranker_pairs(
        self, eval_cases: List[Dict[str, Any]], filename: str = "train_reranker_pairs.jsonl"
    ) -> Path:
        """Exports (query, document_text, label) for training Cross-Encoder rerankers."""
        out_path = self.output_dir / filename
        records = []

        for case in eval_cases:
            query = case.get("query", "")
            results = case.get("results", [])
            relevant_ids = case.get("relevant_ids", [])

            for item in results[:10]:
                doc_id = str(item.get("document_id") or item.get("id") or "")
                is_rel = any(rel in doc_id or doc_id in rel for rel in relevant_ids)
                records.append({
                    "query": query,
                    "document": item.get("text", ""),
                    "label": 1.0 if is_rel else 0.0,
                    "document_id": doc_id,
                })

        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        return out_path

    def export_llm_sft_dataset(
        self, eval_cases: List[Dict[str, Any]], filename: str = "train_llm_sft.jsonl"
    ) -> Path:
        """Exports (instruction, input_context, output_answer) for Supervised Fine-Tuning (SFT) of SLMs/LLMs."""
        out_path = self.output_dir / filename
        records = []

        for case in eval_cases:
            query = case.get("query", "")
            evidence = case.get("evidence", [])
            answer = case.get("answer", "")

            context_str = "\n\n".join([f"[{e.get('modality', 'text').upper()}] {e.get('text', '')}" for e in evidence[:3]])
            records.append({
                "instruction": "Synthesize a grounded answer using the provided multi-modal evidence context.",
                "input": f"Query: {query}\n\nEvidence Context:\n{context_str}",
                "output": answer,
            })

        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        return out_path
