from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from src.tripath.utils import get_logger, trace_execution

logger = get_logger(__name__)


class GenerationModule:
    """Production-grade multi-modal SLM & LLM generation module.

    Supports:
    1. Local SLM Engine ('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen2.5-1.5B-Instruct', or local path)
    2. Cloud LLM Engine (Gemini 2.5 / 1.5 Flash API)
    3. Multi-modal Offline Template Synthesizer (100% reliable fallback)
    """

    def __init__(
        self,
        backend: str = "auto",
        model_name_or_path: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ) -> None:
        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self._slm_pipeline: Any = None
        self._slm_loaded: bool = False

    def _load_slm_pipeline_if_available(self) -> Any:
        if self._slm_loaded:
            return self._slm_pipeline
        self._slm_loaded = True
        try:
            from transformers import pipeline
            logger.info("Loading local SLM model for generation: %s", self.model_name_or_path)
            self._slm_pipeline = pipeline(
                "text-generation",
                model=self.model_name_or_path,
                device_map="auto",
                torch_dtype="auto",
                max_new_tokens=512,
            )
        except Exception as exc:
            logger.warning("Local SLM pipeline load skipped/failed: %s", exc)
            self._slm_pipeline = None
        return self._slm_pipeline

    @trace_execution(logger=logger)
    def generate(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        sql_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate human-manner grounded response with citations and optional DeepSeek reasoning chain."""
        if not evidence and not sql_results:
            return {
                "answer": "No relevant context or evidence found to answer the query.",
                "reasoning_chain": None,
                "citations": [],
                "engine": "none",
            }

        # 1. Package Multi-Modal Context
        context_prompt, citations = self._build_context_prompt(query, evidence, sql_results)

        # 2. Determine execution engine
        if self.backend in ("slm", "auto"):
            slm = self._load_slm_pipeline_if_available()
            if slm is not None:
                return self._generate_with_slm(slm, query, context_prompt, citations)

        if self.backend in ("cloud", "auto") and os.getenv("GEMINI_API_KEY"):
            try:
                return self._generate_with_gemini(query, context_prompt, citations)
            except Exception as exc:
                logger.warning("Gemini API generation failed: %s", exc)

        # 3. Fallback: Multi-Modal Template Synthesizer Engine
        return self._synthesize_offline_template(query, evidence, sql_results, citations)

    def _build_context_prompt(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        sql_results: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Format qualitative text, DuckDB SQL execution tables, and visual chart evidence into a clean prompt."""
        context_blocks = []
        citations: List[Dict[str, str]] = []

        # DuckDB Table SQL Execution Block
        if sql_results and sql_results.get("executed"):
            sql_query = sql_results.get("sql_query", "N/A")
            rows = sql_results.get("sql_results", [])
            context_blocks.append(f"--- DUCKDB TEXT-TO-SQL RESULT ---\nSQL Query: {sql_query}\nRows: {rows}")
            citations.append({"type": "table_sql", "source": "DuckDB Text-to-SQL Engine", "query": sql_query})

        # Multi-modal Retrieval Candidates
        for idx, item in enumerate(evidence[:6], start=1):
            modality = item.get("modality", "text")
            doc_id = item.get("document_id") or item.get("id") or f"Doc-{idx}"
            text_content = item.get("text") or item.get("linearized") or ""
            context_blocks.append(f"--- EVIDENCE ITEM {idx} [{modality.upper()}] (Source: {doc_id}) ---\n{text_content}")
            citations.append({"type": modality, "source": str(doc_id), "chunk_id": item.get("chunk_id", f"c-{idx}")})

        joined_context = "\n\n".join(context_blocks)
        prompt = (
            f"You are an enterprise technical and financial analyst. Answer the user question based strictly on the provided context.\n\n"
            f"CONTEXT:\n{joined_context}\n\n"
            f"QUESTION: {query}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Provide a direct 1-2 sentence executive answer.\n"
            f"2. Use bullet points for key metrics and insights.\n"
            f"3. Include inline citations like [Source: DocName].\n"
        )
        return prompt, citations

    def _generate_with_slm(
        self, slm_pipe: Any, query: str, prompt: str, citations: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Execute local SLM (e.g. DeepSeek-R1-Distill-Qwen-1.5B) and extract <think> reasoning chain."""
        try:
            output = slm_pipe(prompt)
            raw_text = output[0]["generated_text"] if isinstance(output, list) and output else str(output)
            reasoning, clean_answer = self._extract_reasoning_chain(raw_text)
            return {
                "answer": clean_answer,
                "reasoning_chain": reasoning,
                "citations": citations,
                "engine": "local_slm_deepseek_r1",
            }
        except Exception as exc:
            logger.warning("SLM generation error: %s", exc)
            return self._synthesize_offline_template(query, [], None, citations)

    def _generate_with_gemini(
        self, query: str, prompt: str, citations: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Execute Cloud Gemini LLM API."""
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        reasoning, clean_answer = self._extract_reasoning_chain(response.text)
        return {
            "answer": clean_answer,
            "reasoning_chain": reasoning,
            "citations": citations,
            "engine": "cloud_gemini_api",
        }

    def _synthesize_offline_template(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        sql_results: Optional[Dict[str, Any]],
        citations: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Deterministic Multi-Modal Synthesizer Fallback for 100% offline uptime."""
        lines = []

        # Executive Summary Answer
        if sql_results and sql_results.get("executed") and sql_results.get("sql_results"):
            first_row = sql_results["sql_results"][0]
            val_summary = ", ".join([f"{k}: {v}" for k, v in first_row.items()])
            lines.append(f"Based on the structured database query, **{val_summary}**.")
        elif evidence:
            top_text = evidence[0].get("text") or evidence[0].get("linearized") or ""
            lines.append(f"Based on the retrieved document context: {top_text.strip()[:300]}")
        else:
            lines.append("No conclusive information was found in the indexed documents.")

        # Key Findings & Evidence Highlights
        lines.append("\n### Key Findings & Multi-Modal Evidence")
        for idx, item in enumerate(evidence[:4], start=1):
            source = item.get("document_id") or item.get("id") or f"doc_{idx}"
            modality = item.get("modality", "text").upper()
            text_snippet = (item.get("text") or item.get("linearized") or "").strip()[:200]
            lines.append(f"* **[{modality}]** {text_snippet} `[Source: {source}]`")

        # Tabular Summary if SQL executed
        if sql_results and sql_results.get("executed") and sql_results.get("sql_results"):
            rows = sql_results["sql_results"]
            if rows:
                cols = list(rows[0].keys())
                table_md = ["\n### Structured Table Summary", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
                for r in rows[:5]:
                    table_md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
                lines.append("\n".join(table_md))

        return {
            "answer": "\n".join(lines),
            "reasoning_chain": "Deterministic multi-modal context aggregation and template synthesis.",
            "citations": citations,
            "engine": "offline_template_synthesizer",
        }

    @staticmethod
    def _extract_reasoning_chain(text: str) -> Tuple[Optional[str], str]:
        """Extract DeepSeek-R1 <think> reasoning steps if present."""
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            clean_answer = text.replace(match.group(0), "").strip()
            return reasoning, clean_answer
        return None, text.strip()
