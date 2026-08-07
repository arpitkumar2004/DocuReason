from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from src.tripath.config import DocuReasonConfig, GenerationConfig
from src.tripath.utils import get_logger, trace_execution

logger = get_logger(__name__)


class GenerationModule:
    """Production-grade multi-modal SLM & LLM generation module."""

    def __init__(
        self,
        backend: Optional[str] = None,
        model_name_or_path: Optional[str] = None,
        max_context_tokens: int = 4096,
        truncation_strategy: str = "smart_relevance",
        config: Optional[Union[GenerationConfig, DocuReasonConfig]] = None,
    ) -> None:
        if isinstance(config, DocuReasonConfig):
            g_cfg = config.generation
        elif isinstance(config, GenerationConfig):
            g_cfg = config
        else:
            g_cfg = None

        env_backend = os.getenv("DOCUREASON_GEN_BACKEND")
        if g_cfg:
            self.backend = env_backend if env_backend else (backend or g_cfg.backend)
            self.model_name_or_path = model_name_or_path or g_cfg.model_name_or_path
            self.max_context_tokens = g_cfg.max_context_tokens
            self.truncation_strategy = g_cfg.truncation_strategy
            self.temperature = g_cfg.temperature
            self.max_new_tokens = g_cfg.max_new_tokens
        else:
            self.backend = env_backend if env_backend else (backend or "auto")
            self.model_name_or_path = model_name_or_path or "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
            self.max_context_tokens = max_context_tokens
            self.truncation_strategy = truncation_strategy
            self.temperature = 0.1
            self.max_new_tokens = 512

        self._slm_pipeline: Any = None
        self._slm_loaded: bool = False

    @staticmethod
    def _is_gpu_available() -> bool:
        """Check if PyTorch is available with an active CUDA GPU."""
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _load_slm_pipeline_if_available(self) -> Any:
        if self._slm_loaded:
            return self._slm_pipeline
        self._slm_loaded = True
        if self.backend == "template" or os.getenv("DOCUREASON_DISABLE_SLM") == "1":
            return None

        # Check GPU availability
        gpu_available = self._is_gpu_available()

        # In 'auto' mode without CUDA GPU, skip loading heavy SLM models on CPU
        if self.backend == "auto" and not gpu_available:
            logger.info("GPU unavailable (CUDA disabled/absent). Skipping local SLM loading on CPU; fallback engine will generate text.")
            return None

        try:
            from transformers import pipeline
            logger.info("Loading local SLM model for generation: %s (device_map=auto)", self.model_name_or_path)
            self._slm_pipeline = pipeline(
                "text-generation",
                model=self.model_name_or_path,
                device_map="auto",
                torch_dtype="auto",
                max_new_tokens=512,
            )
        except BaseException as exc:
            logger.warning("Local SLM pipeline load skipped/failed: %s. Using fallback generation.", exc)
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
                "device": "none",
                "fallback_used": False,
            }

        # 1. Package Multi-Modal Context with Context Token Budget Management
        context_prompt, citations = self._build_context_prompt(query, evidence, sql_results)
        gpu_active = self._is_gpu_available()

        # 2. Multi-tier generation with automatic GPU -> Cloud -> Offline Fallback
        if self.backend in ("slm", "auto"):
            slm = self._load_slm_pipeline_if_available()
            if slm is not None:
                try:
                    result = self._generate_with_slm(slm, query, context_prompt, citations)
                    result["device"] = "cuda" if gpu_active else "cpu"
                    result["fallback_used"] = False
                    return result
                except BaseException as exc:
                    logger.warning("Local SLM generation failed (%s). Triggering fallback generation.", exc)

        # Fallback Level 1: Cloud Gemini LLM API
        if self.backend in ("cloud", "auto") and os.getenv("GEMINI_API_KEY"):
            try:
                result = self._generate_with_gemini(query, context_prompt, citations)
                result["device"] = "cloud"
                result["fallback_used"] = not gpu_active
                result["fallback_reason"] = "GPU unavailable; routed to Cloud Gemini API" if not gpu_active else None
                return result
            except Exception as exc:
                logger.warning("Cloud LLM generation failed (%s). Proceeding to offline fallback.", exc)

        # Fallback Level 2: Offline Multi-Modal Synthesizer Engine (100% CPU Uptime)
        result = self._synthesize_offline_template(query, evidence, sql_results, citations)
        result["device"] = "cpu"
        result["fallback_used"] = True
        result["fallback_reason"] = "GPU unavailable or SLM offline; generated text using offline multi-modal synthesizer fallback."
        return result

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 characters per token heuristic)."""
        return max(1, len(text) // 4)

    def _build_context_prompt(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        sql_results: Optional[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """Format multi-modal evidence with token budget allocation & smart relevance truncation."""
        context_blocks = []
        citations: List[Dict[str, str]] = []
        accumulated_tokens = 0
        max_budget = self.max_context_tokens - self._estimate_tokens(query) - 300  # reserve prompt headroom

        # Priority 1: DuckDB Table SQL Execution Block (Exact numerical math)
        if sql_results and sql_results.get("executed"):
            sql_query = sql_results.get("sql_query", "N/A")
            rows = sql_results.get("sql_results", [])
            sql_block = f"--- DUCKDB TEXT-TO-SQL RESULT ---\nSQL Query: {sql_query}\nRows: {rows}"
            sql_tokens = self._estimate_tokens(sql_block)
            context_blocks.append(sql_block)
            accumulated_tokens += sql_tokens
            citations.append({"type": "table_sql", "source": "DuckDB Text-to-SQL Engine", "query": sql_query})

        # Priority 2: Multi-modal Retrieval Candidates (Ranked by Cross-Encoder / RRF)
        truncated_count = 0
        for idx, item in enumerate(evidence, start=1):
            modality = item.get("modality", "text")
            doc_id = item.get("document_id") or item.get("id") or f"Doc-{idx}"
            text_content = item.get("text") or item.get("linearized") or ""

            block_text = f"--- EVIDENCE ITEM {idx} [{modality.upper()}] (Source: {doc_id}) ---\n{text_content}"
            block_tokens = self._estimate_tokens(block_text)

            if accumulated_tokens + block_tokens <= max_budget:
                context_blocks.append(block_text)
                accumulated_tokens += block_tokens
                citations.append({"type": modality, "source": str(doc_id), "chunk_id": item.get("chunk_id", f"c-{idx}")})
            else:
                # Apply smart relevance truncation to remaining budget
                remaining_tokens = max(0, max_budget - accumulated_tokens)
                if remaining_tokens > 50:
                    max_chars = remaining_tokens * 4
                    truncated_text = text_content[:max_chars].rsplit(" ", 1)[0] + f"... [truncated (max_context_tokens={self.max_context_tokens})]"
                    trunc_block = f"--- EVIDENCE ITEM {idx} [{modality.upper()}] (Source: {doc_id}) ---\n{truncated_text}"
                    context_blocks.append(trunc_block)
                    accumulated_tokens += self._estimate_tokens(trunc_block)
                    citations.append({"type": modality, "source": str(doc_id), "chunk_id": item.get("chunk_id", f"c-{idx}")})
                    truncated_count += 1
                break

        logger.info(
            "Context Budget Manager: packaged %d blocks (~%d tokens / limit %d, truncated=%d)",
            len(context_blocks), accumulated_tokens, self.max_context_tokens, truncated_count
        )

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
        except BaseException as exc:
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

    @staticmethod
    def _clean_snippet(text: str, max_chars: int = 250) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        cut = text[:max_chars].rsplit(" ", 1)[0]
        if not cut.endswith((".", "!", "?")):
            cut += "..."
        return cut

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
            lines.append(f"Based on the structured database query, **{val_summary}**.\n")
        elif evidence:
            top_text = evidence[0].get("text") or evidence[0].get("linearized") or ""
            clean_top = self._clean_snippet(top_text)
            lines.append(f"Based on the retrieved document context:\n\n> {clean_top}\n")
        else:
            lines.append("No conclusive information was found in the indexed documents.\n")

        # Key Findings & Evidence Highlights
        if evidence:
            lines.append("### Key Findings & Multi-Modal Evidence\n")
            for idx, item in enumerate(evidence[:4], start=1):
                source = item.get("document_id") or item.get("id") or f"doc_{idx}"
                modality = item.get("modality", "text").upper()
                text_snippet = self._clean_snippet(item.get("text") or item.get("linearized") or "", max_chars=220)
                lines.append(f"* **[{modality}]** {text_snippet} [Source: {source}]")

        # Tabular Summary if SQL executed
        if sql_results and sql_results.get("executed") and sql_results.get("sql_results"):
            rows = sql_results["sql_results"]
            if rows:
                cols = list(rows[0].keys())
                table_md = ["\n### Structured Table Summary\n", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
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
