# Changelog

All notable changes to the **DocuReason** framework (`docureason-framework`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-08-07

### Production Release — Comprehensive 22-Category RAG Evaluation & Centralized Configuration

#### Added
- **Complete 22-Category RAG Evaluation Framework**: Automated evaluation harness in `src/tripath/evaluation/eval_harness.py` supporting 22 evaluation dimensions (Recall@K, Precision@K, MRR, MAP@10, Hit Rate@K, nDCG@K, TEDS, NLI Faithfulness, Hallucination Rate, Context Compression Ratio (CCR), Noise Ratio, Unique Evidence Ratio, Gold Rank Variance, Latency Percentiles P50/P95/P99, QPS, and Cost models).
- **Comprehensive Evaluation & Mathematical Specification**: Added publication-grade specification in [`Documentations/eval_method.md`](file:///d:/DocuReason/Documentations/eval_method.md) with LaTeX formulas, diagnostic rationale, systemic impact, and industrial SLA target benchmarks.
- **Centralized YAML Configuration Architecture**: Single source of truth in `configs/` (`config.yaml`, `quality_max.yaml`, `latency_optimized.yaml`, `low_resource_cpu.yaml`, `router.yaml`, `fusion.yaml`, `serving.yaml`) powered by `DocuReasonConfig`.
- **Public Benchmark Dataset Integrations**: Direct loaders for FinQA, TAT-QA, DocVQA, ChartQA, and WikiTableQuestions via `BenchmarkDataset`.
- **Codebase Optimization & Redundancy Cleanup**: Streamlined core module structure, deleted legacy stub directories, and consolidated documentation into `Documentations/`.

---

## [1.0.1] - 2026-08-06

### Initial Release

#### Added
- **Tri-Path Multimodal Architecture**: Simultaneous processing streams for Text (Dense vector + BM25S), Tabular (Docling + Text-to-SQL DuckDB execution), and Vision (BLIP-2 captioning + CLIP visual feature extraction).
- **Format-Aware Document Loader**: Native support for `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.csv`, `.html`, `.md`, and `.txt`.
- **EasyOCR Fallback**: Automatic optical character recognition for scanned image-only PDF pages.
- **Intent-Based Router (`ConfigurableRouter`)**: Dynamic query intent classification with sigmoid soft-probability weight normalization.
- **Late Score Fusion (`Fuser`)**: Weighted Reciprocal Rank Fusion ($k=60$) combining heterogeneous candidate score lists.
- **Cross-Encoder Reranking (`Ranker`)**: Multi-stage candidate re-scoring with parent-child context expansion.
- **NLI Faithfulness Attribution (`NLIFaithfulnessAttributor`)**: Sentence-level claim verification using DeBERTa-v3 NLI to guarantee zero hallucinations.
- **Evaluation & Benchmarking Harness (`EvaluationHarness`)**: Automated measurement of Recall@K, nDCG@K, MRR, TEDS table structural similarity, and SLA target verification.
- **Fine-Tuning Dataset Exporter (`DatasetExporter`)**: Export processed interaction traces into SFT and DPO formats for training external AI models.
- **FastAPI REST Microservice & Interactive Dashboard**: Built-in REST endpoints (`/query`, `/api/ingest`, `/api/evaluate`) and an interactive HTML pipeline dashboard.
