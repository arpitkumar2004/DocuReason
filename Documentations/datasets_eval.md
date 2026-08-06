# DocuReason v1.1.0 — Multimodal Benchmark Dataset Evaluation Specification

> **The Authoritative Benchmark Dataset & Testing Specification Manual**  
> *A Comprehensive Reference Guide Detailing Dataset Contents, Subsystem Mappings, Mathematical Formulations, Automated Metric Counts, Kaggle GPU Time Estimates, and Framework Code Implementations.*

---

## Executive Summary & System Testing Paradigm

Evaluating an enterprise-grade multimodal Retrieval-Augmented Generation (RAG) framework requires testing against diverse, real-world document collections. Single-domain datasets (e.g., plain-text news articles) fail to stress-test complex RAG subsystems such as Text-to-SQL execution, OCR rendering, or chart figure understanding.

**DocuReason v1.1.0** features an automated benchmark dataset evaluation suite integrated into [`src/tripath/evaluation/benchmark_dataset.py`](file:///d:/DocuReason/src/tripath/evaluation/benchmark_dataset.py) and [`src/tripath/evaluation/eval_harness.py`](file:///d:/DocuReason/src/tripath/evaluation/eval_harness.py).

This specification details all **6 Primary Benchmark Datasets** that form the system testing suite, explaining:
1. **Dataset Contents & Modality Breakdown**
2. **Targeted Pipeline Subsystems & Processing Paths**
3. **Automated Metric Output Counts & Formulations**
4. **Estimated Execution Time & Memory Footprint on Kaggle GPUs**
5. **Framework Automated Code Implementation** (Zero manual parsing required in notebooks)

---

## Master Dataset Comparison Matrix

| Benchmark Dataset | Domain / Modality | Sample Count | Targeted Framework Subsystem / Path | Output Metric Count | Kaggle GPU Time (100 Sample Split) | Target SLA / Baseline Accuracy |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **1. FinQA** | Financial Reports (Table + Text) | 8,281 | **Table/Text-to-SQL Path** (`TableSQLRetriever` + DuckDB) | **10 Metrics** | **~3.5 Min** | $\ge 88.0\%$ Accuracy |
| **2. TAT-QA** | Hybrid Financial Reports | 16,552 | **Hybrid RRF Fusion Path** (`Fuser` + `ConfigurableRouter`) | **11 Metrics** | **~4.0 Min** | $\ge 85.0\%$ Accuracy |
| **3. DocVQA** | Scanned PDF Document Images | 50,000+ | **Ingestion & OCR Fallback** (`Docling` + `EasyOCR`) | **9 Metrics** | **~6.0 Min** | $\ge 82.0\%$ ANLS / Recall@5 |
| **4. ChartQA** | Visual Graphs & Infographics | 32,700 | **Vision / Chart Path** (`VisionRetrieval` + `BLIP-2`) | **8 Metrics** | **~4.5 Min** | $\ge 80.0\%$ Accuracy |
| **5. WikiTableQuestions** | General Tabular Data | 22,033 | **Table Serialization** (`TableSerializer` + GFM) | **8 Metrics** | **~3.0 Min** | $\ge 86.0\%$ TEDS / Exec Acc. |
| **6. DocLayNet** | Multi-Domain PDF Layouts | 80,863 | **Layout Parsing & Region Segmentation** (`DocLayNet`) | **7 Metrics** | **~5.0 Min** | $\ge 0.950$ Quality Score |

---

## Detailed Dataset Specifications & Subsystem Mapping

---

### 1. FinQA Dataset (Financial Reports Table + Text QA)

#### Dataset Description & Contents
- **Source**: Financial Annual Reports (SEC 10-K Filings of S&P 500 Companies).
- **Contents**: 8,281 complex multi-row financial tables, calculation steps, and surrounding narrative prose.
- **Challenge**: Requires multi-step numerical calculation (`SUM`, `AVG`, `PERCENTAGE_CHANGE`, `GROWTH_RATE`) across aligned table cells. Plain-text vector LLMs fail completely on this dataset due to vector arithmetic hallucinations.

#### Targeted Framework Subsystem & Processing Path
- **Primary Subsystem**: **Tabular / Text-to-SQL Path** ([`src/tripath/retrieval/table_sql.py`](file:///d:/DocuReason/src/tripath/retrieval/table_sql.py)).
- **Mechanism**: Extracts tabular regions, serializes to GFM Markdown and HTML/JSON schemas, synthesizes SQL queries, and executes aggregations in-memory via [DuckDB](https://duckdb.org/).

#### Automated Output Metrics (10 Metrics)
1. `sql_execution_success_rate` (1.0 = Clean DuckDB execution)
2. `teds_score` (Tree-Edit-Distance Similarity for table structure)
3. `cell_content_accuracy` (Exact string match across table grid)
4. `exact_match` (EM for numerical calculation answers)
5. `token_f1` (Token-level F1 score vs ground truth)
6. `recall_at_5` (Retrieval recall of target table chunk)
7. `ndcg_at_5` (Ranking order quality)
8. `attribution_precision` (Sentence NLI claim verification)
9. `latency_p90_ms` (90th percentile response time)
10. `est_cost_per_1k_queries_usd` (Estimated compute OpEx)

#### Kaggle GPU Performance Estimates
- **Kaggle Execution Time (100 Sample Split)**: **~3.5 Minutes**
- **VRAM Consumption**: $\approx 3.2 \text{ GB}$
- **RAM Footprint**: $\approx 2.4 \text{ GB}$

#### Automated Framework Implementation
Fully automated via `BenchmarkDataset().load_from_huggingface("fin_qa")` or `build_smoke_suite()`.

---

### 2. TAT-QA Dataset (Hybrid Financial Reports QA)

#### Dataset Description & Contents
- **Source**: Hybrid Financial Reports (FinTabNet / Corporate Annual Statements).
- **Contents**: 16,552 query examples where answers require retrieving evidence from **both** narrative paragraphs and structured data tables simultaneously.
- **Challenge**: Stress-tests multi-path candidate collection and late fusion score aggregation.

#### Targeted Framework Subsystem & Processing Path
- **Primary Subsystem**: **Hybrid RRF Fusion & Intent Router** ([`src/tripath/fusion/fuse.py`](file:///d:/DocuReason/src/tripath/fusion/fuse.py) & [`src/tripath/router/configurable_router.py`](file:///d:/DocuReason/src/tripath/router/configurable_router.py)).
- **Mechanism**: Soft probability routing dispatches parallel queries across Text (Dense bi-encoder + BM25S) and Table (DuckDB) streams, fusing hit lists using Weighted Reciprocal Rank Fusion ($k=60$).

#### Automated Output Metrics (11 Metrics)
1. `average_recall_at_5`
2. `average_recall_at_10`
3. `average_precision_at_5`
4. `average_ndcg_at_5`
5. `mrr` (Mean Reciprocal Rank)
6. `map_at_10` (Mean Average Precision)
7. `unique_evidence_ratio` (Evidence source diversity)
8. `context_redundancy_ratio` (Token overlap control)
9. `attribution_precision` (NLI DeBERTa claim checking)
10. `hallucination_rate` ($1.0 - \text{attribution\_precision}$)
11. `queries_per_second_qps` (System throughput)

#### Kaggle GPU Performance Estimates
- **Kaggle Execution Time (100 Sample Split)**: **~4.0 Minutes**
- **VRAM Consumption**: $\approx 3.8 \text{ GB}$
- **RAM Footprint**: $\approx 2.6 \text{ GB}$

#### Automated Framework Implementation
Fully automated via `BenchmarkDataset().load_from_huggingface("tat_qa")`.

---

### 3. DocVQA Dataset (Scanned Document Image QA)

#### Dataset Description & Contents
- **Source**: Industry Documents, Letters, Forms, and Scanned Reports.
- **Contents**: 50,000+ scanned PDF pages with visual noise, stamps, low-contrast text, and complex multi-column layouts.
- **Challenge**: Tests optical character recognition error recovery and reading order preservation on non-native, image-only documents.

#### Targeted Framework Subsystem & Processing Path
- **Primary Subsystem**: **Ingestion & OCR Fallback Subsystem** ([`src/tripath/ingestion/docling_wrapper.py`](file:///d:/DocuReason/src/tripath/ingestion/docling_wrapper.py) & [`src/tripath/ingestion/ocr_fallback.py`](file:///d:/DocuReason/src/tripath/ingestion/ocr_fallback.py)).
- **Mechanism**: Evaluates per-page character density ($\text{char\_count} < 50$). Automatically renders high-DPI page images and passes them to EasyOCR.

#### Automated Output Metrics (9 Metrics)
1. `artifact_quality_score` (Proportion of valid non-empty chunks)
2. `ocr_fallback_count` (Number of scanned pages EasyOCR triggered on)
3. `average_recall_at_5`
4. `hit_rate_at_5`
5. `average_ndcg_at_5`
6. `attribution_precision`
7. `latency_p50_ms`
8. `latency_p99_ms`
9. `peak_memory_mb`

#### Kaggle GPU Performance Estimates
- **Kaggle Execution Time (100 Sample Split)**: **~6.0 Minutes** (Includes GPU EasyOCR rendering)
- **VRAM Consumption**: $\approx 4.5 \text{ GB}$ (EasyOCR + PyTorch)
- **RAM Footprint**: $\approx 3.1 \text{ GB}$

#### Automated Framework Implementation
Fully automated via `DoclingWrapper(input_dir=...).ingest()` and `ArtifactQualityAuditor().audit_documents()`.

---

### 4. ChartQA Dataset (Visual Graphs & Infographics)

#### Dataset Description & Contents
- **Source**: Financial Infographics, Bar Charts, Line Plots, and Pie Graphs from Pew Research and Statista.
- **Contents**: 32,700 visual charts paired with analytical visual questions.
- **Challenge**: Tests visual region feature extraction without destroying graphic context into plain text.

#### Targeted Framework Subsystem & Processing Path
- **Primary Subsystem**: **Vision / Chart Path** ([`src/tripath/retrieval/vision_retrieval.py`](file:///d:/DocuReason/src/tripath/retrieval/vision_retrieval.py) & [`src/tripath/ingestion/figure_captioner.py`](file:///d:/DocuReason/src/tripath/ingestion/figure_captioner.py)).
- **Mechanism**: Uses ColPali visual embeddings + BLIP-2 figure captioning to index chart figures alongside surrounding text.

#### Automated Output Metrics (8 Metrics)
1. `figure_caption_preservation_rate`
2. `average_recall_at_5`
3. `hit_rate_at_5`
4. `average_ndcg_at_5`
5. `mrr`
6. `attribution_precision`
7. `hallucination_rate`
8. `mean_latency_ms`

#### Kaggle GPU Performance Estimates
- **Kaggle Execution Time (100 Sample Split)**: **~4.5 Minutes**
- **VRAM Consumption**: $\approx 4.2 \text{ GB}$
- **RAM Footprint**: $\approx 2.8 \text{ GB}$

#### Automated Framework Implementation
Fully automated via `BenchmarkDataset().load_from_huggingface("chartqa")`.

---

### 5. WikiTableQuestions Dataset (General Tabular SQL QA)

#### Dataset Description & Contents
- **Source**: Wikipedia Data Tables.
- **Contents**: 22,033 complex general-domain tables across sports, geography, history, and science.
- **Challenge**: Tests table schema serialization and DuckDB SQL execution across non-financial, highly diverse data schemas.

#### Targeted Framework Subsystem & Processing Path
- **Primary Subsystem**: **Table Serializer & SQL Execution** ([`src/tripath/ingestion/table_serializer.py`](file:///d:/DocuReason/src/tripath/ingestion/table_serializer.py)).
- **Mechanism**: Converts raw HTML/Markdown table grids into normalized GFM tables and JSON schema payloads `{"columns": [...], "rows": [[...]]}` for DuckDB Text-to-SQL.

#### Automated Output Metrics (8 Metrics)
1. `teds_structural_similarity`
2. `cell_content_accuracy`
3. `sql_execution_success_rate`
4. `exact_match` (EM)
5. `token_f1`
6. `recall_at_5`
7. `ndcg_at_5`
8. `latency_p90_ms`

#### Kaggle GPU Performance Estimates
- **Kaggle Execution Time (100 Sample Split)**: **~3.0 Minutes**
- **VRAM Consumption**: $\approx 3.0 \text{ GB}$
- **RAM Footprint**: $\approx 2.2 \text{ GB}$

#### Automated Framework Implementation
Fully automated via `BenchmarkDataset().load_from_huggingface("wikitablequestions")`.

---

### 6. DocLayNet Dataset (Document Layout Ingestion Benchmark)

#### Dataset Description & Contents
- **Source**: IBM Research Annotated PDF Repository.
- **Contents**: 80,863 gold-standard annotated PDF document pages across Financial, Legal, Scientific, and Government domains.
- **Challenge**: Measures layout region segmentation accuracy (Text, Table, Picture, Header, Footer) and reading order reconstruction.

#### Targeted Framework Subsystem & Processing Path
- **Primary Subsystem**: **Layout Parsing & Chunking Subsystem** ([`src/tripath/ingestion/docling_layout_parser.py`](file:///d:/DocuReason/src/tripath/ingestion/docling_layout_parser.py)).
- **Mechanism**: Evaluates TableFormer + DocLayNet region classification, character density, empty chunk ratios, and sentence split prevention.

#### Automated Output Metrics (7 Metrics)
1. `parsing_throughput_pages_per_sec`
2. `artifact_quality_score`
3. `duplicate_chunk_ratio`
4. `table_preservation_rate`
5. `heading_preservation_rate`
6. `cross_sentence_split_rate`
7. `memory_peak_mb`

#### Kaggle GPU Performance Estimates
- **Kaggle Execution Time (500 PDF Page Batch)**: **~5.0 Minutes**
- **VRAM Consumption**: $\approx 2.5 \text{ GB}$
- **RAM Footprint**: $\approx 2.4 \text{ GB}$

#### Automated Framework Implementation
Fully automated via `DoclingLayoutParser()` and `ArtifactQualityAuditor().audit_documents()`.

---

## Kaggle Execution & Python Code Integration Guide

### Zero Manual Parsing Required
The DocuReason framework **completely automates** dataset loading, preprocessing, query execution, and metric reporting. You do **not** need to write manual dataset parsers in your Kaggle Notebook.

### Complete Kaggle Notebook Execution Script

```python
# ==============================================================================
# DocuReason v1.1.0 — Automated Kaggle Benchmark Evaluation Notebook Cell
# ==============================================================================

# 1. Install framework in Kaggle environment
!pip install docureason-framework

import sys
from src.tripath.evaluation.benchmark_dataset import BenchmarkDataset
from src.tripath.evaluation.eval_harness import EvaluationHarness

print("=" * 70)
print("DocuReason v1.1.0 — Kaggle Multimodal Dataset Evaluation Suite")
print("=" * 70)

# 2. Build multi-modal benchmark suite (FinQA, TAT-QA, DocVQA, ChartQA, WikiTable)
benchmark_cases = BenchmarkDataset().build_extended_suite(total_items=100)

# 3. Define SLA Target Benchmarks
target_slas = {
    "average_recall_at_5": 0.850,
    "average_ndcg_at_5": 0.800,
    "attribution_precision": 0.900,
    "sql_execution_success_rate": 0.850,
    "latency_p90_ms": 1500.0,
}

# 4. Execute automated evaluation harness
harness = EvaluationHarness(output_dir="artifacts/kaggle_evaluation")
report = harness.evaluate_suite(benchmark_cases, target_metrics=target_slas)

# 5. Display Summary Results
print("\n" + "=" * 70)
print("EVALUATION RESULTS & TARGET METRICS VERIFICATION SUMMARY")
print("=" * 70)

for metric_name, info in report["target_verification"]["metric_breakdown"].items():
    print(f"{metric_name:<30} | Achieved: {info['achieved']:<8} | Target: {info['target']:<8} | Status: [{info['status']}]")

print("=" * 70)
print("OVERALL EVALUATION STATUS:", "ALL TARGET METRICS MET!" if report["target_verification"]["all_targets_met"] else "OPTIMIZATION REQUIRED")
print("Report Saved To:", harness.save(report, "kaggle_benchmark_report"))
```
