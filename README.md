# DocuReason RAG

This repository contains a Phase 1 and Phase 2 prototype for multimodal document ingestion, chunking, retrieval, routing, and evaluation.

## What is implemented

- A lightweight ingestion pipeline over the sample documents in the samples folder
- Region segmentation for titles, body text, tables, and figures
- Section-aware chunking with token windows and overlap
- Hybrid retrieval across text, table, and vision modalities
- Configurable routing and ranking
- A simple evaluation harness and benchmark dataset structure
- A local dashboard that shows the full processing report

## Run the full pipeline

```bash
python scripts/run_phase_pipeline.py
```

This writes a report to:

- artifacts/test_run/pipeline_report.json

## Start the local dashboard

```bash
python scripts/serve_dashboard.py
```

Then open:

```text
http://127.0.0.1:8001
```

## Run tests

```bash
python -m pytest -q
```
