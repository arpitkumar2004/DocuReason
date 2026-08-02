# DocuReason RAG

This repository contains an initial Phase 1 implementation for a multimodal document ingestion and indexing pipeline.

## What is implemented

- A lightweight ingestion pipeline that loads text documents from the samples folder
- Region segmentation for titles, body text, tables, and figures
- A reproducible corpus/index export written to JSON artifacts
- A smoke-test suite that validates the pipeline output

## Run the sample pipeline

```bash
python -m docureason --input-dir samples --output-dir artifacts/phase1
```

The command writes:

- artifacts/phase1/corpus.json
- artifacts/phase1/index.json
- artifacts/phase1/quality_audit.json

## Run tests

```bash
python -m pytest -q
```
