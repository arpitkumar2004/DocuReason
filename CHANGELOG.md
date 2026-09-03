# Changelog

All notable changes to the **DocuReason** framework (`docureason-framework`) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-09-02

### Measurability Release — audit remediation (Sprint 2)

Continues [`Documentations/audit.md`](Documentations/audit.md) §13, Sprint 2. Sprint 1 made the architecture real; this sprint makes it **measurable** — a real corpus, a baseline to compare against, executed ablations, and intervals around every number.

#### Added — evaluation infrastructure
- **A real evaluation corpus from SEC EDGAR** (`docureason/tripath/evaluation/edgar.py`, `scripts/build_eval_corpus.py`). Everything scoreable before came from eight fixture files totalling ~3 KB with invented answers (`benchmark_dataset.py` asserted `"answer": "2.4M"` against a table containing 100M and 125M), so no retrieval number was falsifiable. The fetcher pulls 10-K primary documents with the SEC-required `User-Agent` (`DOCUREASON_SEC_USER_AGENT`, no default — requests are refused rather than sent under a fabricated identity), rate-limits itself well under the 10 req/s fair-access limit, caches by accession number, and writes `provenance.json` with each filing's URL, accession, filing date and SHA-256.
- **Stratified gold queries whose labels are derivable** (`docureason/tripath/evaluation/gold_queries.py`). Table-cell queries know their gold region by construction (row label × column header); prose and figure queries are anchored on a phrase **verified unique across the whole corpus**, and a candidate whose anchor is not unique is discarded rather than emitted — scoring a correct region as a miss would measure the label format, not retrieval. Rejections are counted by reason, and every record carries `provenance: "programmatic"`: these are derived labels, not human annotation, which over-represents lookup and cannot express multi-hop reasoning.
- **The naive-RAG baseline** (`docureason/tripath/evaluation/baselines.py`). The docs quoted improvements over "naive RAG" with no naive RAG in the repository, so nothing achieved could be attributed to the tri-path architecture rather than the corpus or the generator. Recursive character splitting, one flat exact index, top-5, **the same generator and the same pinned embedding model** — only the retrieval architecture differs. Chunks record every region they overlap, so a splitter that cuts across gold units is not penalised for the label format.
- **A real ablation runner** (`docureason/tripath/evaluation/ablation.py`). The module it replaces subtracted one pre-computed `average_recall_at_5` from another and had **zero callers**, while `Documentations/eval_method.md` §21 published seven deltas to one decimal place that nothing in the codebase could have produced. Each ablation now re-runs every query with the component disabled and reports a **paired** comparison: mean delta, bootstrap CI and a sign-flip p-value. Each row carries its published claim beside the measurement, and an ablation that could not be executed is reported `executed: false` with a reason — never as a delta of zero.
- **Five ablation switches that did not exist.** `retrieval.enable_sparse_retrieval`, `retrieval.enable_table_sql`, `reranker.enable_cross_encoder`, `reranker.enable_heading_penalty` and `ingestion.enable_layout_parser`. Four of the seven documented ablations were not operations the system supported: there was no way to turn those components off, so "remove X and re-measure" could only ever have been asserted. Each switch is also reported through the `degradation` channel, because an ablated response still means something different from a healthy one.
- **Bootstrap intervals and paired significance testing** (`docureason/tripath/evaluation/statistics.py`). Percentile bootstrap for a metric's mean, paired bootstrap for the difference between two systems, and an exact sign-flip permutation test (enumerated for n ≤ 14, sampled with an add-one correction above it, so a sampled p-value is never reported as 0). Paired because query-to-query variance dwarfs between-system variance. Pure Python against a seeded `random` — the CI gate needs determinism more than speed.
- **`confidence_intervals`, `stage_latency_ms` and `per_stratum` in every suite report.** A corpus-wide mean can hold steady while tabular lookup collapses and prose improves; the strata are the only place that shows.
- **Per-stage latency instrumentation** (`docureason/tripath/utils/timing.py`). Reported latency was one number measured *around* the call (audit P0-5), so an embedding reload, a cross-encoder cold start and a DuckDB retry were indistinguishable. `StageTimer` accumulates `route`, `dense_search`, `sparse_search`, `index_merge`, `corpus_scan`, `table_sql`, `fuse`, `chart_enrich`, `parent_expand`, `rerank`, `generate` and `attribute`, and reports a **signed residual** — total minus the sum of the stages — so a decomposition with a hole in it says so instead of quietly describing 60% of the request. A negative residual is the parallel dispatch overlapping, and is reported rather than clamped.
- **A metric regression gate in CI** (`scripts/check_metric_regression.py`, `tests/fixtures/`). 359 lines of metric arithmetic had no test that would notice a change. A frozen fixture — 20 cases across four strata, fixed retrieval output, no models, no network — is scored and compared against a committed baseline of 99 values; anything moving more than 0.01 fails the build. Latency and throughput are excluded because they measure the runner. It is a **blocking** gate: an advisory metric check reads as enforced and is worse than none (P3-6).
- **DuckDB tables registered at ingestion** (`docureason/tripath/indexing/table_store.py`). `TableSQLRetriever` re-created every table it touched on **every query** — one `CREATE TABLE` plus one `INSERT` per row into a throwaway `:memory:` database — and re-inferred column types each time, so the same column could be `DECIMAL` for one question and `VARCHAR` for the next. Tables are now registered once into `tables.duckdb` with `tables_index.json` beside the corpus, and queries run against a read-only connection. On the 8-filing corpus that is **1,952 tables / 17,310 rows** prepared once instead of per query. A region absent from the index falls back to the in-memory path, so a partially built store degrades rather than failing.
- **Parallel modality dispatch** (`serving.enable_parallel_paths`). The three retrieval paths are independent and now run concurrently, with results collected in a fixed task order so ranking is unchanged. A failing path no longer takes the query down: it returns empty, is logged, and surfaces as `mode: "failed"` in that modality's stats. The dense and sparse index builders gained load locks and a metadata cache — without them three threads could each load the embedding model, and `search()` re-read the whole metadata sidecar from disk on every query.
- **A bounded response cache** (`serving.response_cache_size`, default 64). Keyed on the query **and** a fingerprint of the effective config, so a per-query override or an ablation switch can never be served a response computed under the opposite setting. A hit is marked `cached: true` and its `latency_breakdown` describes the **cache lookup**, with the original run preserved under `latency_breakdown_first_run` — replaying the original timings would credit the pipeline with work it did not do and drag every reported percentile toward the cache's.
- **`ServingConfig`.** `configs/serving.yaml` existed but was never loaded (`load_from_yaml` reads `config.yaml` only), so both settings above were unreachable configuration for behaviour the service did not have.

#### Fixed
- **Per-query overrides mutated the shared service.** `apply_query_overrides` mutated `self` and returned it, so one request carrying `{"final_top_k": 1}` permanently narrowed the `QueryService` every later request used, and an ablation run left its own switches behind for the next one. It now returns a deep copy. Overrides are also driven by a single table shared with the REST layer and the ablation definitions, and an unknown key is recorded in `source_map["unknown_overrides"]` instead of silently doing nothing.
- **SEC filings were parsed without a layout parser.** `_build_regions` routed only `.pdf/.docx/.pptx/.xlsx` to Docling, so an HTML filing reached the rule-based segmenter, which saw raw markup and produced **no table regions at all**. `.html`/`.htm` are now layout-parsed: one Apple 10-K yields 62 tables and 1,107 body regions where it previously yielded neither.
- **Figures were captioned from the wrong image, or from a blank one** (P1-3, P1-4). The captioner returned `page.images[0]` for *every* figure on a page, so three charts on one page were all described from whichever image came first; when a page had no embedded image it built a **blank white canvas**, cropped it and captioned that, and the result was indexed as a figure. A new `PageRasterizer` (`pypdfium2`) renders the page and crops each figure at its own bbox, and a crop with no visual variation is refused and counted as `blank_crop_skipped`.
- **OCR read only the first embedded image per page** (P2-13), so a scan split across two images OCR'd to half its text — and a page with no embedded image had a blank canvas handed to EasyOCR, costing seconds per page to read nothing. Pages are now rendered and blank renders are skipped before OCR runs.
- **`image_rendering_dpi` had no consumer.** It was in the config, in `config.yaml` and in the startup banner, and reached no code (§6). It is now the rasterization DPI for both figure crops and OCR.
- **The scanned-page detector was off by one page** (P1-2). Character counts were accumulated under Docling's 1-based `page_no` and looked up under a 0-based page loop index, so page *N*'s text was compared against page *N+1*'s content. Conversion now happens in one place (`page_index_from_region`), and OCR regions are emitted 1-based to match Docling.
- **A pinned model revision turned a network hiccup into an empty index.** Pinning makes the loader resolve that commit against the hub; when the hub was unreachable but the commit was already cached, the load failed outright and the dense index silently degraded. Both loaders now retry with `local_files_only=True` before giving up.
- **Windows console logging lost every message containing an arrow or em dash.** `UnicodeEncodeError` inside `logging` printed a traceback instead of the line, so ingestion logs were unreadable exactly where they mattered. The stream handler is now reconfigured to UTF-8.
- **`bootstrap_ci` iteration count is a parameter.** The ablation runner computes intervals for eight systems and every stratum; at the default 10,000 resamples that dominated the sweep.

#### Measured, on 8 real 10-K filings
| | |
| :--- | ---: |
| filings (Apple, Microsoft, JPMorgan, J&J, Exxon, Walmart, Caterpillar, Pfizer) | 8 |
| regions | 20,966 |
| chunks | 24,919 |
| DuckDB tables registered / rows | 1,952 / 17,310 |
| gold queries (tabular_lookup / numeric / prose) | 150 (50/50/50) |
| gold-id and anchor-uniqueness validation | PASS |
| hermetic unit tests | 202 (was 95) |

#### Known gaps after this sprint
- **The `figure` stratum is empty**: Docling's HTML pipeline emits no figure regions for these filings, so the corpus cannot support figure queries. Recorded in the generation report as `figure:no_regions_in_corpus` rather than as a silent zero.
- **The corpus is 8 filings, not the 20–50 the roadmap calls for.** Ingestion is ~2.4 min/filing on CPU and gold ids only exist after it; `scripts/build_eval_corpus.py --filings 24` scales without code changes.
- **`no_layout_parser` cannot be measured from a single corpus.** It is ingestion-scoped, so it needs a second corpus built with `enable_layout_parser: false`; the runner reports it as not executed until one is supplied.
- Gold labels are **programmatically derived, not annotated**. Real NLI entailment (P0-2), authentication (S1) and the remaining P1 findings are unchanged.

### Architecture Release — audit remediation (Sprint 1)

Continues the remediation of [`Documentations/audit.md`](Documentations/audit.md) (§13, Sprint 1).
Sprint 0 deliberately changed no capability; this one does — the indexes the framework builds are now the indexes it queries.

#### Added — capability
- **The dense and sparse indexes are queried at retrieval time (P0-1).** `DenseIndexBuilder.search()` and `BM25SIndexBuilder.search()` were fully implemented with **zero callers**, so each modality "retriever" was an in-memory substring scan over every region of every document — `O(R)` per query, and wrong: the token `in` matched *inside* `financial` and the spurious hit was then multiplied by the 2.5× body field weight. New `retrieval/index_backed.py` holds what the three retrievers share (corpus lookups, rank-based dense/sparse RRF merge, cap resolution, fallback decision); the substring scan survives as an explicit fallback, reported as `retrieval_mode` on every result and in `last_stats["mode"]`.
- **`degradation: [...]` on every query response and in every ingestion report (T3).** Six fallback boundaries logged at `warning`/`debug` and were invisible in the output, so a run with a stub FAISS index, a Jaccard "reranker", rule-based layout parsing and template generation was **byte-shape identical** to a fully functioning one. Each boundary now names itself as a short `subsystem:reason` label.
- **`capabilities_used` in `quality_audit.json` (P0-9).** Replaces seven hard-coded `notes` strings that asserted Docling, TableFormer, EasyOCR, BLIP-2 and FAISS had all run, emitted regardless of what happened on the machine. Every field is now an observation: regions by layout source, caption tier counts, resolved chunk sizes, per-modality index `ntotal`, BM25 parameters, and the embedding model with its resolved revision.
- **`Dockerfile` + `.dockerignore` (P3-3).** Two-stage CPU-only image: dependencies resolved from `requirements.lock` (not the ranges in `pyproject.toml`), CPU torch from PyTorch's index instead of the CUDA wheels, non-root user, `/health` healthcheck, weights mounted at `/models` rather than baked in.
- **`requirements.lock` + `scripts/lock_requirements.py` (P3-4).** The fully pinned runtime closure — 90 packages — walked from `Requires-Dist` rather than dumped with `pip freeze`, so build and lint tooling stays out of it. `--check` reports staleness.
- **`docureason/tripath/model_pins.py` (P3-4).** Commit pins for model weights, keyed by **model name** rather than held as one scalar: `indexing.domain` swaps the embedding model, and a scalar pin would then hand one model's commit to a different model. Applied to the embedder, the cross-encoder and CLIP; `indexing.model_revision` / `reranker.model_revision` still override. Unpinned models are reported as `hub-default` instead of implying a pin.

#### Fixed
- **`top_k_text` / `top_k_table` / `top_k_vision` were config fields with no consumer (P0-3).** Every retriever returned every match and the cross-encoder reranked all of them. The caps are now resolved per modality in the retrievers and **re-applied inside the fuser**, so the candidate budget holds regardless of what a caller passes in.
- **The evaluated system was not the served system (P0-8a).** `serving/main.py` carried its own retrieval/generation implementation that ran the DuckDB SQL path, chart augmentation and parent expansion, while every evaluation script used `QueryService`, which ran none of them — and each script read `response["sql_results"]`, always got nothing, and silently fell back to fixture presence. `QueryService` is now the only implementation; the REST layer handles HTTP concerns and delegates.
- **Two importable copies of the same code (P0-8b, P3-2).** `src/tripath/` is gone; `docureason/tripath/` is the single import root, and the scripts put only the repository root on `sys.path`.
- **The pipeline emitted one chunk per region, so `SectionAwareChunker` was inert (P2-7, P2-8).** Parent expansion swapped a string for itself, indexed text carried no breadcrumb, and a long table's markdown ran past the embedding model's 256-token limit and was silently truncated. The chunker existed, was tested, and was called by two scripts but never by the pipeline; it is now wired with config-driven sizes.
- **A FAISS build failure wrote a zero-byte index (P0-1 related).** At query time that returns `[]`, which is indistinguishable from "no relevant documents". A build whose dependencies are present now raises `IndexBuildError`; only a *missing optional dependency* degrades, and it says so in `build_stats`.
- **`torch.set_num_threads(1)` was a process-wide side effect of importing the layout parser (P1-14).** Now scoped to the Docling call by a context manager that restores the previous thread counts.
- **Path traversal and arbitrary write through the request body (S2, S3).** `input_dir`/`output_dir` were joined onto the deployment root and used as-is, so `"../../.."` walked out of it and any absolute path ignored it: the query route would ingest any directory the process could read and the ingest route would create directories and write artifacts anywhere it could write. `resolve_within_root()` now requires a relative path that is still inside the root after resolution — which also rejects a symlink inside the root pointing out of it — and a rejected path is a `400`, not a `503`. **S1 is still open:** these routes remain unauthenticated, so containment is not authorization.
- **SQL identifier injection from a document's own table header (S4).** Column names come from parsed table content and were interpolated straight into `CREATE TABLE` and `SELECT` as quoted identifiers, so a header cell containing `"` closed the identifier and the rest of the cell was parsed as SQL — in a DuckDB session reachable from the unauthenticated `/api/query` route. `quote_identifier()` doubles the quote at every interpolation site. Cell *values* were already bound as parameters.

#### Testing
- **95 hermetic tests in the `unit` gate**, up from 66. New: golden-value nDCG/MAP/MRR asserted against hand-computed numbers (§10 #1), RRF scale-invariance and cap enforcement (#5), index build-then-search round-trip including the raise-instead-of-stub path (#6), pipeline wiring with a manifest-join assertion on every retrieved `region_id` (#4), 20 concurrent first requests asserting exactly one ingestion (#10), and the path-traversal and header-injection boundaries above (#11).
- `conftest.py` adds the `unit` / `integration` split (P1-21): unmarked tests default to `integration`, so the fast gate cannot silently absorb a slow one.

#### Known gaps after this sprint
- The `Dockerfile` has **not been built** — Docker is unavailable in the environment it was written in. Treat the first `docker build` as the verification step.
- `requirements.lock` was generated on `win32`; platform-specific transitive dependencies (the CUDA runtime wheels torch pulls in on linux) resolve at install time rather than being pinned. Regenerate on a target platform to pin those.
- Real NLI entailment (P0-2, numeric consistency only), a committed evaluation artifact (P0-9), and `mypy` in the local venv (P3-5) remain open, as do `P1-2`–`P1-11`, `P1-17`, `P1-19`, `P1-20` and all of Sprint 2.

## [1.1.4] - 2026-09-01

### Correctness & Accuracy Release — audit remediation (Sprint 0)

Driven by the full-repository code audit in [`Documentations/audit.md`](Documentations/audit.md).
This release fixes defects and removes unsupported claims; it adds no new capability.

#### Fixed — correctness
- **Short-heading rerank penalty inverted its own purpose.** Cross-encoder outputs are unbounded *signed* logits, so the multiplicative penalty (`× 0.5`) moved a negative score toward zero and **promoted** the table-of-contents fragments it was written to suppress. Replaced with a sign-safe subtractive penalty (`RerankerConfig.heading_penalty_offset`, default `2.0`). The multiplier is retained for the non-negative fallback scorer.
- **Pipeline wrote indexes to the wrong directory.** `DocuReasonPipeline(output_dir=X)` wrote `corpus.json`/`index.json`/`manifest.json` to `X` while FAISS and BM25S indexes went to `config.output_dir` (default `artifacts/`). Indexes never lived beside the corpus they indexed, and two runs with different output directories overwrote each other's index files.
- **Retrieval metrics matched gold IDs by bidirectional substring.** `"doc_1"` matched `"doc_10"` (false positive) and `"finqa_doc_01"` matched no `"doc-finqa_doc-<digest>"` (false negative, pinning every retrieval metric on the extended benchmark suite to `0`). Matching is now **exact** on `region_id`/`document_id`/`id`, with a new `resolve_gold_ids()` helper mapping readable fixture stems to canonical corpus IDs and **logging** any that do not resolve.
- **Claim verification approved contradicted figures.** Lexical overlap alone marked `"Q2 revenue was $8M"` as supported by evidence reading `"$12M"`. A numeric-consistency check now runs alongside the overlap test.
- **A disabled verification gate reported perfect faithfulness.** `enable_nli: false` returned `attribution_precision = 1.0`, which could satisfy an SLA threshold. It now returns `None` and the harness emits no attribution metric.
- **Document IDs were not content-addressed.** The digest covered only the file path, so an edited document kept its ID and re-ingestion never refreshed the index. The digest now covers content **and** path, widened from 12 to 16 hex characters (48 → 64 bits of collision space).
- **Three retrievers invented their own region IDs** (`-region-{i+1}`, `-table-{i}`, `-table-{i+1}`), so retrieval payloads did not join back to `manifest.json` and one physical region appeared as two fused candidates. All retrievers now resolve the canonical ID via `canonical_region_id()`.
- **BM25 `k1`/`b` were read from config and never applied** — `bm25s.BM25()` was constructed with no arguments, so config and preset changes had no effect.
- **OCR used a hard-coded threshold of 20** while config, YAML, the startup banner and the docs all said 50; `enable_ocr_fallback` was never checked. Both are now honoured.
- **`manifest.json` timestamp was the literal string `"."`** (`Path().as_posix()`). Replaced with a full provenance block: UTC timestamp, framework version, Python version, platform and installed ML package versions.
- **`except BaseException`** in the generation module (3 sites) and the layout parser swallowed `KeyboardInterrupt`/`SystemExit`. Narrowed to `Exception`; `MemoryError` handling is unaffected.
- **Missing runtime dependencies.** `faiss-cpu`, `pypdf` and `pyyaml` are imported by the code but were **not declared**, so a clean `pip install docureason-framework` could not build a dense index, read a PDF, or load YAML config. Now declared, with upper version bounds added to all heavy ML dependencies.
- **Broken console script.** `pyproject.toml` pointed at `src.tripath.cli:main`, a module that does not exist, so `docureason` failed for every installed user. Now `docureason.__main__:main`, and CI installs the built wheel into a clean venv and runs it.

#### Changed — honest naming (deprecated aliases retained, nothing breaks)
| Old | New | Reason |
| :--- | :--- | :--- |
| `NLIFaithfulnessAttributor` | `ClaimSupportAttributor` | Performs lexical containment + numeric consistency; the configured NLI model was never loaded |
| `MLflowTracker` | `JsonMetricWriter` | Writes a JSON file; has no MLflow dependency and makes no MLflow calls |
| `AsyncQueryService` | `CompositeQueryService` | Contains no `async`/`await` |
| `RouterTrainer` | `RouterVocabularyBuilder` | Performs no training — accumulates token counts over 6 hand-written queries |
| `teds_structural_similarity` | `teds_proxy_score` | Not tree-edit distance; a column-overlap + row-count surrogate |

`src/tripath/attribution/nli_attributor.py` and `src/tripath/evaluation/mlflow_tracker.py` remain importable as deprecation shims.

#### Changed — artifacts and observability
- `quality_audit.json` no longer contains a hard-coded `notes` list asserting that Docling, EasyOCR, BLIP-2 and FAISS all ran. It now reports **observed** `capabilities_used`: region counts by `layout_source` (`docling`/`rule`/`ocr`), structured tables actually serialized, resolved OCR settings, dense index model and type, and the BM25 parameters in force.
- Reranked results carry `reranker_mode` (`cross_encoder` / `token_similarity_fallback`) and `heading_penalty_applied`, so a silent model-unavailable fallback is now visible to callers.
- Attribution reports carry `method`, declaring the verification mechanism used.
- Evaluation callers (`scripts/evaluate_system.py`, `scripts/generate_large_evaluation.py`, `BenchmarkRunner`) now time the pipeline call and pass `latency_ms`. Previously the harness timed **its own metric computation**, which is why reports showed sub-millisecond p50 and four-figure QPS.
- The version is defined once in `src/tripath/version.py` and read by all runtime components; `tests/test_version_consistency.py` asserts it matches `pyproject.toml`.

#### Removed
- **BREAKING (minor):** `QueryService.query()` no longer returns an `embeddings` key. It held non-semantic hash vectors (one dimension per unique token, variable length) recomputed over the **entire corpus on every query** and never consumed. `ChunkEmbedder` is deprecated and now warns.
- `colpali-engine`, `qdrant-client`, `peft` and `accelerate` removed from core dependencies — all four were declared and never imported.
- The README benchmark comparison table. Its figures were not reproducible from this repository and no baseline implementation existed. Replaced with an [Evaluation Status](README.md#evaluation-status) section stating what is and is not measured.
- `Development Status :: 5 - Production/Stable` → `4 - Beta`.

#### Added
- 33 tests across 6 new modules, all hermetic and running in under a second except the pipeline-layout suite: `test_ranker_penalty.py`, `test_claim_attribution.py`, `test_identity.py`, `test_metric_id_matching.py`, `test_pipeline_output_dir.py`, `test_version_consistency.py`. Five of these fail against v1.1.3.
- CI: HuggingFace model cache (tests construct real models, so every matrix leg previously re-downloaded them) and a clean-venv wheel install + console-script smoke step.
- `Documentations/audit.md` — full repository audit: 59 findings with locations, fixes, a configuration coverage table, a dead-code inventory, and a four-sprint roadmap.

---

## [1.1.3] - 2026-08-23

### Maintenance & Release Update — Version Bump v1.1.3

#### Added & Fixed
- **Version Alignment (`v1.1.3`)**: Synchronized package version to `1.1.3` across `pyproject.toml`, `docureason/__init__.py`, `src/tripath/__init__.py`, REST API services, configuration transparency logs, test suites, and documentation.
- **Pipeline Walkthrough Assets & Documentation**: Integrated 4-step pipeline walkthrough diagrams, transparency summary outputs, and benchmark performance figures in `README.md`.

---

## [1.1.2] - 2026-08-07

### Maintenance Release — Packaging & Wheel Validation Fixes

#### Fixed
- **Wheel Content Structure Validation (`check-wheel-contents.cfg`)**: Added repository root configuration for `check-wheel-contents` (`ignore = W005, W009`) to resolve multi-toplevel packaging warnings (`src/` and `docureason/`) during PyPI wheel distribution audits.
- **PyPI Release Update**: Bumped version to `1.1.2` across framework packages, configuration files, and documentation.

---

## [1.1.1] - 2026-08-07

### Production Release — Enterprise CI/CD Pipeline & PyPI Release Engineering

#### Added
- **GitHub Actions CI Pipeline (`.github/workflows/ci.yml`)**: Automated multi-stage CI featuring linting (`ruff`), static typing (`mypy`), vulnerability auditing (`pip-audit`), multi-python test matrix (**Python 3.10, 3.11, 3.12** via `pytest`), and PyPI package build verification (`build` + `twine check --strict`).
- **PyPI Release Pipeline (`.github/workflows/release-pypi.yml`)**: Production deployment pipeline supporting PyPI **OIDC Trusted Publishing** (`pypa/gh-action-pypi-publish@release/v1`) with automated GitHub Release binary artifact attachments (`.tar.gz`, `.whl`).
- **TestPyPI Staging Pipeline (`.github/workflows/testpypi-publish.yml`)**: Pre-release staging workflow for deploying and verifying package releases on TestPyPI.
- **Automated Dependabot Updates (`.github/dependabot.yml`)**: Weekly tracking for Python package and GitHub Actions updates.
- **Packaging Extras (`pyproject.toml`)**: Modular optional dependency extras (`dev`, `test`, `lint`, `build`) for easy developer environment bootstrap via `pip install -e ".[dev]"`.
- **Local Packaging Verification Utility (`scripts/verify_pypi_package.py`)**: Local verification script to build `.tar.gz` and `.whl` distributions, validate metadata strictness with `twine check`, and test importing installed wheel packages in isolated sandboxes.
- **CI/CD Architectural Specification (`Documentations/cicd_pipeline.md`)**: Comprehensive release engineering documentation detailing workflow triggers, OIDC security model, matrix testing strategy, and maintainer release checklists.

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
