#!/usr/bin/env python3
"""scripts/build_kaggle_eval_notebook.py - emit the full-pipeline Kaggle notebook.

    python scripts/build_kaggle_eval_notebook.py

Writes ``notebooks/docureason_full_eval_kaggle.ipynb``: clone this repository at a ref,
install it, build the SEC EDGAR evaluation corpus, score the full system, execute the
seven-row ablation matrix against the naive-RAG baseline, run the metric-regression gate,
and export every artifact - corpus, FAISS/BM25 indexes, DuckDB table store, model
weights cache, reports - as a reusable Kaggle dataset.

Why a builder instead of a committed ``.ipynb``: ``notebooks/`` is gitignored, notebook
JSON reviews badly, and the cells are mostly thin calls into ``scripts/`` so the repo
stays the single source of truth for what the pipeline does. Edit here, regenerate.

The notebook is deliberately **resumable**. A first run on a T4 spends most of its time
in ingestion and embedding; attaching the previous run's output as a dataset skips both,
so a second run measures instead of rebuilding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_OUTPUT = Path("notebooks/docureason_full_eval_kaggle.ipynb")
REPO_URL = "https://github.com/arpitkumar2004/DocuReason.git"

CELLS: List[Tuple[str, str]] = []


def md(source: str) -> None:
    CELLS.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    CELLS.append(("code", source.strip("\n")))


md(
    r"""
# DocuReason — full pipeline, measured end to end

Runs the **whole** DocuReason evaluation chain from the GitHub source: real SEC EDGAR
filings in, a scored ablation matrix out. Nothing here is a fixture and nothing is
asserted — every number this notebook prints is computed in this session.

| Stage | What runs | First run (T4) |
| :-- | :--- | :-- |
| 1–4 | Config, environment audit, clone + install, import audit | ~4 min |
| 5 | Artifact-cache restore (skipped on a first run) | seconds |
| 6 | **Smoke gate** — one query end to end over `samples/` | ~2 min |
| 7 | EDGAR fetch → ingest → derive gold queries → validate | ~45–90 min |
| 8 | Second corpus, layout parser off (the ingestion-scoped ablation row) | ~15 min |
| 9 | Full system scored over every gold query | ~15 min |
| 10 | Seven ablations + naive-RAG baseline, paired with CIs | ~60–120 min |
| 11 | Metric-regression gate | seconds |
| 12 | Results tables | seconds |
| 13 | Seven figures (PNG + PDF) and a computed insight digest | ~1 min |
| 14 | Export corpus, indexes, weights cache and reports as a reusable dataset | ~3 min |

**Before you run:**

1. **Settings → Accelerator → GPU** (T4 or P100). CPU works, but TableFormer table
   structure and the NLI path are CUDA-gated, and ingestion is roughly 4× slower.
2. **Settings → Internet → On.** Required for the clone, PyPI, the HuggingFace hub and
   EDGAR.
3. **Edit `SEC_USER_AGENT` in the next cell.** The SEC requires a real contact string in
   the `User-Agent` header, and `edgar.py` refuses to send a request without one rather
   than inventing an identity. Requests are rate-limited well under the 10 req/s
   fair-access limit.

**To make the second run cheap:** when this finishes, *Save Version → Save & Run All*,
then in a new session attach this notebook's output under
`/kaggle/input/docureason-artifacts`. Cell 5 restores the corpus, the indexes, the
DuckDB table store and the model-weight cache, and cells 7–8 become no-ops.
"""
)

code(
    r"""
# ============================================================================
# CELL 1 - CONFIGURATION. This is the only cell you need to edit.
# ============================================================================
import os
from pathlib import Path

# --- required ---------------------------------------------------------------
# The SEC requires a real identity. Format: "Org Name contact@example.com".
SEC_USER_AGENT = "DocuReason Research your-email@example.com"

# --- source ----------------------------------------------------------------
REPO_URL = "https://github.com/arpitkumar2004/DocuReason.git"
REPO_REF = "main"          # branch, tag or commit sha

# --- evaluation scale ------------------------------------------------------
FILINGS = 24               # 10-K filings to fetch (roadmap asks for 20-50)
PER_COMPANY = 2
PER_STRATUM = 60           # target gold queries per stratum
SEED = 0
BOOTSTRAP_ITERATIONS = 2000

EVAL_LIMIT = 0             # queries for the full-system suite (0 = all)
ABLATION_LIMIT = 0         # queries per ablation row (0 = all; this is the long pole)
ABLATION_STRATIFIED = True # if you set a limit, sample it evenly across strata

RUN_NO_LAYOUT_CORPUS = True  # build corpus #2 so the 7th ablation row can execute
RUN_ABLATIONS = True
RUN_REGRESSION_GATE = True

# --- caching ---------------------------------------------------------------
# Attach a previous run's output here to skip fetch, ingestion and embedding.
CACHE_INPUT = Path("/kaggle/input/docureason-artifacts")

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "DocuReason"
EXPORT_DIR = WORK / "docureason_artifacts"
HF_CACHE = WORK / "hf_cache"

# Set before anything imports transformers: the weights must land somewhere that
# survives the session, or every rerun re-downloads them.
HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(HF_CACHE / "sentence-transformers")
os.environ["DOCUREASON_SEC_USER_AGENT"] = SEC_USER_AGENT
os.environ["TOKENIZERS_PARALLELISM"] = "false"

assert "your-email@example.com" not in SEC_USER_AGENT, (
    "Set SEC_USER_AGENT to a real contact string before running: the SEC's fair-access "
    "policy requires it, and edgar.py refuses to send a request without one."
)
print("Configuration accepted.")
print(f"  repo        : {REPO_URL} @ {REPO_REF}")
print(f"  filings     : {FILINGS}   per stratum: {PER_STRATUM}   seed: {SEED}")
print(f"  cache input : {CACHE_INPUT}  (exists: {CACHE_INPUT.exists()})")
print(f"  export to   : {EXPORT_DIR}")
"""
)


code(
    r"""
# ============================================================================
# CELL 2 - ENVIRONMENT AUDIT + the command runner every later cell uses
# ============================================================================
import json
import shutil
import subprocess
import sys
import time


def run(cmd, cwd=None, check=True, label=None):
    # Streams output as it arrives: an ingestion step that prints nothing for 40 minutes
    # is indistinguishable from a hung one.
    printable = " ".join(str(part) for part in cmd)
    print(f"$ {printable}\n")
    started = time.perf_counter()
    process = subprocess.Popen(
        [str(part) for part in cmd],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    captured = []
    for line in process.stdout:
        captured.append(line)
        print(line.rstrip())
    process.wait()
    minutes = (time.perf_counter() - started) / 60.0
    print(f"\n[exit {process.returncode} after {minutes:.1f} min]")
    if check and process.returncode != 0:
        raise RuntimeError(f"{label or printable} failed with exit code {process.returncode}")
    return "".join(captured)


print("=" * 78)
print("ENVIRONMENT")
print("=" * 78)
print(f"python      : {sys.version.split()[0]}")

try:
    import torch

    cuda = torch.cuda.is_available()
    print(f"torch       : {torch.__version__}  cuda={cuda}")
    if cuda:
        properties = torch.cuda.get_device_properties(0)
        print(f"gpu         : {properties.name}  {properties.total_memory / 1024 ** 3:.1f} GB")
    else:
        print("gpu         : none - TableFormer structure recognition and the NLI path are")
        print("              CUDA-gated, so this run will report them as degraded, honestly.")
except ImportError:
    print("torch       : not importable yet")

try:
    import psutil

    print(f"ram         : {psutil.virtual_memory().total / 1024 ** 3:.1f} GB")
except ImportError:
    pass

usage = shutil.disk_usage("/kaggle/working")
print(f"disk (work) : {usage.free / 1024 ** 3:.1f} GB free of {usage.total / 1024 ** 3:.1f} GB")

# Internet is needed for the clone, PyPI, the HF hub and EDGAR. Fail here, not in step 7.
probe = subprocess.run(
    [sys.executable, "-c", "import urllib.request;urllib.request.urlopen('https://pypi.org', timeout=15)"],
    capture_output=True,
)
print(f"internet    : {'reachable' if probe.returncode == 0 else 'UNREACHABLE - enable Settings > Internet'}")
if probe.returncode != 0:
    raise SystemExit("Internet is off. Every remaining cell needs it.")
print("=" * 78)
"""
)

code(
    r"""
# ============================================================================
# CELL 3 - CLONE THE REPOSITORY AND INSTALL IT
# ============================================================================
# The framework on PyPI is 1.1.3; the evaluation machinery this notebook drives lives in
# the repository. So: clone a ref, install from source, and print the sha that produced
# every number below.
if REPO_DIR.exists():
    print(f"{REPO_DIR} already present - reusing it (delete it to re-clone).")
else:
    run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(REPO_DIR)], label="git clone")

commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=str(REPO_DIR), capture_output=True, text=True
).stdout.strip()
print(f"\nrepository sha: {commit}")

os.chdir(REPO_DIR)  # configs/config.yaml is resolved relative to the working directory
print(f"cwd: {Path.cwd()}")

# Pin whatever torch stack Kaggle already has, so resolving docling's dependency tree
# cannot swap the CUDA build for a CPU wheel and silently cost us the GPU.
constraints = WORK / "constraints.txt"
pins = []
for module in ("torch", "torchvision", "torchaudio", "transformers"):
    try:
        version = __import__("importlib.metadata", fromlist=["version"]).version(module)
        pins.append(f"{module}=={version}")
    except Exception:
        continue
constraints.write_text("\n".join(pins) + "\n", encoding="utf-8")
print("constraints:", pins)

run([sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps"], label="pip install -e .")
run(
    [
        sys.executable, "-m", "pip", "install", "-q", "-c", str(constraints),
        "duckdb", "bm25s", "docling", "easyocr", "faiss-cpu", "pypdf", "pypdfium2",
        "sentence-transformers", "pyyaml", "fastapi", "uvicorn", "pydantic",
    ],
    label="pip install runtime dependencies",
)

import importlib

import torch

importlib.invalidate_caches()
print(f"\ntorch after install: {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("NOTE: no CUDA now. If it was available in Cell 2, the install replaced the")
    print("      torch build - restart the session and pin torch in the constraints list.")
"""
)


code(
    r"""
# ============================================================================
# CELL 4 - IMPORT AUDIT: what is available decides what the run can claim
# ============================================================================
import docureason

print(f"docureason {docureason.__version__}  (sha {commit[:10]})\n")

MODULES = [
    ("docling", "layout parsing, table structure, reading order"),
    ("duckdb", "the SQL path over serialized tables"),
    ("bm25s", "the sparse retrieval path"),
    ("faiss", "the dense index"),
    ("sentence_transformers", "the embedding model and the cross-encoder"),
    ("pypdfium2", "page rasterization for figure crops and OCR"),
    ("pypdf", "PDF text and page counts"),
    ("easyocr", "OCR fallback for scanned pages"),
]
available = {}
for name, purpose in MODULES:
    try:
        __import__(name)
        available[name] = True
        print(f"  OK      {name:24s} {purpose}")
    except Exception as exc:
        available[name] = False
        print(f"  MISSING {name:24s} {purpose}  ({type(exc).__name__})")

missing_core = [name for name in ("docling", "duckdb", "bm25s", "faiss") if not available[name]]
if missing_core:
    raise SystemExit(f"Core dependencies missing: {missing_core}. Re-run Cell 3.")
print("\nEvery capability this run reports will be an observation, not a claim:")
print("quality_audit.json records what actually ran, and each response carries a")
print("degradation list naming any fallback that fired.")
"""
)

code(
    r"""
# ============================================================================
# CELL 5 - RESTORE FROM A PREVIOUS RUN'S OUTPUT (skip on a first run)
# ============================================================================
# What is worth carrying between sessions: the fetched filings, the ingested corpus with
# its FAISS/BM25 indexes and DuckDB store, the derived gold queries, and the model
# weights. Ingestion and embedding are the expensive parts; measuring is cheap.
RESTORED = {"filings": False, "corpus": False, "queries": False, "corpus_no_layout": False, "hf_cache": False}


def _restore(source: Path, destination: Path, key: str) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    already = any(destination.iterdir()) if destination.is_dir() else destination.exists()
    if already:
        # A rerun of this cell must not overwrite work the session has already done.
        print(f"  keeping existing {destination}")
        RESTORED[key] = True
        return
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    RESTORED[key] = True
    print(f"  restored {key:18s} <- {source}")


if CACHE_INPUT.exists():
    print(f"Restoring from {CACHE_INPUT}")
    _restore(CACHE_INPUT / "data" / "eval" / "filings", REPO_DIR / "data" / "eval" / "filings", "filings")
    _restore(CACHE_INPUT / "data" / "eval" / "corpus", REPO_DIR / "data" / "eval" / "corpus", "corpus")
    _restore(CACHE_INPUT / "data" / "eval" / "queries.json", REPO_DIR / "data" / "eval" / "queries.json", "queries")
    _restore(
        CACHE_INPUT / "data" / "eval" / "corpus_no_layout",
        REPO_DIR / "data" / "eval" / "corpus_no_layout",
        "corpus_no_layout",
    )
    if (CACHE_INPUT / "hf_cache").exists() and not any(HF_CACHE.iterdir()):
        shutil.copytree(CACHE_INPUT / "hf_cache", HF_CACHE, dirs_exist_ok=True)
        RESTORED["hf_cache"] = True
        print("  restored hf_cache          <- model weights, no re-download")
else:
    print(f"No cache attached at {CACHE_INPUT} - this is a full first run.")

print("\nrestored:", {key: value for key, value in RESTORED.items() if value} or "nothing")
"""
)

code(
    r"""
# ============================================================================
# CELL 6 - SMOKE GATE: one query, end to end, before spending an hour on EDGAR
# ============================================================================
# Ingests the repository's own samples/ into a throwaway corpus and asks one question.
# If the wiring is broken, it breaks here in two minutes rather than after step 7.
from docureason.pipeline import DocuReasonPipeline
from docureason.tripath.serving.query_service import QueryService

SMOKE_CORPUS = WORK / "smoke_corpus"
smoke_report = DocuReasonPipeline(input_dir="samples", output_dir=SMOKE_CORPUS, verbose=False).run()
print(f"ingested {smoke_report.get('document_count')} document(s) / {smoke_report.get('chunk_count')} chunk(s)")
print(f"degradation: {smoke_report.get('degradation') or 'none'}")

smoke_service = QueryService(input_dir="samples", output_dir=SMOKE_CORPUS, verbose=False)
smoke_response = smoke_service.query("What revenue figure is reported in the table?")

REQUIRED_KEYS = ["results", "answer", "retrieved_evidence", "latency_breakdown", "degradation", "cached"]
missing = [key for key in REQUIRED_KEYS if key not in smoke_response]
assert not missing, f"response is missing {missing} - the query path is not wired as documented"

breakdown = smoke_response["latency_breakdown"]
print(f"\nresults          : {len(smoke_response['results'])}")
print(f"answer           : {str(smoke_response['answer'])[:160]}")
print(f"retrieval mode   : {[(m, s.get('mode')) for m, s in (smoke_response.get('retrieval_stats') or {}).items()]}")
print(f"total_ms         : {breakdown.get('total_ms')}   residual_ms: {breakdown.get('residual_ms')}")
print(f"stages           : {sorted((breakdown.get('stages_ms') or {}).keys())}")
print(f"degradation      : {smoke_response['degradation'] or 'none'}")

# The cache must return the same answer without re-running the pipeline.
again = smoke_service.query("What revenue figure is reported in the table?")
print(f"\nsecond call cached: {again.get('cached')}  (first-run timings kept: {'latency_breakdown_first_run' in again})")
assert again["answer"] == smoke_response["answer"], "cache returned a different answer"
print("\nSMOKE GATE PASSED")
"""
)


code(
    r"""
# ============================================================================
# CELL 7 - BUILD THE EVALUATION CORPUS: fetch, ingest, derive gold queries, validate
# ============================================================================
# Four idempotent steps. The expensive one is ingestion: layout parsing a 10-K takes
# minutes, and gold region ids only exist after it, so the corpus is the schedule.
EVAL_ROOT = REPO_DIR / "data" / "eval"
CORPUS_DIR = EVAL_ROOT / "corpus"
QUERIES = EVAL_ROOT / "queries.json"
NO_LAYOUT_DIR = EVAL_ROOT / "corpus_no_layout"

build_cmd = [
    sys.executable, "scripts/build_eval_corpus.py",
    "--out", str(EVAL_ROOT),
    "--filings", str(FILINGS),
    "--per-company", str(PER_COMPANY),
    "--per-stratum", str(PER_STRATUM),
    "--seed", str(SEED),
]
if RESTORED["filings"]:
    build_cmd.append("--skip-fetch")
if RESTORED["corpus"] and (CORPUS_DIR / "corpus.json").exists():
    build_cmd.append("--skip-ingest")

if RESTORED["queries"] and QUERIES.exists() and (CORPUS_DIR / "corpus.json").exists():
    print(f"Reusing the restored corpus and query set at {EVAL_ROOT}")
else:
    run(build_cmd, label="build_eval_corpus")

payload = json.loads(QUERIES.read_text(encoding="utf-8"))
report, validation = payload["generation_report"], payload["validation"]
print("\nCorpus")
print(f"  filings      : {payload['filing_count']}")
print(f"  documents    : {report['documents']}")
print(f"  regions      : {report['regions_scanned']}")
print(f"  gold queries : {report['query_count']}  {report['per_stratum']}")
print(f"  rejected     : {report['rejected']}")
print(f"  validation   : {'PASS' if validation['valid'] else 'FAIL'}")
if not validation["valid"]:
    raise SystemExit("gold ids do not resolve against the corpus - refusing to measure against broken labels")

audit = json.loads((CORPUS_DIR / "quality_audit.json").read_text(encoding="utf-8"))
capabilities = audit.get("capabilities_used", {})
print("\nWhat actually ran (observed, not claimed)")
print(f"  regions by layout source : {capabilities.get('regions_by_layout_source')}")
print(f"  tables serialized        : {capabilities.get('structured_tables_serialized')}")
print(f"  table structure enabled  : {capabilities.get('table_structure_enabled')}")
print(f"  duckdb table store       : {capabilities.get('table_store')}")
print(f"  embedding model          : {capabilities.get('dense_index_model')} @ {capabilities.get('dense_index_revision')}")
print(f"  index stats              : {capabilities.get('index_stats')}")
print(f"  degradation              : {audit.get('degradation') or 'none'}")
"""
)

code(
    r"""
# ============================================================================
# CELL 8 - CORPUS #2: the same filings, layout parser off
# ============================================================================
# The layout-parser ablation is ingestion-scoped: the parser runs once per corpus, so it
# cannot be switched off per query. Without this second corpus the seventh matrix row can
# only be reported as "not executed" - which the runner does, rather than printing a zero.
#
# Its gold ids are NOT comparable with the main corpus's: re-ingesting renames every
# region, so the runner reports the gold-id metrics for that row as not comparable and
# carries it on answer quality plus a region/table census instead.
if not RUN_NO_LAYOUT_CORPUS:
    print("Skipped by configuration; the no_layout_parser row will report as not executed.")
elif (NO_LAYOUT_DIR / "corpus.json").exists():
    print(f"Reusing {NO_LAYOUT_DIR}")
else:
    run(
        [
            sys.executable, "scripts/build_eval_corpus.py",
            "--out", str(EVAL_ROOT),
            "--corpus-dir", str(NO_LAYOUT_DIR),
            "--no-layout-parser",
            "--skip-derive",     # deriving here would relabel every gold id against this corpus
            "--skip-fetch",
        ],
        label="build_eval_corpus --no-layout-parser",
    )

if (NO_LAYOUT_DIR / "quality_audit.json").exists():
    ablated_audit = json.loads((NO_LAYOUT_DIR / "quality_audit.json").read_text(encoding="utf-8"))
    ablated_caps = ablated_audit.get("capabilities_used", {})
    print("\nlayout parser on  vs  off")
    print(f"  regions : {audit.get('total_regions')}  vs  {ablated_audit.get('total_regions')}")
    print(f"  chunks  : {audit.get('total_chunks')}  vs  {ablated_audit.get('total_chunks')}")
    print(
        f"  tables  : {capabilities.get('structured_tables_serialized')}"
        f"  vs  {ablated_caps.get('structured_tables_serialized')}"
    )
    print(f"  sources : {capabilities.get('regions_by_layout_source')}  vs  {ablated_caps.get('regions_by_layout_source')}")
"""
)

code(
    r"""
# ============================================================================
# CELL 9 - SCORE THE FULL SYSTEM OVER EVERY GOLD QUERY
# ============================================================================
# Bootstrap intervals, a per-stratum breakdown, the per-stage latency decomposition and a
# census of every degradation label the responses carried.
eval_cmd = [
    sys.executable, "scripts/run_eval_suite.py",
    "--queries", str(QUERIES),
    "--filings", str(EVAL_ROOT / "filings"),
    "--corpus", str(CORPUS_DIR),
    "--out", "artifacts/evaluation",
    "--iterations", str(BOOTSTRAP_ITERATIONS),
    "--run-name", "eval_suite",
]
if EVAL_LIMIT:
    eval_cmd += ["--limit", str(EVAL_LIMIT), "--stratified"]
run(eval_cmd, label="run_eval_suite")

suite = json.loads((REPO_DIR / "artifacts" / "evaluation" / "eval_suite.json").read_text(encoding="utf-8"))
print(f"\nscored {suite['queries_scored']} queries in {suite['wall_clock_sec']}s")
"""
)


code(
    r"""
# ============================================================================
# CELL 10 - THE SEVEN-ROW ABLATION MATRIX, AND THE NAIVE-RAG BASELINE
# ============================================================================
# Every row re-runs every query with one component disabled and reports a *paired*
# comparison: mean delta, bootstrap CI, sign-flip p-value. Each row prints its measured
# delta beside the delta Documentations/eval_method.md publishes, with a verdict, so a
# claim the measurement does not support is visible instead of inherited.
#
# This is the long pole: (1 + rows + baseline) x queries full pipeline executions.
ablation_cmd = [
    sys.executable, "scripts/run_ablations.py",
    "--queries", str(QUERIES),
    "--filings", str(EVAL_ROOT / "filings"),
    "--corpus", str(CORPUS_DIR),
    "--out", "artifacts/ablations",
    "--iterations", str(BOOTSTRAP_ITERATIONS),
    "--seed", str(SEED),
]
if ABLATION_LIMIT:
    ablation_cmd += ["--limit", str(ABLATION_LIMIT)]
    if ABLATION_STRATIFIED:
        ablation_cmd.append("--stratified")
if RUN_NO_LAYOUT_CORPUS and (NO_LAYOUT_DIR / "corpus.json").exists():
    ablation_cmd += ["--ingestion-corpus", f"no_layout_parser={NO_LAYOUT_DIR}"]

if RUN_ABLATIONS:
    run(ablation_cmd, label="run_ablations")
else:
    print("Skipped by configuration.")
"""
)

code(
    r"""
# ============================================================================
# CELL 11 - THE METRIC REGRESSION GATE
# ============================================================================
# Scores a frozen fixture - 20 cases across four strata, fixed retrieval output, no
# models, no network - against a committed baseline of 99 values. It runs in CI as a
# blocking step; running it here proves this environment reproduces those numbers before
# any of the numbers above are trusted.
if RUN_REGRESSION_GATE:
    run([sys.executable, "scripts/check_metric_regression.py", "--tolerance", "0.01"], label="metric regression gate")
else:
    print("Skipped by configuration.")
"""
)

code(
    r"""
# ============================================================================
# CELL 12 - RESULTS
# ============================================================================
from IPython.display import Image, Markdown, display

ABL_DIR = REPO_DIR / "artifacts" / "ablations"
lines = ["## Full system", ""]

summary = suite["summary_metrics"]
intervals = suite.get("confidence_intervals") or {}
alias = {"average_recall_at_5": "recall_at_5", "average_ndcg_at_5": "ndcg_at_5", "mean_latency_ms": "latency_ms"}
lines += ["| metric | value | 95% CI |", "| :--- | ---: | :--- |"]
for key in (
    "average_recall_at_5", "average_ndcg_at_5", "mrr", "map_at_10",
    "attribution_precision", "hallucination_rate", "token_f1", "exact_match",
    "sql_execution_success_rate", "latency_p50_ms", "latency_p90_ms",
):
    if key not in summary:
        continue
    interval = intervals.get(alias.get(key, key)) or {}
    low, high = interval.get("ci_low"), interval.get("ci_high")
    span = "—" if low is None else f"[{low:.4f}, {high:.4f}]"
    lines.append(f"| `{key}` | {summary[key]:.4f} | {span} |")

lines += ["", "### Per stratum", "", "| stratum | n | recall@5 | ndcg@5 | token_f1 |", "| :--- | ---: | ---: | ---: | ---: |"]
def cell(metrics, key):
    value = (metrics or {}).get(key)
    # "not measured" and "measured zero" are different findings; an em dash says which.
    return "—" if not isinstance(value, (int, float)) else f"{value:.4f}"


for stratum, block in sorted((suite.get("per_stratum") or {}).items()):
    metrics = block.get("summary_metrics") or {}
    lines.append(
        f"| {stratum} | {block.get('n')} | {cell(metrics, 'average_recall_at_5')} "
        f"| {cell(metrics, 'average_ndcg_at_5')} | {cell(metrics, 'token_f1')} |"
    )

stages = suite.get("stage_latency_ms") or {}
stage_rows = [(name, block) for name, block in stages.items() if isinstance(block, dict)]
if stage_rows:
    lines += ["", "### Per-stage latency (ms)", "", "| stage | mean | p95 |", "| :--- | ---: | ---: |"]
    for name, block in sorted(stage_rows, key=lambda row: -row[1].get("mean_ms", 0)):
        lines.append(f"| `{name}` | {block.get('mean_ms')} | {block.get('p95_ms')} |")

lines += ["", "### Degradation", ""]
lines.append(
    "\n".join(f"- `{label}` on {count} query(ies)" for label, count in (suite.get("degradation_counts") or {}).items())
    or "- none reported on any query"
)

matrix_path = ABL_DIR / "ablation_matrix.md"
if matrix_path.exists():
    lines += ["", "## Ablation matrix", "", matrix_path.read_text(encoding="utf-8").split("\n", 1)[1].strip()]

ablation_report = {}
if (ABL_DIR / "ablation_report.json").exists():
    ablation_report = json.loads((ABL_DIR / "ablation_report.json").read_text(encoding="utf-8"))
    baseline = ablation_report.get("baseline") or {}
    if baseline:
        lines += [
            "", f"## Full system vs {baseline.get('system')}", "",
            "Sign convention: positive means the full system scores higher.", "",
            "| metric | Δ | 95% CI | p | significant |", "| :--- | ---: | :--- | ---: | :-- |",
        ]
        for metric, comparison in baseline.get("full_vs_baseline", {}).items():
            delta = comparison.get("delta")
            if delta is None:
                continue
            low, high = comparison.get("delta_ci_low"), comparison.get("delta_ci_high")
            p_value = comparison.get("p_value")
            lines.append(
                f"| `{metric}` | {delta:+.4f} | [{low:+.4f}, {high:+.4f}] "
                f"| {'—' if p_value is None else f'{p_value:.4f}'} | {'yes' if comparison.get('significant') else 'no'} |"
            )
    subset = ablation_report.get("query_subset") or {}
    lines += [
        "", "### Scope of this run", "",
        f"- queries used: **{subset.get('queries_used')}** (full matrix: **{subset.get('is_full_matrix')}**)",
        f"- bootstrap iterations: {ablation_report.get('bootstrap_iterations')}, seed {ablation_report.get('seed')}",
        f"- rows executed: {ablation_report.get('executed')} of {len(ablation_report.get('ablations') or [])}",
        f"- not executed: {ablation_report.get('not_executed') or 'none'}",
    ]

RESULTS_MD = "\n".join(lines)
display(Markdown(RESULTS_MD))
"""
)


code(
    r"""
# ============================================================================
# CELL 13 - SCIENTIFIC FIGURES AND THE INSIGHT DIGEST
# ============================================================================
# Seven figures plus INSIGHTS.md, every number in them read out of the two JSON reports:
# point estimates with their bootstrap intervals, quality per stratum, the latency
# decomposition and its full distribution, the degradation census, each ablation's measured
# effect beside the claim it is meant to support, and the paired comparison against naive
# RAG. PDF alongside PNG, so the same figure goes into a paper without restyling.
run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", str(constraints), "matplotlib"],
    label="pip install matplotlib",
)

FIG_DIR = REPO_DIR / "artifacts" / "figures"
run(
    [
        sys.executable, "scripts/plot_results.py",
        "--eval-report", "artifacts/evaluation/eval_suite.json",
        "--ablation-report", "artifacts/ablations/ablation_report.json",
        "--out", str(FIG_DIR),
        "--formats", "png,pdf",
    ],
    label="plot_results",
)

INSIGHTS_MD = (FIG_DIR / "INSIGHTS.md").read_text(encoding="utf-8")
display(Markdown(INSIGHTS_MD))
for png in sorted(FIG_DIR.glob("fig*.png")):
    display(Image(filename=str(png)))
"""
)


code(
    r"""
# ============================================================================
# CELL 14 - EXPORT EVERYTHING AS A REUSABLE DATASET
# ============================================================================
# Anything expensive to recompute goes here: the fetched filings and their provenance,
# the ingested corpus with its FAISS/BM25 indexes and DuckDB table store (these *are* the
# cached embeddings), the derived gold queries, the second corpus, every report, and the
# model-weight cache. Attach this output under /kaggle/input/docureason-artifacts next
# session and Cell 5 skips straight to measurement.
EXPORT_MODEL_CACHE = True

if EXPORT_DIR.exists():
    shutil.rmtree(EXPORT_DIR)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _copy(source: Path, relative: str) -> int:
    if not source.exists():
        return 0
    destination = EXPORT_DIR / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    return sum(path.stat().st_size for path in destination.rglob("*") if path.is_file()) if destination.is_dir() else destination.stat().st_size


sizes = {
    "data/eval/filings": _copy(EVAL_ROOT / "filings", "data/eval/filings"),
    "data/eval/queries.json": _copy(QUERIES, "data/eval/queries.json"),
    "data/eval/corpus": _copy(CORPUS_DIR, "data/eval/corpus"),
    "data/eval/corpus_no_layout": _copy(NO_LAYOUT_DIR, "data/eval/corpus_no_layout"),
    "artifacts/ablations": _copy(REPO_DIR / "artifacts" / "ablations", "artifacts/ablations"),
    "artifacts/evaluation": _copy(REPO_DIR / "artifacts" / "evaluation", "artifacts/evaluation"),
    "artifacts/figures": _copy(FIG_DIR, "artifacts/figures"),
    "artifacts/logs": _copy(REPO_DIR / "artifacts" / "logs", "artifacts/logs"),
}
if EXPORT_MODEL_CACHE:
    sizes["hf_cache"] = _copy(HF_CACHE, "hf_cache")

(EXPORT_DIR / "RESULTS.md").write_text(RESULTS_MD, encoding="utf-8")

manifest = {
    "produced_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "repository": {"url": REPO_URL, "ref": REPO_REF, "sha": commit},
    "framework_version": docureason.__version__,
    "environment": {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "modules_available": available,
    },
    "configuration": {
        "filings": FILINGS,
        "per_company": PER_COMPANY,
        "per_stratum": PER_STRATUM,
        "seed": SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "eval_limit": EVAL_LIMIT,
        "ablation_limit": ABLATION_LIMIT,
        "no_layout_corpus": RUN_NO_LAYOUT_CORPUS,
    },
    "corpus": {
        "filings": payload["filing_count"],
        "documents": report["documents"],
        "regions": report["regions_scanned"],
        "gold_queries": report["query_count"],
        "per_stratum": report["per_stratum"],
        "rejected": report["rejected"],
        "validation_passed": validation["valid"],
        "capabilities_used": capabilities,
    },
    "results": {
        "summary_metrics": suite["summary_metrics"],
        "queries_scored": suite["queries_scored"],
        "degradation_counts": suite.get("degradation_counts"),
        "ablation_rows_executed": ablation_report.get("executed"),
        "ablation_rows_not_executed": ablation_report.get("not_executed"),
        "is_full_matrix": (ablation_report.get("query_subset") or {}).get("is_full_matrix"),
    },
    "restored_from_cache": RESTORED,
    "bytes": sizes,
}
(EXPORT_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

total = sum(sizes.values())
print(f"Exported to {EXPORT_DIR}\n")
for name, size in sorted(sizes.items(), key=lambda item: -item[1]):
    print(f"  {name:34s} {size / 1024 ** 2:9.1f} MB")
print(f"  {'TOTAL':34s} {total / 1024 ** 2:9.1f} MB")
print("\nTo reuse: Save Version -> Save & Run All, then in the next session")
print("Add Data -> Your Work -> this notebook's output, mounted at")
print("/kaggle/input/docureason-artifacts. Cell 5 will restore the corpus, the indexes,")
print("the DuckDB store and the model weights, and cells 7-8 become no-ops.")
"""
)

md(
    r"""
## What this run does and does not establish

**Established, in this session, from these filings:** every metric above, with the
interval around it and the stratum it came from; the per-stage latency decomposition
including its signed residual; each ablation's paired delta against the full system, with
the published claim printed beside it; the naive-RAG comparison, scored through the same
harness with the same generator and the same embedding model.

**Not established:**

- **Gold labels are derived, not annotated.** Table cells know their region by
  construction and prose anchors are phrases verified unique across the corpus, but
  derivable queries over-represent lookup and cannot express multi-hop reasoning.
- **The `figure` stratum is usually empty.** Docling's HTML pipeline emits no figure
  regions for these filings, so the rasterization work is exercised by tests, not by this
  measurement. A PDF corpus would change that.
- **The `no_layout_parser` row cannot test its own recall claim.** Its corpus renames
  every gold region, so the runner reports the gold-id metrics as not comparable and
  carries the row on answer quality plus a region/table census.
- **Attribution is lexical plus numeric consistency, not entailment.** `entailment_threshold`
  is inert until a real NLI model is loaded, and `attribution_precision` should be read as
  claim-support coverage rather than faithfulness.

`Documentations/audit.md` is the plan of record; §13 lists what remains.
"""
)


def build_notebook() -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    for kind, source in CELLS:
        cell: Dict[str, Any] = {
            "cell_type": kind,
            "metadata": {},
            # nbformat wants a list of lines, each keeping its newline except the last.
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "language": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help=f"output path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    notebook = build_notebook()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    code_cells = sum(1 for kind, _ in CELLS if kind == "code")
    print(f"wrote {args.out}")
    print(f"  {len(CELLS)} cells ({code_cells} code, {len(CELLS) - code_cells} markdown)")
    print(f"  repository: {REPO_URL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
