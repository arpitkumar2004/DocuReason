# syntax=docker/dockerfile:1
#
# DocuReason — CPU-only container image for the FastAPI serving surface.
#
# Two stages, so the runtime image carries no compiler toolchain and no pip cache.
# Dependencies come from requirements.lock rather than the ranges in pyproject.toml,
# so an image built today and one built next month contain the same code. Model
# *weights* are pinned separately, by commit, in docureason/tripath/model_pins.py.
#
#   docker build -t docureason:1.1.4 .
#   docker run --rm -p 127.0.0.1:8000:8000 \
#       -v "$PWD/samples:/app/samples:ro" \
#       -v docureason-models:/models \
#       -v docureason-artifacts:/app/artifacts \
#       docureason:1.1.4
#
# SECURITY: no route is authenticated, and /api/query and /api/ingest take a corpus
# path from the request body (contained to /app by serving/main.py, but still
# attacker-chosen within it). Publishing this port puts an unauthenticated
# document-processing service on the network. Keep the published port on loopback as
# shown above, or terminate it behind a gateway that authenticates.

# ---------------------------------------------------------------------------
# Stage 1 — resolve dependencies into a virtualenv
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

# build-essential: a few wheels in this tree still compile from source.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# The CPU build of torch, from PyTorch's own index, before anything else. Installing
# torch from PyPI on linux drags in the CUDA runtime wheels — several GB this image
# never executes. The version tracks requirements.lock; move both together.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0

COPY requirements.lock ./
RUN pip install -r requirements.lock

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# libgl1 / libglib2.0-0: OpenCV, pulled in by easyocr and docling, links against them.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# The source tree is run from /app rather than pip-installed, because
# serving/main.py resolves ui/, configs/ and samples/ relative to the package's
# own location. Installed into site-packages it would look for them there.
ENV PYTHONPATH=/app

# transformers probes TensorFlow at import unless told not to. On this torch-only
# image that probe can take sentence-transformers down with it, which surfaces as a
# silently degraded dense index rather than an error.
ENV USE_TF=0
ENV USE_TORCH=1
ENV TOKENIZERS_PARALLELISM=false

# Weights download on first use rather than being baked in. Mount /models as a
# volume or every container start re-downloads them.
ENV HF_HOME=/models

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY docureason ./docureason
COPY configs ./configs
COPY samples ./samples
COPY ui ./ui

RUN useradd --create-home --uid 10001 docureason \
    && mkdir -p /app/artifacts /models \
    && chown -R docureason:docureason /app /models
USER docureason

EXPOSE 8000

# The long start period is not padding: uvicorn's lifespan handler warms the default
# corpus before the server accepts connections, and on a cold /models volume that
# includes downloading the embedding and cross-encoder weights.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# Binds 0.0.0.0 because the container's network namespace, not the process, is the
# boundary here. Control exposure with the -p mapping (see the security note above).
CMD ["uvicorn", "docureason.tripath.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
