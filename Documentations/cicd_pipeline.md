# DocuReason v1.1.1 — CI/CD Pipeline & PyPI Release Engineering Specification

This document provides a comprehensive technical reference for the Continuous Integration, Continuous Delivery (CI/CD), and PyPI deployment infrastructure powering `docureason-framework`.

---

## Table of Contents

1. [Architectural Overview](#architectural-overview)
2. [GitHub Actions Workflow Architecture](#github-actions-workflow-architecture)
   - [1. Continuous Integration (`ci.yml`)](#1-continuous-integration-ciyml)
   - [2. Production PyPI Publishing (`release-pypi.yml`)](#2-production-pypi-publishing-release-pypiyml)
   - [3. TestPyPI Staging Deployment (`testpypi-publish.yml`)](#3-testpypi-staging-deployment-testpypi-publishyml)
   - [4. Automated Dependency Management (`dependabot.yml`)](#4-automated-dependency-management-dependabotyml)
3. [PyPI Integration & Security Model](#pypi-integration--security-model)
   - [OIDC Trusted Publishing](#oidc-trusted-publishing)
   - [Fallback Authentication & GitHub Secrets](#fallback-authentication--github-secrets)
4. [Packaging Extras & Standardized Tooling](#packaging-extras--standardized-tooling)
5. [Local Build Verification & Quality Assurance](#local-build-verification--quality-assurance)
6. [Release Procedure & Checklist](#release-procedure--checklist)

---

## Architectural Overview

DocuReason employs an enterprise-grade, multi-stage CI/CD pipeline designed according to PyPA (Python Packaging Authority) best practices and GitHub Security Hardening standards.

```
                   +---------------------------------------+
                   |            Git Push / PR              |
                   +---------------------------------------+
                                       |
                                       v
          +---------------------------------------------------------+
          |                 CI Pipeline (ci.yml)                    |
          |  +--------------------+  +---------------------------+  |
          |  |  Ruff / Mypy /     |  | Pytest Matrix             |  |
          |  |  pip-audit         |  | (Py 3.10, 3.11, 3.12)     |  |
          |  +--------------------+  +---------------------------+  |
          |  +--------------------------------------------------+  |
          |  | Build & Twine Metadata Validation                |  |
          |  +--------------------------------------------------+  |
          +---------------------------------------------------------+
                                       |
                                       v
                   +---------------------------------------+
                   |      GitHub Release Published /       |
                   |      Manual Workflow Dispatch         |
                   +---------------------------------------+
                                       |
                                       v
          +---------------------------------------------------------+
          |             Release Pipeline (release-pypi.yml)         |
          |  +--------------------------------------------------+  |
          |  | Build Source Dist (.tar.gz) & Wheel (.whl)       |  |
          |  +--------------------------------------------------+  |
          |  | OIDC Trusted Publishing to PyPI                  |  |
          |  +--------------------------------------------------+  |
          |  | Upload Compiled Dist Artifacts to GitHub Release |  |
          |  +--------------------------------------------------+  |
          +---------------------------------------------------------+
```

---

## GitHub Actions Workflow Architecture

### 1. Continuous Integration (`.github/workflows/ci.yml`)

The primary CI pipeline triggers on `push` to `main`, `master`, and `release/*` branches, as well as on all incoming `pull_request` events.

- **Concurrency Control**: Automatically cancels stale in-flight runs when new commits are pushed to the same branch or PR (`cancel-in-progress: true`).
- **Job Breakdown**:
  1. **Lint & Security Audit (`lint-and-audit`)**:
     - **Ruff**: Enforces PEP 8 style guidelines, import sorting (`isort`), and static code safety rules.
     - **Mypy**: Conducts static type check verification against `src/` and `docureason/`.
     - **pip-audit**: Audits installed dependency tree against known CVE vulnerability databases.
  2. **Multi-Version Test Matrix (`test`)**:
     - Runs `pytest` with code coverage tracking (`pytest-cov`) across **Python 3.10, 3.11, and 3.12**.
     - Generates XML coverage artifacts (`coverage.xml`).
  3. **Package Build & Metadata Check (`build-and-verify-package`)**:
     - Builds source distribution (`sdist`) and wheel (`wheel`) via PyPA `build`.
     - Validates PyPI long description markdown rendering and metadata compliance with `twine check --strict`.
     - Inspects wheel structural integrity using `check-wheel-contents`.

### 2. Production PyPI Publishing (`.github/workflows/release-pypi.yml`)

The production release workflow deploys compiled packages to [PyPI (`docureason-framework`)](https.pypi.org/project/docureason-framework/).

- **Triggers**: Triggered when a new GitHub Release is `published` or via manual `workflow_dispatch` (with optional `dry_run` mode).
- **Environment**: Scoped to the `pypi` deployment environment.
- **Artifact Upload**: Publishes sdist and wheel distributions to PyPI and attaches `.whl` / `.tar.gz` packages directly to the corresponding GitHub Release.

### 3. TestPyPI Staging Deployment (`.github/workflows/testpypi-publish.yml`)

Provides a staging ground on [TestPyPI](https://test.pypi.org/) to test releases prior to production deployment.

- **Trigger**: Manual `workflow_dispatch`.
- **Environment**: Scoped to `testpypi`.

### 4. Automated Dependency Management (`.github/dependabot.yml`)

Configures GitHub Dependabot to check weekly for:
- GitHub Actions action updates (e.g., `actions/checkout`, `actions/setup-python`).
- Python package dependency updates declared in `pyproject.toml`.

---

## PyPI Integration & Security Model

### OIDC Trusted Publishing

DocuReason utilizes **OpenID Connect (OIDC) Trusted Publishing**, eliminating long-lived API tokens stored in repository secrets.

#### PyPI Configuration Requirements:
- **PyPI Owner**: `arpitkumar2004`
- **PyPI Project Name**: `docureason-framework`
- **Publisher**: GitHub
- **Owner / Organization**: `arpitkumar2004`
- **Repository Name**: `DocuReason`
- **Workflow Name**: `release-pypi.yml`
- **Environment Name**: `pypi`

During workflow execution, GitHub issues a short-lived OIDC JWT token containing repository context claims, which PyPI authenticates to authorize package publishing.

### Fallback Authentication & GitHub Secrets

For legacy setups or custom mirrors, `pypa/gh-action-pypi-publish` gracefully accepts `PYPI_API_TOKEN` stored in GitHub Repository Secrets.

---

## Packaging Extras & Standardized Tooling

`pyproject.toml` defines modular optional dependency groups to streamline installation for developers and CI runners:

```toml
[project.optional-dependencies]
test = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.24.0"
]
lint = [
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "pip-audit>=2.6.0"
]
build = [
    "build>=1.0.0",
    "twine>=4.0.2",
    "wheel>=0.41.0"
]
dev = [
    "docureason-framework[test,lint,build]"
]
```

### Tool Settings in `pyproject.toml`:
- **Ruff**: `line-length = 120`, target `py310`, checks `E, F, W, I, B`.
- **Pytest**: Automated test discovery under `tests/` with verbose output.
- **Coverage**: Traces `src/` and `docureason/` modules.
- **Mypy**: Configured with `ignore_missing_imports = true` for ML third-party libraries.

---

## Local Build Verification & Quality Assurance

DocuReason provides a dedicated local verification utility script: `scripts/verify_pypi_package.py`.

### Execution:
```bash
python scripts/verify_pypi_package.py
```

### Verification Steps:
1. Cleans leftover `dist/`, `build/`, and `*.egg-info` directories.
2. Builds clean `sdist` (`.tar.gz`) and `wheel` (`.whl`) via `python -m build`.
3. Runs `twine check --strict` to verify PyPI description syntax, license tags, and author metadata.
4. Checks wheel interior file distribution.
5. Installs the built wheel in an isolated temporary virtual environment (`tempfile`) and verifies that `import docureason` succeeds and reports correct `__version__`.

---

## Release Procedure & Checklist

To perform a new release of `docureason-framework`:

1. **Update Version**:
   - Update version string in `pyproject.toml` (e.g., `version = "1.2.0"`).
   - Update `__version__` in `docureason/__init__.py`.
   - Document changes in `CHANGELOG.md`.

2. **Run Local Verification**:
   ```bash
   python scripts/verify_pypi_package.py
   python -m pytest
   ```

3. **Deploy to TestPyPI (Optional Staging)**:
   - Go to GitHub Repository -> **Actions** -> **TestPyPI Publishing Pipeline** -> **Run workflow**.

4. **Publish Production Release**:
   - Create a tag and publish a release via GitHub Releases UI (e.g., Tag `v1.2.0`).
   - The `release-pypi.yml` GitHub Action automatically triggers, validates, publishes to PyPI, and attaches artifacts to the GitHub release.
