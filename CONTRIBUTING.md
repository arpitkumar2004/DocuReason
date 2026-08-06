# Contributing to DocuReason

Thank you for your interest in contributing to **DocuReason** (`docureason-framework`)!

DocuReason is an open-source, enterprise-grade Tri-Path Multimodal RAG framework. We welcome contributions from researchers, machine learning engineers, and open-source developers.

---

## Development Setup

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/DocuReason.git
   cd DocuReason
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Package with Developer Extras**:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

---

## Code Quality & Verification Suite

Before submitting a Pull Request, run the local verification suite:

1. **Unit & Integration Tests**:
   ```bash
   python -m pytest -v
   ```

2. **Linting & Code Formatting**:
   ```bash
   ruff check .
   ruff format --check .
   ```

3. **Type Check**:
   ```bash
   mypy src docureason
   ```

4. **Security Vulnerability Audit**:
   ```bash
   pip-audit
   ```

5. **PyPI Package Verification**:
   ```bash
   python scripts/verify_pypi_package.py
   ```

---

## Contribution Guidelines

1. **Modular Architecture**: Ensure new retrievers, routers, or encoders adhere to the `src/tripath/` modular folder structure.
2. **Type Annotations & Docstrings**: Include type hints (`typing`) and clear docstrings for all new classes and methods.
3. **No Breaking Changes**: Maintain backward compatibility with the high-level `docureason` package interface (`DocuReasonPipeline`, `QueryService`).
4. **Clean Code Style**: Format code using `ruff` and adhere to PEP 8 standards.
5. **CI Pipeline Pass**: Ensure all GitHub Actions CI checks (`.github/workflows/ci.yml`) pass on your pull request.
