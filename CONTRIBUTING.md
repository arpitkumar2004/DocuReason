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

3. **Install Package in Editable Mode**:
   ```bash
   pip install -e ".[dev]"
   pip install build twine pytest
   ```

---

## Running Tests

Before submitting a Pull Request, verify that all pytest unit and integration tests pass:

```bash
python -m pytest -q
```

---

## Contribution Guidelines

1. **Modular Architecture**: Ensure new retrievers, routers, or encoders adhere to the `src/tripath/` modular folder structure.
2. **Type Annotations & Docstrings**: Include type hints (`typing`) and clear docstrings for all new classes and methods.
3. **No Breaking Changes**: Maintain backward compatibility with the high-level `docureason` package interface (`DocuReasonPipeline`, `QueryService`).
4. **Clean Code Style**: Format code using `black` and adhere to PEP 8 standards.
