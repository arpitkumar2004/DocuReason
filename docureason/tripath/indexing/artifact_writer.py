from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from docureason.tripath.version import __version__ as _RUNTIME_VERSION


def _package_version(name: str) -> str:
    """Return an installed package version, or a marker when unavailable."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "not-installed"


def build_provenance(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Capture the environment that produced an artifact.

    Recorded so that a result can be traced back to the exact code and model
    stack that generated it.  Without this, no artifact in the repository is
    reproducible on another machine.
    """
    provenance: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # Version of the code that ran.  Distinct from the installed distribution
        # metadata below, which can lag in an editable install.
        "framework_version": _RUNTIME_VERSION,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "installed_packages": {
            pkg: _package_version(pkg)
            for pkg in (
                "docureason-framework",
                "torch",
                "transformers",
                "sentence-transformers",
                "faiss-cpu",
                "bm25s",
                "docling",
                "duckdb",
            )
        },
    }
    if extra:
        provenance.update(extra)
    return provenance


class ArtifactWriter:
    """Write structured pipeline artifacts for downstream reuse and reproducibility."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.output_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_manifest(
        self,
        documents: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Path:
        manifest = {
            "documents": documents,
            "chunks": chunks,
            "provenance": provenance or build_provenance(),
        }
        return self.write_json("manifest", manifest)
