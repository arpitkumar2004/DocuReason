from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class ArtifactWriter:
    """Write structured pipeline artifacts for downstream reuse and reproducibility."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.output_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_manifest(self, documents: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Path:
        manifest = {
            "documents": documents,
            "chunks": chunks,
            "generated_at": Path().as_posix(),
        }
        return self.write_json("manifest", manifest)
