from __future__ import annotations

from pathlib import Path
from typing import List


class QdrantWriter:
    """Writes lightweight index payloads to disk as a stand-in for Qdrant."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, payload: List[dict]) -> Path:
        output_path = self.output_dir / f"{name}.json"
        output_path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
        return output_path
