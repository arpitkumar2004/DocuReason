from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
# Only the repository root goes on sys.path.  Adding ROOT/"src" as well used to make
# the same files importable as both ``tripath.X`` and ``src.tripath.X``, creating two
# distinct module objects per file — so ``isinstance(config, DocuReasonConfig)`` failed
# across the boundary and model caches were duplicated.  There is now one import root.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docureason.tripath.serving.main import app
from docureason.tripath.utils import setup_logger

logger = setup_logger("scripts.serve_dashboard")


def _is_port_in_use(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    port = 8001
    while port <= 8010 and _is_port_in_use(port):
        port += 1
    logger.info("Starting DocuReason Tri-Path Multimodal RAG Server at http://127.0.0.1:%d", port)
    print("\n=======================================================")
    print(" Tri-Path Multimodal RAG Server Live at:")
    print(f"   http://127.0.0.1:{port}")
    print("=======================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
