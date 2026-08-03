from __future__ import annotations

import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "artifacts" / "test_run" / "pipeline_report.json"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = self._build_html()
            self.wfile.write(html.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _build_html(self) -> str:
        if REPORT_PATH.exists():
            report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        else:
            report = {"status": "Run the pipeline first"}

        sections = []
        for key, value in report.items():
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, indent=2)
            else:
                rendered = str(value)
            sections.append(f"<h2>{key}</h2><pre>{rendered}</pre>")
        return f"""<!doctype html>
<html>
  <head><meta charset='utf-8'><title>DocuReason Dashboard</title></head>
  <body>
    <h1>DocuReason Phase 1 + Phase 2 Dashboard</h1>
    {''.join(sections)}
  </body>
</html>"""


def main() -> None:
    server = HTTPServer(("127.0.0.1", 8001), Handler)
    print("Dashboard running at http://127.0.0.1:8001")
    server.serve_forever()


if __name__ == "__main__":
    main()
