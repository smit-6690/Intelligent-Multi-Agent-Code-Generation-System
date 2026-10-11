from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import Settings
from .pipeline import MultiAgentPipeline


class Handler(BaseHTTPRequestHandler):
    pipeline: MultiAgentPipeline | None = None

    def _write(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write(200, {"status": "ok", "service": "magent-codegen"})
            return
        self._write(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/generate":
            self._write(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            specification = payload["specification"]
            if not isinstance(specification, str) or not specification.strip():
                raise ValueError("specification must be a non-empty string")
            if self.pipeline is None:
                self.pipeline = MultiAgentPipeline(Settings.from_env())
            result = self.pipeline.run(specification)
            self._write(
                200,
                {
                    "code": result.code,
                    "passed": result.passed,
                    "repair_attempts": result.repair_attempts,
                    "review": result.review,
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._write(400, {"error": str(exc)})
        except Exception as exc:  # Keep the HTTP contract stable for backend failures.
            self._write(500, {"error": f"generation failed: {exc}"})

    def log_message(self, format: str, *args) -> None:
        del format, args


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"magent-codegen listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
