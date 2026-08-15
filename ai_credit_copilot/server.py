"""Local HTTP server for the AI credit-review workbench."""

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .contracts import InputValidationError
from .qwen_client import QwenConfigurationError, QwenRequestError
from .service import CreditCopilotService


PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PACKAGE_ROOT / "static"
SAMPLE_CASE_PATH = PACKAGE_ROOT / "sample_case.json"
MAX_REQUEST_BYTES = 1024 * 1024


class CopilotRequestHandler(BaseHTTPRequestHandler):
    service = CreditCopilotService()
    server_version = "CrediFusion-AICopilot/0.1"

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, self.service.health())
            return
        if path == "/api/demo-case":
            self._send_sample_case()
            return
        self._serve_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/analyze", "/api/chat"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            payload = self._read_json()
            result = self.service.analyze(payload) if path == "/api/analyze" else self.service.chat(payload)
            self._send_json(HTTPStatus.OK, result)
        except InputValidationError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "message": str(exc)})
        except QwenConfigurationError as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "qwen_not_configured", "message": str(exc)})
        except QwenRequestError as exc:
            self._send_json(HTTPStatus.BAD_GATEWAY, {"error": "qwen_request_failed", "message": str(exc)})
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_request", "message": str(exc)})
        except Exception:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "internal_error", "message": "Unexpected server error"})

    def _read_json(self):
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise InputValidationError("Content-Length is required")
        try:
            length = int(content_length)
        except ValueError:
            raise InputValidationError("Content-Length must be an integer")
        if length <= 0 or length > MAX_REQUEST_BYTES:
            raise InputValidationError("request body must be between 1 byte and 1 MiB")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise InputValidationError("request body must be UTF-8 JSON")

    def _send_sample_case(self):
        try:
            with SAMPLE_CASE_PATH.open("r", encoding="utf-8") as handle:
                self._send_json(HTTPStatus.OK, json.load(handle))
        except (OSError, ValueError):
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "sample_case_unavailable"})

    def _serve_static(self, request_path):
        relative_path = "index.html" if request_path in ("", "/") else unquote(request_path).lstrip("/")
        candidate = (STATIC_ROOT / relative_path).resolve()
        try:
            candidate.relative_to(STATIC_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(str(candidate))
        content_type = content_type or "application/octet-stream"
        data = candidate.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type + ("; charset=utf-8" if content_type.startswith("text/") else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def run_server(host="127.0.0.1", port=8787):
    httpd = ThreadingHTTPServer((host, port), CopilotRequestHandler)
    print("AI credit copilot available at http://%s:%s" % (host, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run the CrediFusion AI credit copilot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    run_server(args.host, args.port)
