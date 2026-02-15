#!/usr/bin/env python3
import argparse
import hmac
import json
import os
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class DashboardHandler(SimpleHTTPRequestHandler):
    root = Path(".").resolve()
    poll_interval = 0.5
    chat_model = "gpt-4o-mini"
    allowed_tailscale_login = ""
    auth_token = ""
    cors_origin = ""
    protect_results = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.root), **kwargs)

    def end_headers(self):
        if self.cors_origin:
            self.send_header("Access-Control-Allow-Origin", self.cors_origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Auth-Token")
            self.send_header("Vary", "Origin")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if self.protect_results and parsed.path.startswith("/results/"):
            if not self.is_authorized(parsed=parsed):
                self.send_unauthorized()
                return
        if parsed.path == "/events":
            if not self.is_authorized(parsed=parsed):
                self.send_unauthorized()
                return
            self.handle_events(parsed)
            return
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/chat":
            self.handle_chat(parsed)
            return
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":"not found"}')

    def get_tailscale_login(self) -> str:
        for key in (
            "Tailscale-User-Login",
            "X-Tailscale-User-Login",
            "Tailscale-User-Name",
            "X-Tailscale-User-Name",
        ):
            v = self.headers.get(key)
            if v:
                return v.strip().lower()
        return ""

    def get_token(self, parsed=None, payload=None) -> str:
        auth = (self.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        header_token = (self.headers.get("X-Auth-Token") or "").strip()
        if header_token:
            return header_token
        if parsed:
            qs_token = parse_qs(parsed.query).get("token", [""])[0].strip()
            if qs_token:
                return qs_token
        if isinstance(payload, dict):
            body_token = payload.get("token")
            if isinstance(body_token, str):
                return body_token.strip()
        return ""

    def is_authorized(self, parsed=None, payload=None) -> bool:
        expected_login = (self.allowed_tailscale_login or "").strip().lower()
        expected_token = (self.auth_token or "").strip()
        if not expected_login and not expected_token:
            return True

        if expected_login:
            login = self.get_tailscale_login()
            if login and login == expected_login:
                return True

        if expected_token:
            token = self.get_token(parsed=parsed, payload=payload)
            if token and hmac.compare_digest(token, expected_token):
                return True

        return False

    def send_unauthorized(self):
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("WWW-Authenticate", 'Bearer realm="su2-dashboard"')
        self.end_headers()
        self.wfile.write(
            b'{"error":"unauthorized","hint":"use Tailscale login allowlist or valid auth token"}'
        )

    def resolve_local_path(self, url_path: str) -> Path:
        p = (url_path or "").strip()
        if p.startswith("http://") or p.startswith("https://"):
            raise ValueError("absolute URLs are not supported for streaming endpoint")
        rel = p[1:] if p.startswith("/") else p
        full = (self.root / rel).resolve()
        if not (full == self.root or self.root in full.parents):
            raise ValueError("path escapes workspace root")
        return full

    @staticmethod
    def read_json(path: Path):
        if not path.exists():
            return None, f"{path} not found"
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f), None
        except Exception as e:
            return None, str(e)

    def handle_events(self, parsed):
        qs = parse_qs(parsed.query)
        progress_arg = qs.get("progress", ["/results/su2_signal_scan/progress_petrus-su2-signal.json"])[0]
        live_arg = qs.get("live", ["/results/su2_signal_scan/live_petrus-su2-signal.json"])[0]
        try:
            progress_path = self.resolve_local_path(progress_arg)
            live_path = self.resolve_local_path(live_arg)
        except ValueError as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        last_sig = None
        last_ping = 0.0
        try:
            while True:
                pm = progress_path.stat().st_mtime if progress_path.exists() else -1
                lm = live_path.stat().st_mtime if live_path.exists() else -1
                sig = (pm, lm)

                if sig != last_sig:
                    progress, progress_err = self.read_json(progress_path)
                    live, live_err = self.read_json(live_path)
                    payload = {
                        "ts": time.time(),
                        "progress": progress,
                        "live": live,
                        "errors": {},
                    }
                    if progress_err:
                        payload["errors"]["progress"] = progress_err
                    if live_err:
                        payload["errors"]["live"] = live_err
                    msg = f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.flush()
                    last_sig = sig
                    last_ping = time.time()
                elif time.time() - last_ping >= 15:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    last_ping = time.time()

                time.sleep(self.poll_interval)
        except (BrokenPipeError, ConnectionResetError):
            return

    def handle_chat(self, parsed):
        clen = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(clen) if clen > 0 else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"invalid json"}')
            return

        if not self.is_authorized(parsed=parsed, payload=payload):
            self.send_unauthorized()
            return

        messages = payload.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"messages must be a non-empty list"}')
            return

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "error": "OPENAI_API_KEY is not set for dashboard server; chat endpoint unavailable."
                    }
                ).encode("utf-8")
            )
            return

        system = {
            "role": "system",
            "content": (
                "You are an admin helper for SU(2) lattice runs. "
                "Answer concisely and prioritize run status, interpretation, and next actions."
            ),
        }
        chat_messages = [system]
        for m in messages[-20:]:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                chat_messages.append({"role": role, "content": content})

        req_payload = {
            "model": self.chat_model,
            "messages": chat_messages,
            "temperature": 0.2,
        }
        req = Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(req_payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=60) as r:
                out = json.loads(r.read().decode("utf-8"))
            reply = (
                out.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"reply": reply}).encode("utf-8"))
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"error": f"upstream http error: {e.code}", "detail": detail}).encode("utf-8")
            )
        except URLError as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"network error: {e.reason}"}).encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


def main():
    default_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="SU2 dashboard static+SSE server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--root", default=str(default_root))
    p.add_argument("--poll-ms", type=int, default=500)
    p.add_argument("--chat-model", default=os.environ.get("SU2_DASHBOARD_CHAT_MODEL", "gpt-4o-mini"))
    p.add_argument(
        "--allowed-tailscale-login",
        default=os.environ.get("SU2_DASHBOARD_ALLOWED_TS_LOGIN", os.environ.get("SU2_DASHBOARD_ALLOWED_USER", "")),
        help="Allow requests with matching Tailscale user login header (email).",
    )
    p.add_argument(
        "--auth-token",
        default=os.environ.get("SU2_DASHBOARD_AUTH_TOKEN", ""),
        help="Shared token fallback (Authorization: Bearer <token>, X-Auth-Token, or ?token=...).",
    )
    p.add_argument(
        "--cors-origin",
        default=os.environ.get("SU2_DASHBOARD_CORS_ORIGIN", "*"),
        help="Optional CORS allow origin (for external dashboard UI).",
    )
    p.add_argument(
        "--no-protect-results",
        action="store_true",
        help="Do not require auth for /results/* files.",
    )
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    DashboardHandler.root = root
    DashboardHandler.poll_interval = max(0.05, args.poll_ms / 1000.0)
    DashboardHandler.chat_model = args.chat_model
    DashboardHandler.allowed_tailscale_login = args.allowed_tailscale_login
    DashboardHandler.auth_token = args.auth_token
    DashboardHandler.cors_origin = args.cors_origin
    DashboardHandler.protect_results = not args.no_protect_results

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Serving {root} at http://{args.host}:{args.port}")
    print("SSE endpoint: /events")
    print(f"Chat endpoint: /chat (model={DashboardHandler.chat_model})")
    if DashboardHandler.allowed_tailscale_login or DashboardHandler.auth_token:
        print("Auth enabled:")
        if DashboardHandler.allowed_tailscale_login:
            print(f"  - Tailscale login allowlist: {DashboardHandler.allowed_tailscale_login}")
        if DashboardHandler.auth_token:
            print("  - Shared auth token: enabled")
        print(f"  - Protect /results/*: {DashboardHandler.protect_results}")
    else:
        print("Auth disabled (no allowlist or token configured)")
    server.serve_forever()


if __name__ == "__main__":
    main()
