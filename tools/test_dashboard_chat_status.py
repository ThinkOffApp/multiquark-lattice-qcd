"""The admin chat must not claim readiness without a key; the status endpoint
reports availability and model id, never the key (issue #50).
Run: python3 -m unittest tools/test_dashboard_chat_status.py"""
import importlib.util
import io
import json
import os
import pathlib
import sys
import unittest
from unittest import mock

TOOLS = pathlib.Path(__file__).resolve().parent


def load_server():
    spec = importlib.util.spec_from_file_location("su2_dashboard_server", TOOLS / "su2_dashboard_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [str(TOOLS / "su2_dashboard_server.py")]
    spec.loader.exec_module(mod)
    return mod


SERVER = load_server()


def get(path):
    """Drive do_GET on a socketless handler and return (status, json body)."""
    h = SERVER.DashboardHandler.__new__(SERVER.DashboardHandler)
    h.client_address = ("127.0.0.1", 40000)
    h.headers = {}
    h.request_version = "HTTP/1.1"
    h.command = "GET"
    h.path = path
    h.requestline = f"GET {path} HTTP/1.1"
    h.wfile = io.BytesIO()
    h.log_request = lambda *a, **k: None
    h.do_GET()
    raw = h.wfile.getvalue().decode("utf-8", "replace")
    status = int(raw.split(" ", 2)[1])
    body = raw.split("\r\n\r\n", 1)[1]
    return status, json.loads(body)


class ChatStatus(unittest.TestCase):
    def test_no_key_reports_unavailable_with_reason(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            st = SERVER.DashboardHandler.chat_status()
        self.assertFalse(st["available"])
        self.assertIn("OPENAI_API_KEY", st["reason"])
        self.assertEqual(st["model"], SERVER.DashboardHandler.chat_model)

    def test_key_present_reports_available_and_never_the_key(self):
        secret = "sk-test-0123456789abcdef"
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": secret}):
            status, body = get("/chat/status?_t=1")
        self.assertEqual(status, 200)
        self.assertTrue(body["available"])
        self.assertEqual(body["model"], SERVER.DashboardHandler.chat_model)
        self.assertNotIn(secret, json.dumps(body))

    def test_endpoint_without_key(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            status, body = get("/chat/status")
        self.assertEqual(status, 200)
        self.assertFalse(body["available"])

    def test_model_default_is_configuration(self):
        self.assertEqual(SERVER.DashboardHandler.chat_model, SERVER.DEFAULT_CHAT_MODEL)
        with mock.patch.object(SERVER.DashboardHandler, "chat_model", "some-configured-model"):
            self.assertEqual(SERVER.DashboardHandler.chat_status()["model"], "some-configured-model")


if __name__ == "__main__":
    unittest.main()
