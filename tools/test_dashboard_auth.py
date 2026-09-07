"""Identity headers are believed only from the proxy that sets them; query
tokens only where the browser has no header channel (issue #37).
Run: python3 -m unittest tools/test_dashboard_auth.py"""
import email.message
import importlib.util
import io
import pathlib
import sys
import unittest
from urllib.parse import urlparse

TOOLS = pathlib.Path(__file__).resolve().parent


def load_server():
    spec = importlib.util.spec_from_file_location("su2_dashboard_server", TOOLS / "su2_dashboard_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [str(TOOLS / "su2_dashboard_server.py")]
    spec.loader.exec_module(mod)
    return mod


SERVER = load_server()
LOGIN = "petrus@example.com"
TOKEN = "s3cret-token"


def handler(peer, headers=None, *, login=LOGIN, token="", proxies=()):
    """A DashboardHandler with no socket: only the auth methods are exercised."""
    h = SERVER.DashboardHandler.__new__(SERVER.DashboardHandler)
    h.client_address = (peer, 40000)
    h.headers = email.message.Message()
    for k, v in (headers or {}).items():
        h.headers[k] = v
    h.allowed_tailscale_login = login
    h.auth_token = token
    h.trusted_proxies = tuple(proxies)
    return h


class IdentityHeaders(unittest.TestCase):
    def test_header_from_lan_peer_is_ignored(self):
        h = handler("192.168.0.77", {"Tailscale-User-Login": LOGIN})
        self.assertEqual(h.get_tailscale_login(), "")
        self.assertFalse(h.is_authorized(parsed=urlparse("/thread_control")))

    def test_header_from_loopback_proxy_is_honoured(self):
        h = handler("127.0.0.1", {"Tailscale-User-Login": LOGIN.upper()})
        self.assertEqual(h.get_tailscale_login(), LOGIN)
        self.assertTrue(h.is_authorized(parsed=urlparse("/thread_control")))

    def test_ipv6_and_mapped_loopback_are_loopback(self):
        for peer in ("::1", "::ffff:127.0.0.1"):
            h = handler(peer, {"X-Tailscale-User-Login": LOGIN})
            self.assertTrue(h.is_authorized(), peer)

    def test_explicit_trusted_proxy_cidr(self):
        h = handler("100.64.0.9", {"Tailscale-User-Login": LOGIN}, proxies=("100.64.0.0/24",))
        self.assertTrue(h.is_authorized())
        h = handler("100.64.1.9", {"Tailscale-User-Login": LOGIN}, proxies=("100.64.0.0/24",))
        self.assertFalse(h.is_authorized())

    def test_wrong_login_from_loopback_still_denied(self):
        h = handler("127.0.0.1", {"Tailscale-User-Login": "someone@else.example"})
        self.assertFalse(h.is_authorized())


class Tokens(unittest.TestCase):
    def test_header_and_body_tokens_work_everywhere(self):
        h = handler("192.168.0.77", {"Authorization": f"Bearer {TOKEN}"}, login="", token=TOKEN)
        self.assertTrue(h.is_authorized(parsed=urlparse("/thread_control")))
        h = handler("192.168.0.77", {"X-Auth-Token": TOKEN}, login="", token=TOKEN)
        self.assertTrue(h.is_authorized(parsed=urlparse("/thread_telemetry?seed=9101")))
        h = handler("192.168.0.77", login="", token=TOKEN)
        self.assertTrue(h.is_authorized(payload={"token": TOKEN}))

    def test_query_token_rejected_except_on_events(self):
        h = handler("192.168.0.77", login="", token=TOKEN)
        parsed = urlparse(f"/thread_telemetry?seed=9101&token={TOKEN}")
        self.assertFalse(h.is_authorized(parsed=parsed))
        parsed = urlparse(f"/events?seed=9101&token={TOKEN}")
        self.assertTrue(h.is_authorized(parsed=parsed, allow_query_token=True))
        self.assertFalse(h.is_authorized(parsed=urlparse("/events?seed=9101&token=wrong"), allow_query_token=True))

    def test_access_log_redacts_query_token(self):
        h = handler("192.168.0.77", login="", token=TOKEN)
        h.requestline = f"GET /events?seed=9101&token={TOKEN} HTTP/1.1"
        h.request_version = "HTTP/1.1"
        h.command = "GET"
        h.path = f"/events?seed=9101&token={TOKEN}"
        err = io.StringIO()
        real = sys.stderr
        sys.stderr = err
        try:
            h.log_request(200)
        finally:
            sys.stderr = real
        out = err.getvalue()
        self.assertNotIn(TOKEN, out)
        self.assertIn("token=<redacted>", out)

    def test_no_auth_configured_stays_open(self):
        h = handler("192.168.0.77", login="", token="")
        self.assertTrue(h.is_authorized(parsed=urlparse("/thread_control")))


if __name__ == "__main__":
    unittest.main()
