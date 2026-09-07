"""The server's telemetry seed group and the page's thread slots must agree
(issue #52: they were -e..-h versus -b..-d, so telemetry never matched).
Run: python3 -m pytest tools/test_dashboard_seed_group.py  (or unittest)."""
import importlib.util
import pathlib
import re
import sys
import unittest

TOOLS = pathlib.Path(__file__).resolve().parent


def load_server():
    spec = importlib.util.spec_from_file_location("su2_dashboard_server", TOOLS / "su2_dashboard_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [str(TOOLS / "su2_dashboard_server.py")]  # argparse safety if the module reads argv at import
    spec.loader.exec_module(mod)
    return mod


def page_suffixes():
    html = (TOOLS / "su2_dashboard.html").read_text(encoding="utf-8")
    block = re.search(r"const threadSlots = \[(.*?)\];", html, re.S)
    assert block, "threadSlots not found in su2_dashboard.html"
    return re.findall(r'suffix:\s*"([^"]*)"', block.group(1))


class SeedGroupMatchesPage(unittest.TestCase):
    def test_server_group_equals_page_slots(self):
        server = load_server()
        group = server.DashboardHandler.seed_group("9101")
        expected = [f"9101{s}" for s in page_suffixes()]
        self.assertEqual(group, expected)

    def test_root_seed_is_first(self):
        server = load_server()
        self.assertEqual(server.DashboardHandler.seed_group("9101")[0], "9101")


if __name__ == "__main__":
    unittest.main()
