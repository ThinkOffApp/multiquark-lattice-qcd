"""The telemetry payload must carry the host's physical memory so the
next-jobs panel can judge feasibility (issue #49).
Run: python3 -m unittest tools/test_dashboard_host_info.py"""
import importlib.util
import pathlib
import sys
import unittest

TOOLS = pathlib.Path(__file__).resolve().parent


def load_server():
    spec = importlib.util.spec_from_file_location("su2_dashboard_server", TOOLS / "su2_dashboard_server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.argv = [str(TOOLS / "su2_dashboard_server.py")]
    spec.loader.exec_module(mod)
    return mod


SERVER = load_server()


class HostInfo(unittest.TestCase):
    def test_host_memory_is_positive_and_cached(self):
        a = SERVER.DashboardHandler.host_info()
        b = SERVER.DashboardHandler.host_info()
        self.assertIsInstance(a["mem_total_bytes"], int)
        self.assertGreater(a["mem_total_bytes"], 1 << 30)
        self.assertEqual(a, b)

    def test_telemetry_payload_carries_host(self):
        payload = SERVER.DashboardHandler.collect_thread_telemetry("test-no-such-seed-49")
        self.assertIn("host", payload)
        self.assertEqual(payload["host"]["mem_total_bytes"], SERVER.DashboardHandler.host_info()["mem_total_bytes"])


if __name__ == "__main__":
    unittest.main()
