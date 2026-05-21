#!/usr/bin/env python3
"""Fail-closed regression tests for detect_runtime_backend (#9a / PR #10).

Per @CodexMB's review on PR #10: the detector must treat an empty / "none" /
"cpu" / unknown value of grid_acceleration as a non-accelerator (fall through
to backend="cpu"), and only positive enumerated values {cuda, metal, sycl,
hip} as accelerator presence. We also exercise the legacy
accelerator_total_bytes>0 positive path (CUDA-style mem reporting).

The detector imports ``gpt`` at module load, which in turn triggers Grid
init. To keep the test fast and hermetic, we inject lightweight stand-ins
into ``sys.modules`` before loading the driver module, then monkey-patch
``detect_grid_build_info`` on the driver module for each case.

Run:
    python3 scripts/test_detect_runtime_backend.py

Exits 0 on all-green, 1 on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER_PATH = REPO_ROOT / "gpt" / "applications" / "hmc" / "su2_2q_signal_scan.py"


def _install_fake_gpt(accelerator_total: int = 0, accelerator_available: int = 0):
    """Inject a stub ``gpt`` module before the driver imports it.

    The real gpt module imports cgpt and triggers Grid init. We only need
    ``g.mem_info()`` for detect_runtime_backend, so a tiny stub is enough.
    """
    if "gpt" in sys.modules:
        del sys.modules["gpt"]
    stub = types.ModuleType("gpt")
    stub.mem_info = lambda: {
        "accelerator_total": accelerator_total,
        "accelerator_available": accelerator_available,
    }
    # Some downstream code in the driver touches g.qcd, g.algorithms, etc.,
    # but detect_runtime_backend doesn't, so we don't stub those.
    sys.modules["gpt"] = stub


def _load_driver_module():
    """Load gpt/applications/hmc/su2_2q_signal_scan.py without running main().

    The module imports gpt at top level; install the stub first.
    """
    if "su2_2q_signal_scan" in sys.modules:
        del sys.modules["su2_2q_signal_scan"]
    spec = importlib.util.spec_from_file_location(
        "su2_2q_signal_scan", str(DRIVER_PATH)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class DetectRuntimeBackendTests(unittest.TestCase):

    def _detect(
        self,
        *,
        grid_acceleration: str | None,
        accelerator_total: int = 0,
        accelerator_available: int = 0,
    ):
        _install_fake_gpt(accelerator_total, accelerator_available)
        mod = _load_driver_module()
        mod.detect_grid_build_info = lambda: {
            "summary_path": "<fake>",
            "acceleration": grid_acceleration,
            "simd": "NEONv8",
            "threading": "yes",
        }
        return mod.detect_runtime_backend()

    # --- fail-closed cases (must report backend="cpu") -----------------

    def test_none_falls_through_to_cpu(self):
        rb = self._detect(grid_acceleration=None)
        self.assertEqual(rb["backend"], "cpu")
        self.assertIsNone(rb["grid_acceleration"])

    def test_empty_string_falls_through_to_cpu(self):
        rb = self._detect(grid_acceleration="")
        self.assertEqual(rb["backend"], "cpu")
        self.assertIsNone(rb["grid_acceleration"])

    def test_literal_none_value_falls_through_to_cpu(self):
        # Grid writes "Acceleration: none" for CPU-only builds.
        rb = self._detect(grid_acceleration="none")
        self.assertEqual(rb["backend"], "cpu")

    def test_literal_cpu_value_falls_through_to_cpu(self):
        rb = self._detect(grid_acceleration="cpu")
        self.assertEqual(rb["backend"], "cpu")

    def test_unknown_string_falls_through_to_cpu(self):
        rb = self._detect(grid_acceleration="foo")
        self.assertEqual(rb["backend"], "cpu")

    # --- fail-open cases (must report backend="gpu") -------------------

    def test_metal_summary_promotes_to_gpu(self):
        rb = self._detect(grid_acceleration="metal")
        self.assertEqual(rb["backend"], "gpu")
        self.assertEqual(rb["grid_acceleration"], "metal")

    def test_metal_summary_promotes_with_zero_byte_counter(self):
        # The whole point of #9a: Apple Metal builds have accelerator_total=0
        # because cgpt util.cc has no Metal branch, but Metal is real.
        rb = self._detect(grid_acceleration="metal", accelerator_total=0)
        self.assertEqual(rb["backend"], "gpu")
        self.assertEqual(rb["accelerator_total_bytes"], 0)

    def test_cuda_summary_promotes(self):
        rb = self._detect(grid_acceleration="cuda")
        self.assertEqual(rb["backend"], "gpu")

    def test_sycl_summary_promotes(self):
        rb = self._detect(grid_acceleration="sycl")
        self.assertEqual(rb["backend"], "gpu")

    def test_hip_summary_promotes(self):
        rb = self._detect(grid_acceleration="hip")
        self.assertEqual(rb["backend"], "gpu")

    def test_uppercase_metal_normalizes(self):
        rb = self._detect(grid_acceleration="METAL")
        self.assertEqual(rb["backend"], "gpu")

    def test_whitespace_metal_normalizes(self):
        rb = self._detect(grid_acceleration="  metal  ")
        self.assertEqual(rb["backend"], "gpu")

    # --- CUDA-style mem counter path ---------------------------------

    def test_byte_counter_positive_promotes_to_gpu(self):
        # Legacy positive signal: cudaMemGetInfo reports a nonzero pool. This
        # path must still work even when grid_acceleration is missing (e.g.
        # GRID_CONFIG_SUMMARY not set on a CUDA box).
        rb = self._detect(
            grid_acceleration=None,
            accelerator_total=8 * 1024**3,
            accelerator_available=4 * 1024**3,
        )
        self.assertEqual(rb["backend"], "gpu")
        self.assertEqual(rb["accelerator_total_bytes"], 8 * 1024**3)
        self.assertEqual(rb["accelerator_available_bytes"], 4 * 1024**3)

    # --- receipt completeness ----------------------------------------

    def test_receipt_includes_required_fields(self):
        rb = self._detect(grid_acceleration="metal")
        for key in (
            "backend",
            "accelerator_total_bytes",
            "accelerator_available_bytes",
            "grid_acceleration",
            "grid_simd",
            "grid_threading",
            "grid_summary_path",
            "loaded_cgpt_so",
            "python_version",
        ):
            self.assertIn(key, rb, f"receipt missing {key!r}")
        self.assertRegex(rb["python_version"], r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
