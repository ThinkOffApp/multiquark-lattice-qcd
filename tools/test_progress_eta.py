"""#48: ETA must be based on work done in THIS process. Numbers from the
seed 9101 resume on 2026-09-06 (29 measurements from July, resumed at 22:37)."""
import importlib.util
import pathlib
import unittest

MOD = pathlib.Path(__file__).resolve().parents[1] / "gpt" / "applications" / "hmc" / "su2_eta.py"
spec = importlib.util.spec_from_file_location("su2_eta", MOD)
su2_eta = importlib.util.module_from_spec(spec)
spec.loader.exec_module(su2_eta)
E = su2_eta.estimate_eta_sec


class EtaAfterResume(unittest.TestCase):
    def test_no_eta_before_first_measurement_in_this_process(self):
        eta, src = E(phase="production", nmeas=200, meas_done=29, meas_done_at_start=29,
                     total_sweeps=1200, sweeps_done=345, sweeps_done_at_start=345, elapsed_sec=436.8)
        self.assertIsNone(eta)
        self.assertIsNone(src)

    def test_cadence_after_one_measurement(self):
        eta, src = E(phase="production", nmeas=200, meas_done=30, meas_done_at_start=29,
                     total_sweeps=1200, sweeps_done=350, sweeps_done_at_start=345, elapsed_sec=720.0)
        self.assertEqual(src, "measurement-cadence")
        self.assertAlmostEqual(eta, 720.0 * 170, places=6)  # about 34 h, not 14 min

    def test_cadence_uses_only_this_process(self):
        # Same cumulative counts as the old formula saw, but the rate must
        # come from the 3 measurements done here, not the 32 total.
        eta, _ = E(phase="production", nmeas=200, meas_done=32, meas_done_at_start=29,
                   total_sweeps=1200, sweeps_done=360, sweeps_done_at_start=345, elapsed_sec=2160.0)
        self.assertAlmostEqual(eta, 2160.0 / 3 * 168, places=6)

    def test_thermalization_uses_sweep_rate_of_this_process(self):
        eta, src = E(phase="thermalization", nmeas=200, meas_done=0, meas_done_at_start=0,
                     total_sweeps=1200, sweeps_done=50, sweeps_done_at_start=40, elapsed_sec=100.0)
        self.assertEqual(src, "sweep-rate")
        self.assertAlmostEqual(eta, 100.0 / 10 * 1150, places=6)

    def test_done_or_garbage_gives_none(self):
        self.assertEqual(E(phase="production", nmeas=200, meas_done=200, meas_done_at_start=29,
                           total_sweeps=1200, sweeps_done=1200, sweeps_done_at_start=345, elapsed_sec=10.0), (None, None))
        self.assertEqual(E(phase="production", nmeas=200, meas_done=30, meas_done_at_start=29,
                           total_sweeps=1200, sweeps_done=350, sweeps_done_at_start=345, elapsed_sec=None), (None, None))


if __name__ == "__main__":
    unittest.main()
