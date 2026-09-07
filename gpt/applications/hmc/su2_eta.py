"""ETA for the SU(2) signal-scan driver, kept pure so it can be unit-tested
without importing gpt/cgpt.

Issue #48: the old formula divided the PER-PROCESS elapsed time into a
CUMULATIVE sweep count after a resume, so a run that had 29 measurements
from July and 7 minutes of new wall time claimed 14 minutes for the 171
that remained (about 34 hours in reality). Rates must only ever use work
completed in this process.
"""


def estimate_eta_sec(
    *,
    phase,
    nmeas,
    meas_done,
    meas_done_at_start,
    total_sweeps,
    sweeps_done,
    sweeps_done_at_start,
    elapsed_sec,
):
    """Return (eta_sec or None, source or None).

    - In production the cadence of whole measurements completed in this
      process is the only honest rate (a measurement includes its skip
      sweeps and the loop/flux reads).
    - Before production (thermalization) the sweep rate of this process is
      used.
    - With no completed unit of work in this process there is no ETA.
    """
    try:
        elapsed = float(elapsed_sec)
    except (TypeError, ValueError):
        return None, None
    if elapsed <= 0:
        return None, None
    meas_here = int(meas_done) - int(meas_done_at_start or 0)
    if str(phase) == "production":
        if meas_here >= 1 and int(meas_done) < int(nmeas):
            return elapsed / meas_here * (int(nmeas) - int(meas_done)), "measurement-cadence"
        return None, None
    sweeps_here = int(sweeps_done) - int(sweeps_done_at_start or 0)
    if sweeps_here > 0 and int(sweeps_done) < int(total_sweeps):
        return elapsed / sweeps_here * (int(total_sweeps) - int(sweeps_done)), "sweep-rate"
    return None, None
