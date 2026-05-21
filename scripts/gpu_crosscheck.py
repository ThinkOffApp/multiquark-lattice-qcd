#!/usr/bin/env python3
"""DEPRECATED: gpu_crosscheck.py is no longer a Metal parity gate.

Why this script was retired (see issue #9, PRs #10 / #9b):

The original script ran ``gpt/applications/hmc/su2_2q_signal_scan.py`` twice
with ``--backend cpu`` and ``--backend metal``, then compared per-measurement
plaquettes / Wilson loops. Three independent reasons that gate cannot
produce a meaningful Metal pass on this codebase:

1. The driver does not parse ``--backend``. ``su2_2q_signal_scan.py`` only
   reads ``--pipeline-label`` (an opaque label) and ``--require-accelerator``.
   So both legs ran the exact same code path on the exact same cgpt build.
2. cgpt's mem-accounting only populates ``accelerator_total`` under
   ``#ifdef GRID_CUDA`` (``gpt/lib/cgpt/lib/util.cc``); on Apple Metal the
   counter is always zero. The ``accelerator_total > 0`` gate is unobservable
   on Apple even when Metal kernels are firing.
3. ``su2_2q_signal_scan.py`` is pure-gauge SU(2): heat-bath, Wilson **gauge**
   loops, Polyakov, multilevel/multihit. It does not exercise the
   ``WilsonKernels`` Metal Dslash path at all. CPU-vs-Metal parity through
   this driver is tautological.

The replacement gate is a Grid-level Wilson Dslash regression that actually
exercises the Metal kernel and produces a structured receipt:

    scripts/metal_parity_gate.sh

That script verifies a Metal-enabled cgpt is loaded, that the Grid configure
summary reports ``Acceleration: metal``, optionally runs Grid's own
``Benchmark_wilson`` to demonstrate ``AcceleratorMetalInit``, and exits
non-zero unless the receipt is complete.

For CPU-only output-schema and dashboard read-path receipts, see
``scripts/smoke_test.sh``.
"""

import sys

REPLACEMENT = "scripts/metal_parity_gate.sh"
EXIT_DEPRECATED = 2


def main() -> int:
    msg = (
        "ERROR: scripts/gpu_crosscheck.py has been retired.\n"
        "       It is not a valid Metal parity gate. See issue #9 for the\n"
        "       full rationale. Run the replacement gate instead:\n\n"
        f"         {REPLACEMENT}\n\n"
        "       This stub exits non-zero so any CI/cron job that was wired\n"
        "       to the old script fails loudly instead of silently passing\n"
        "       on what was always CPU-vs-CPU.\n"
    )
    sys.stderr.write(msg)
    return EXIT_DEPRECATED


if __name__ == "__main__":
    raise SystemExit(main())
