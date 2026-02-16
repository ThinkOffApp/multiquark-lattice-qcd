# Petrus Pennanen - Lattice QCD Flux-Tube Program

This repository contains the active Grid+GPT workflow for high-statistics lattice QCD measurements of static-source potentials and chromofield flux profiles.

## Scientific Scope

- Reproduce and extend earlier SU(2) flux-tube studies with better statistics and larger lattices.
- Then move to 6-quark systems in SU(3), using gauge configurations from collaborations.
- Main target: resolve flux-tube structure between two nucleons and study its relevance for nuclear fusion dynamics.

## Core Algorithms

- Monte Carlo gauge-field evolution with thermalization + production separation.
- Wilson-loop based extraction of static observables across multiple \(R,T\) geometries.
- Connected correlator measurements for flux observables.
- Gauge-link smearing and operator averaging for overlap improvement.
- Jackknife-based uncertainty estimation with covariance-aware fitting.
- Autocorrelation-aware handling of measurement streams (binning / effective sample control).

## Signal and Performance Optimizations

- Optional all-sample mode (admin toggle) vs quality-filtered mode.
- Raw flux and vacuum/tail-subtracted flux switches in dashboard views.
- Safer checkpoint/resume with parameter validation.
- Missing-config backfill tooling for exact target coverage.
- Reduced dashboard/server overhead via cached state updates and lighter refresh paths.
- Batched processing and reduced progress I/O frequency for faster runs.

## Selected Earlier SU(2) Papers

- P. Pennanen, A. M. Green, C. Michael, *Flux-tube structure and beta-functions in SU(2)*, [arXiv:hep-lat/9705033](https://arxiv.org/abs/hep-lat/9705033)
- A. M. Green, P. Pennanen, C. Michael, *Flux-tube Structure, Sum Rules and Beta-functions in SU(2)*, [arXiv:hep-lat/9708012](https://arxiv.org/abs/hep-lat/9708012)
- A. M. Green, P. Pennanen, *An interquark potential model for multi-quark systems*, [arXiv:hep-lat/9804003](https://arxiv.org/abs/hep-lat/9804003)
- P. Pennanen, A. M. Green, C. Michael, *Four-quark flux distribution and binding in lattice SU(2)*, [arXiv:hep-lat/9804004](https://arxiv.org/abs/hep-lat/9804004)

## Repository Map

- `gpt/applications/hmc/su2_2q_signal_scan.py` - measurement engine
- `tools/su2_signal_postprocess.py` - postprocessing + fit/error pipeline
- `tools/su2_dashboard_server.py` - live dashboard backend
- `tools/su2_dashboard.html` - dashboard frontend
- `tools/su2_chain_to_24.py` - chained run orchestration

