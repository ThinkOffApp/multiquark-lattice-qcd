# Petrus Pennanen - Lattice QCD Flux-Tube Program

This repository contains the active Grid+GPT workflow used for high-statistics lattice QCD measurements of static-source potentials and chromofield flux profiles. The current production pipeline is focused on reliable long-run operation, statistically controlled error estimates, and live observability through a streaming dashboard.

## Scientific Program

The immediate target is to reproduce and extend earlier SU(2) flux-tube studies with better statistics, larger lattices, and tighter uncertainty control. The next stage is to move to multi-quark systems in SU(3), including 6-quark physics on collaboration gauge ensembles. A central long-term goal is to resolve flux-tube structure between two nucleons and quantify how confinement-scale dynamics may connect to nuclear-fusion-relevant effective interactions.

## Methods and Physics Outputs

The measurement engine separates thermalization from production, then samples observables from Wilson-loop and connected-correlator operators across multiple geometries. Smearing, operator averaging, and jackknife-based uncertainty estimation are integrated in the default pipeline, with covariance-aware fitting in post-processing. The dashboard and postprocessor report potential and flux observables in forms directly comparable across runs and volumes, while autocorrelation diagnostics are used to monitor effective sample independence.

The static potential is fit with the Cornell form

$$
V(R) = V_0 + \sigma R - \frac{e}{R},
$$

and the dashboard shows the fitted physical parameters $(\sigma, e, V_0, \chi^2/\mathrm{dof})$ alongside the measured $V(R)$ points. Flux profiles are displayed as connected-field measurements $\Delta P(r_\perp)$, and plaquette autocorrelation is tracked via $\tau_{\mathrm{int}}$ estimators during the run.

## Platform and Runtime

This codebase is actively maintained for both Linux and macOS, including Apple Silicon systems. Production runs are regularly executed on modern M-series MacBook, Mac mini, and Mac Studio machines as well as Linux workstations and servers. The orchestration scripts are POSIX shell plus `tmux`, and path handling is now environment-aware through variables such as `SU2_OUT_DIR` and `SU2_GPT_DIR`, so the same run definitions can move between hosts without hardcoded machine paths.

## Live Run Dashboard

![SU(2) live run dashboard](images/dashboard/live_run_dashboard.png)
![SU(2) physical-comparison and diagnostics view](images/dashboard/su2_admin_progress_view_with_legend.png)

The dashboard streams live data from `progress_<seed>.json` and `live_<seed>.json` and is designed to answer two questions continuously: whether the run is healthy, and whether the physics estimates are converging. In the top dashboard view, the first section reports phase, global progress, measurement counts, and synchronized all-thread current-config progress. The Per-Thread Monitor then shows each worker's current config id, stage pipeline (`skip -> loop -> flux -> final -> done`), and live cursor positions in lattice-direction space.

The second screenshot highlights the analysis side of the same page, especially the physical-comparison diagnostics used to benchmark extracted observables against reference scales and expected behaviors. This includes potential-fit outputs (e.g., \(\sigma, e, V_0, r_0, r_1\)), flux-profile diagnostics, and agreement-status indicators that are updated live during production.

### Analytical and Diagnostic Power

Beyond tracking progress, the dashboard provides automated physics interpretation:
- **Physics Extraction**: Real-time Cornell potential fitting, Sommer scale ($r_0, r_1$) determination, and flux tube width growth analysis.
- **Statistical Health**: Integrated autocorrelation time ($\tau_{int}$) calculation and Signal-to-Noise (SNR) monitoring for all observables.
- **Phase Detection**: Polyakov loop sector visualization for monitoring $Z_2$ center symmetry and confinement transitions.
- **Physical Benchmarks**: Automatic conversion of lattice observables to physical units (GeV/fm) for direct comparison with literature.

### Visualized Sections

- **Phase and global bars**: production/thermalization state, overall progress, production measurement count, and all-thread current-config progress.
- **Per-Thread Monitor (A/B/C/D)**: per-thread config id, stage pipeline (`skip -> loop -> flux -> final -> done`), and real-time cursor positions over lattice directions.
- **Cursor colors**: blue = time direction, amber = space direction, green = other directions, magenta = per-thread config progress track.
- **Observable Charts**: plaquette history (running mean + SEM), selected Wilson-loop history, flux profile \(\Delta P(r_\perp)\), \(V(R)\) with errors and Cornell fit readout (\(\sigma, e, V_0, \chi^2/\mathrm{dof}\)), plaquette autocorrelation (\(\tau_\mathrm{int}\)), and Polyakov-loop sector tracking by direction.
- **Admin Chat + Next Jobs**: run interpretation and suggested follow-up runs.

See [SU2_DASHBOARD.md](tools/SU2_DASHBOARD.md) for detailed setup and usage instructions.

## Selected Earlier SU(2) Papers

1. P. Pennanen, A. M. Green, C. Michael, *Flux-tube structure and beta-functions in SU(2)*, [arXiv:hep-lat/9705033](https://arxiv.org/abs/hep-lat/9705033)
2. A. M. Green, P. Pennanen, C. Michael, *Flux-tube Structure, Sum Rules and Beta-functions in SU(2)*, [arXiv:hep-lat/9708012](https://arxiv.org/abs/hep-lat/9708012)
3. A. M. Green, P. Pennanen, *An interquark potential model for multi-quark systems*, [arXiv:hep-lat/9804003](https://arxiv.org/abs/hep-lat/9804003)
4. P. Pennanen, A. M. Green, C. Michael, *Four-quark flux distribution and binding in lattice SU(2)*, [arXiv:hep-lat/9804004](https://arxiv.org/abs/hep-lat/9804004)

## Scientific Background

The current implementation is directly anchored to earlier SU(2) and SU(3) flux-tube studies. The figures below from *Four-quark flux distribution and binding in lattice SU(2)* summarize the action-density and binding-structure observables that motivate the present measurement program.

### Action Density
| 2-Quark | 4-Quark Planar | 4-Quark Planar |
| :---: | :---: | :---: |
| ![Fig 4d](images/papers/github_figs/fig4d.png) | ![Fig 6d](images/papers/github_figs/fig6d.png) | ![Fig 8d](images/papers/github_figs/fig8d.png) |
| *Flux tube between two quarks* | *Four-quark flux* | *Four quark flux on a plane through the quarks* |

### Binding Action Density
| 4-Quark Planar | 4-Quark Planar |
| :---: | :---: |
| ![Fig 12c](images/papers/github_figs/fig12c.png) | ![Fig 14c](images/papers/github_figs/fig14c.png) |
| *Difference of four quark and two quark action densities* | *On a plane through the quarks* |

### First Excited State Binding Energy
| 4-Quark Planar | 4-Quark Planar |
| :---: | :---: |
| ![Fig 19b](images/papers/github_figs/fig19b.png) | ![Fig 21b](images/papers/github_figs/fig21b.png) |
| *On a plane through the quarks.* | *Binding energy of the first excited state* |

## Repository Guide

The core runtime path is `gpt/applications/hmc/su2_2q_signal_scan.py` for measurement execution, `tools/su2_signal_postprocess.py` for post-processing and fit/error analysis, `tools/su2_dashboard_server.py` for live backend streaming, and `tools/su2_dashboard.html` for the frontend. Chained production orchestration lives in `tools/su2_chain_to_24.py`.

## Contributing and Discussion

Contribution workflow and coding standards are documented in [CONTRIBUTING.md](CONTRIBUTING.md). For run coordination, API discussion, and join-endpoint behavior (including intelligent-agent integration), use the Antfarm lattice-QCD room: <https://antfarm.world/messages/room/lattice-qcd>.

## License

This project is licensed under GPL-2.0. See [LICENSE](LICENSE) for the full text.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy
./scripts/validate.sh
```

## Validation and Baseline

`./scripts/validate.sh` (or `make test`) runs a deterministic baseline validation by generating a synthetic SU(2) live dataset, executing `tools/su2_signal_postprocess.py`, and checking a stable potential fit. The baseline observable is $V(R=2)=0.310280$ on the synthetic dataset, and CI accepts a robust regression window of $0.29 \le V(R=2) \le 0.33$.
