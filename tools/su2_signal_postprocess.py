#!/usr/bin/env python3
"""High-statistics postprocessing for SU(2) signal scans.

Implements improved estimators:
- all-sample ensemble correlator fits (no per-config sign cuts)
- jackknife uncertainties
- autocorrelation-aware binning
- correlated covariance fits with shrinkage + SVD cutoff
- fit-window scan/tuning over Tmin/Tmax
- optional multi-seed combination
- consistent flux vacuum subtraction in analysis
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class FitResult:
    R: int
    V: float
    err: float
    amp: float
    amp_err: float
    tmin: int
    tmax: int
    chi2: float
    dof: int
    chi2_dof: float
    n_bins: int
    n_cfg: int


def load_live(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    meta = data.get("meta", {})
    jsonl_name = meta.get("jsonl_path")
    # If measurements missing or empty, try JSONL
    if not data.get("measurements") and jsonl_name:
        jsonl_path = path.parent / jsonl_name
        if jsonl_path.exists():
            meas = []
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            meas.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            data["measurements"] = meas
    return data


def estimate_tau_int(series: Sequence[float], max_lag: int = 1000, c_window: float = 4.0) -> float:
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 16:
        return 0.5
    
    mu = np.mean(x)
    x = x - mu
    var = np.mean(x**2)
    if var <= 0:
        return 0.5
        
    rhos = []
    # Compute autocorrelations up to max_lag
    # For speed on large N, FFT could be used, but direct sum is fine for typical lattice N
    for lag in range(1, min(n // 2, max_lag) + 1):
        gamma = np.mean(x[:-lag] * x[lag:])
        rhos.append(gamma / var)
    
    tau = 0.5
    for W, rho in enumerate(rhos, 1):
        tau += rho
        # Madras-Sokal self-consistent window.
        if W >= c_window * max(0.5, tau):
            break

    return max(0.5, float(tau))


def mean_measurement_items(items: Sequence[dict]) -> dict:
    if len(items) == 1:
        return items[0]

    out = {
        "plaquette": float(np.mean([x["plaquette"] for x in items])),
        "loops": {},
        "flux_profile_r_perp": [],
    }

    keys = list(items[0]["loops"].keys())
    for k in keys:
        out["loops"][k] = {
            "re": float(np.mean([x["loops"][k]["re"] for x in items])),
            "im": float(np.mean([x["loops"][k]["im"] for x in items])),
        }

    m = len(items[0]["flux_profile_r_perp"])
    for j in range(m):
        out["flux_profile_r_perp"].append(float(np.mean([x["flux_profile_r_perp"][j] for x in items])))

    return out


def bin_measurements(measurements: Sequence[dict], bin_size: int) -> List[dict]:
    if bin_size <= 1:
        return list(measurements)
    out = []
    n = len(measurements)
    nb = n // bin_size
    for b in range(nb):
        chunk = measurements[b * bin_size : (b + 1) * bin_size]
        out.append(mean_measurement_items(chunk))
    rem = n % bin_size
    if rem > 0 and rem >= max(2, bin_size // 2):
        out.append(mean_measurement_items(measurements[-rem:]))
    return out


def make_cov_jackknife(y_jk: np.ndarray) -> np.ndarray:
    # y_jk shape = (njack, nt)
    njack = y_jk.shape[0]
    ybar = np.mean(y_jk, axis=0)
    dy = y_jk - ybar
    cov = (njack - 1.0) / njack * (dy.T @ dy)
    return cov


def regularized_inverse(cov: np.ndarray, shrinkage: float, svd_rcond: float) -> np.ndarray:
    d = np.diag(np.diag(cov))
    c = (1.0 - shrinkage) * cov + shrinkage * d
    u, s, vt = np.linalg.svd(c, full_matrices=False)
    smax = float(np.max(s)) if len(s) else 0.0
    cutoff = svd_rcond * smax
    sinv = np.array([1.0 / x if x > cutoff else 0.0 for x in s], dtype=float)
    return (vt.T * sinv) @ u.T


def weighted_linear_fit(T: np.ndarray, y: np.ndarray, cov_inv: np.ndarray) -> Tuple[np.ndarray, float, int]:
    # Model: y = a - V * T
    X = np.column_stack([np.ones_like(T), -T])
    xtwx = X.T @ cov_inv @ X
    xtwy = X.T @ cov_inv @ y
    beta = np.linalg.pinv(xtwx) @ xtwy
    resid = y - X @ beta
    chi2 = float(resid.T @ cov_inv @ resid)
    dof = max(0, len(T) - 2)
    return beta, chi2, dof


def collect_loop_matrix(binned: Sequence[dict], R: int, Ts: Sequence[int]) -> np.ndarray:
    arr = np.zeros((len(binned), len(Ts)), dtype=float)
    for i, m in enumerate(binned):
        loops = m.get("loops", {})
        for j, t in enumerate(Ts):
            arr[i, j] = float(((loops.get(f"R{R}_T{t}") or {}).get("re")) or 0.0)
    return arr


def select_windows(Ts: Sequence[int], tmin: int | None, tmax: int | None, min_points: int) -> List[Tuple[int, int]]:
    Ts = sorted(int(t) for t in Ts)
    if tmin is not None and tmax is not None:
        return [(tmin, tmax)]
    wins = []
    for i in range(len(Ts)):
        for j in range(i + min_points - 1, len(Ts)):
            wins.append((Ts[i], Ts[j]))
    return wins


def fit_potential_for_R(
    R: int,
    Ts: Sequence[int],
    binned: Sequence[dict],
    raw_ncfg: int,
    shrinkage: float,
    svd_rcond: float,
    tmin: int | None,
    tmax: int | None,
    min_points: int,
) -> FitResult | None:
    if len(binned) < 4:
        return None

    Ts_sorted = sorted(int(t) for t in Ts)
    W = collect_loop_matrix(binned, R, Ts_sorted)
    nbin = W.shape[0]
    windows = select_windows(Ts_sorted, tmin, tmax, min_points)

    best = None

    for lo, hi in windows:
        idx = [k for k, t in enumerate(Ts_sorted) if lo <= t <= hi]
        if len(idx) < 2:
            continue

        Tsel = np.asarray([Ts_sorted[k] for k in idx], dtype=float)
        Wsel = W[:, idx]

        cmean = np.mean(Wsel, axis=0)
        if np.any(cmean <= 0.0):
            continue
        y = np.log(cmean)

        y_jk = []
        valid = True
        for leave in range(nbin):
            mask = np.ones(nbin, dtype=bool)
            mask[leave] = False
            cjk = np.mean(Wsel[mask, :], axis=0)
            if np.any(cjk <= 0.0):
                valid = False
                break
            y_jk.append(np.log(cjk))
        if not valid:
            continue

        y_jk = np.asarray(y_jk, dtype=float)
        cov = make_cov_jackknife(y_jk)
        cov_inv = regularized_inverse(cov, shrinkage=shrinkage, svd_rcond=svd_rcond)

        beta, chi2, dof = weighted_linear_fit(Tsel, y, cov_inv)
        amp = float(math.exp(beta[0]))
        V = float(beta[1])
        if not np.isfinite(amp) or not np.isfinite(V):
            continue

        # Jackknife error: for each leave-one-out pseudo-sample, rebuild the
        # jackknife covariance from the remaining pseudo-samples.
        V_jk = []
        A_jk = []
        for leave in range(nbin):
            yk = y_jk[leave, :]
            y_jk_loo = np.delete(y_jk, leave, axis=0)
            if y_jk_loo.shape[0] >= 2:
                cov_loo = make_cov_jackknife(y_jk_loo)
                cov_inv_loo = regularized_inverse(cov_loo, shrinkage=shrinkage, svd_rcond=svd_rcond)
            else:
                cov_inv_loo = cov_inv
            bk, _, _ = weighted_linear_fit(Tsel, yk, cov_inv_loo)
            V_jk.append(float(bk[1]))
            A_jk.append(float(math.exp(bk[0])))
        V_jk = np.asarray(V_jk, dtype=float)
        A_jk = np.asarray(A_jk, dtype=float)
        V_err = float(np.sqrt((nbin - 1.0) / nbin * np.sum((V_jk - np.mean(V_jk)) ** 2)))
        A_err = float(np.sqrt((nbin - 1.0) / nbin * np.sum((A_jk - np.mean(A_jk)) ** 2)))

        chi2_dof = float(chi2 / dof) if dof > 0 else float("inf")

        # Fit-window tuning score: prefer chi2/dof near 1, then smaller uncertainty.
        score = abs(chi2_dof - 1.0) + 0.15 * V_err
        cand = {
            "score": score,
            "fit": FitResult(
                R=R,
                V=V,
                err=V_err,
                amp=amp,
                amp_err=A_err,
                tmin=lo,
                tmax=hi,
                chi2=chi2,
                dof=dof,
                chi2_dof=chi2_dof,
                n_bins=nbin,
                n_cfg=raw_ncfg,
            ),
        }

        if best is None or cand["score"] < best["score"]:
            best = cand

    return None if best is None else best["fit"]


def flux_stats(binned: Sequence[dict], vacuum_tail: int) -> dict:
    if not binned:
        return {"mean": [], "err": [], "n_bins": 0}
    m = len(binned[0].get("flux_profile_r_perp", []))
    arr = np.zeros((len(binned), m), dtype=float)
    for i, x in enumerate(binned):
        arr[i, :] = np.asarray(x.get("flux_profile_r_perp", [0.0] * m), dtype=float)

    # Apply vacuum subtraction per-bin to correctly propagate variance
    if vacuum_tail > 0 and m >= vacuum_tail:
        # Calculate tail mean for each sample (axis=1)
        tail_means = np.mean(arr[:, -vacuum_tail:], axis=1, keepdims=True)
        arr = arr - tail_means

    n = len(binned)
    if n > 1:
        # Construct Jackknife samples for the mean profile
        sum_all = np.sum(arr, axis=0)
        jk_samples = (sum_all[None, :] - arr) / (n - 1)
        
        cov = make_cov_jackknife(jk_samples)
        err = np.sqrt(np.diag(cov))
    else:
        err = np.zeros(m, dtype=float)

    mean_prof = np.mean(arr, axis=0)

    return {
        "mean": [float(x) for x in mean_prof],
        "err": [float(x) for x in err],
        "n_bins": n,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Postprocess SU2 signal scans with high-signal estimators")
    p.add_argument(
        "--live",
        action="append",
        default=[],
        help="Path to live_<seed>.json (repeatable).",
    )
    p.add_argument(
        "--out",
        default="",
        help="Optional JSON output path.",
    )
    p.add_argument("--bin-size", type=int, default=0, help="Fixed bin size (0 => auto from autocorr).")
    p.add_argument("--max-lag", type=int, default=200, help="Max lag for autocorrelation estimate.")
    p.add_argument("--shrinkage", type=float, default=0.15, help="Covariance shrinkage to diagonal [0,1].")
    p.add_argument("--svd-rcond", type=float, default=1e-10, help="Relative SVD cutoff for covariance inverse.")
    p.add_argument("--fit-tmin", type=int, default=None, help="Force Tmin for fits.")
    p.add_argument("--fit-tmax", type=int, default=None, help="Force Tmax for fits.")
    p.add_argument("--min-fit-points", type=int, default=3, help="Minimum T points per fit window.")
    p.add_argument(
        "--flux-vacuum-tail",
        type=int,
        default=2,
        help="Subtract mean of last N r_perp points from flux profile in analysis.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.live:
        raise SystemExit("Provide at least one --live file")

    live_paths = [Path(x).expanduser().resolve() for x in args.live]
    runs = [load_live(p) for p in live_paths]

    # Basic consistency from first run.
    meta0 = runs[0].get("meta", {})
    Rs = [int(x) for x in meta0.get("R", [])]
    Ts = [int(x) for x in meta0.get("T", [])]

    all_binned = []
    tau_by_seed: Dict[str, float] = {}
    bin_by_seed: Dict[str, int] = {}
    total_cfg = 0

    for run, path in zip(runs, live_paths):
        meta = run.get("meta", {})
        seed = str(meta.get("seed", path.stem))
        ms = list(run.get("measurements", []))
        if not ms:
            continue

        total_cfg += len(ms)
        plaq = [float(x.get("plaquette", 0.0)) for x in ms]
        tau = estimate_tau_int(plaq, max_lag=args.max_lag)
        tau_by_seed[seed] = tau

        if args.bin_size > 0:
            bsz = args.bin_size
        else:
            bsz = max(1, int(math.ceil(2.0 * tau)))
        bin_by_seed[seed] = bsz

        binned = bin_measurements(ms, bsz)
        all_binned.extend(binned)

    if len(all_binned) < 4:
        raise SystemExit("Not enough binned samples after preprocessing")

    fits = []
    for R in Rs:
        fr = fit_potential_for_R(
            R=R,
            Ts=Ts,
            binned=all_binned,
            raw_ncfg=total_cfg,
            shrinkage=float(np.clip(args.shrinkage, 0.0, 1.0)),
            svd_rcond=max(1e-16, float(args.svd_rcond)),
            tmin=args.fit_tmin,
            tmax=args.fit_tmax,
            min_points=max(2, args.min_fit_points),
        )
        if fr is not None:
            fits.append(fr)

    flux = flux_stats(all_binned, vacuum_tail=max(0, args.flux_vacuum_tail))

    output = {
        "summary": {
            "n_live_files": len(live_paths),
            "n_cfg_total": total_cfg,
            "n_binned_total": len(all_binned),
            "tau_int_by_seed": tau_by_seed,
            "bin_size_by_seed": bin_by_seed,
            "settings": {
                "bin_size": args.bin_size,
                "max_lag": args.max_lag,
                "shrinkage": args.shrinkage,
                "svd_rcond": args.svd_rcond,
                "fit_tmin": args.fit_tmin,
                "fit_tmax": args.fit_tmax,
                "min_fit_points": args.min_fit_points,
                "flux_vacuum_tail": args.flux_vacuum_tail,
            },
        },
        "potential": [
            {
                "R": f.R,
                "V": f.V,
                "err": f.err,
                "amp": f.amp,
                "amp_err": f.amp_err,
                "tmin": f.tmin,
                "tmax": f.tmax,
                "chi2": f.chi2,
                "dof": f.dof,
                "chi2_dof": f.chi2_dof,
                "n_bins": f.n_bins,
                "n_cfg": f.n_cfg,
            }
            for f in fits
        ],
        "flux_profile": flux,
    }

    print(
        f"combined cfg={total_cfg}, binned={len(all_binned)}, "
        f"tau_int~{max(tau_by_seed.values()) if tau_by_seed else 0.0:.2f}"
    )
    for f in fits:
        print(
            f"R={f.R}: V={f.V:.6f} +/- {f.err:.6f} "
            f"[T={f.tmin}..{f.tmax}, chi2/dof={f.chi2_dof:.3f}, bins={f.n_bins}]"
        )

    if args.out:
        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fp:
            json.dump(output, fp, indent=2)
        print(f"wrote {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
