#!/usr/bin/env python3
#
# SU(2) pure-gauge scan for:
# 1) 2q potential from Wilson loops W(R,T)
# 2) 2q flux-tube signal from connected plaquette-Wilson-loop correlator
#
# This variant defaults to an improved signal pipeline:
# - orientation averaging over all time directions
# - measurement-time stout smearing (operator-level)
# - multilevel/multihit-style averaging on copied fields
# - flux tail vacuum subtraction for stable profiles
#
import json
import math
import os
import signal
import sys
import time
from datetime import UTC, datetime

import gpt as g


def parse_list_int(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_time_dirs(value, nd):
    txt = (value or "").strip().lower()
    if txt in {"", "all", "*"}:
        return list(range(nd))
    out = []
    for x in value.split(","):
        x = x.strip()
        if not x:
            continue
        mu = int(x)
        if 0 <= mu < nd:
            out.append(mu)
    out = sorted(set(out))
    if not out:
        return list(range(nd))
    return out


def cli_arg_value(name):
    argv = sys.argv[1:]
    prefix = f"{name}="
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def unique_shifts(shifts):
    out = []
    seen = set()
    for s in shifts:
        key = tuple(s)
        if key not in seen:
            out.append(s)
            seen.add(key)
    return out


def shifted(lattice, shift_vec):
    out = lattice
    for mu, s in enumerate(shift_vec):
        if s != 0:
            out = g.cshift(out, mu, s)
    return out


def scalar_complex(x):
    if hasattr(x, "real"):
        return float(x.real), float(x.imag)
    return float(x), 0.0


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def estimate_tau_int(series, max_lag=200, c_window=4.0):
    vals = [float(x) for x in series if x is not None]
    n = len(vals)
    if n < 8:
        return 0.5
    mu = mean(vals)
    centered = [x - mu for x in vals]
    var = mean([x * x for x in centered])
    if var <= 0.0:
        return 0.5

    tau = 0.5
    max_lag = min(int(max_lag), n // 2)
    for lag in range(1, max_lag + 1):
        c = mean([centered[i] * centered[i + lag] for i in range(n - lag)])
        rho = c / var
        tau += rho
        # Madras-Sokal self-consistent window.
        if lag >= c_window * max(0.5, tau):
            break
    return max(0.5, float(tau))


def auto_bin_size(tau_int):
    return max(1, int(math.ceil(2.0 * max(0.5, float(tau_int)))))


def mean_complex(vals):
    if not vals:
        return 0.0, 0.0
    re = sum(v[0] for v in vals) / len(vals)
    im = sum(v[1] for v in vals) / len(vals)
    return re, im


def loop_orientations(nd, time_dirs, avg_space_dirs):
    out = []
    for tdir in time_dirs:
        space_dirs = [mu for mu in range(nd) if mu != tdir]
        if avg_space_dirs:
            for sdir in space_dirs:
                out.append((tdir, sdir))
        else:
            out.append((tdir, space_dirs[0]))
    return out


def make_shift(nd, axis, val):
    v = [0] * nd
    v[axis] = val
    return v


class SiteSampler:
    def __init__(self, grid, rng, sample_sites):
        self.rng = rng
        self.gsites = int(grid.gsites)
        self.sample_sites = max(0, int(sample_sites))
        self.enabled = 0 < self.sample_sites < self.gsites
        if self.enabled:
            self.p = self.sample_sites / float(self.gsites)
            self.selector = g.complex(grid)
            self.one = g.complex(grid)
            self.one[:] = 1.0
            self.zero = g.complex(grid)
            self.zero[:] = 0.0
        else:
            self.p = 1.0

    def mean(self, field):
        if not self.enabled:
            return g.sum(field) / self.gsites

        # Bernoulli site mask with inclusion probability p gives an unbiased
        # estimator of the volume average: E[(1/(V p)) sum m_x f_x] = (1/V) sum f_x.
        for _ in range(4):
            self.rng.uniform_real(self.selector)
            chosen = self.selector < self.p
            nsel = int(round(g.norm2(g.where(chosen, self.one, self.zero))))
            if nsel > 0:
                masked = g.where(chosen, field, 0.0 * field)
                return g.sum(masked) / (self.gsites * self.p)
        return g.sum(field) / self.gsites


def write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_progress_payload(
    *,
    seed,
    out_dir,
    phase,
    ntherm,
    nmeas,
    therm_done,
    meas_done,
    sweeps_done,
    total_sweeps,
    elapsed_sec,
    last_plaquette=None,
    last_loop_re=None,
    last_flux0=None,
    done=False,
):
    eta_sec = None
    if sweeps_done > 0 and sweeps_done < total_sweeps:
        eta_sec = elapsed_sec * (total_sweeps - sweeps_done) / sweeps_done
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "out_dir": out_dir,
        "phase": phase,
        "done": done,
        "ntherm": ntherm,
        "nmeas": nmeas,
        "therm_done": therm_done,
        "meas_done": meas_done,
        "sweeps_done": sweeps_done,
        "total_sweeps": total_sweeps,
        "progress": 0.0 if total_sweeps == 0 else sweeps_done / total_sweeps,
        "elapsed_sec": elapsed_sec,
        "eta_sec": eta_sec,
        "last_plaquette": last_plaquette,
        "last_loop_re": last_loop_re,
        "last_flux0": last_flux0,
    }


def one_sweep(U_field, hb, action, mask, mask_rb, mu_dirs):
    for cb in [g.even, g.odd]:
        mask[:] = 0
        mask_rb.checkerboard(cb)
        g.set_checkerboard(mask, mask_rb)
        for mu in mu_dirs:
            hb(U_field[mu], action.staple(U_field, mu), mask)


def build_smear_ops(time_dirs, smear_steps, smear_rho, smear_spatial_only):
    if smear_steps <= 0:
        return {}
    if smear_spatial_only:
        return {
            tdir: g.qcd.gauge.smear.stout(rho=smear_rho, orthogonal_dimension=tdir)
            for tdir in time_dirs
        }
    return {None: g.qcd.gauge.smear.stout(rho=smear_rho)}


def iter_measurement_fields(U, time_dirs, smear_steps, smear_ops, smear_spatial_only):
    if smear_steps <= 0:
        for tdir in time_dirs:
            yield tdir, U
        return

    if smear_spatial_only:
        for tdir in time_dirs:
            U_sm = g.copy(U)
            sm = smear_ops[tdir]
            for _ in range(smear_steps):
                U_sm = sm(U_sm)
            yield tdir, U_sm
    else:
        U_sm = g.copy(U)
        sm = smear_ops[None]
        for _ in range(smear_steps):
            U_sm = sm(U_sm)
        for tdir in time_dirs:
            yield tdir, U_sm


def measure_loops_for_tdir(U_use, tdir, Rs, Ts, orientations_for_tdir, sampler, loops_acc):
    if not orientations_for_tdir:
        return

    loop_specs = []
    keys = []
    for sdir in orientations_for_tdir:
        for r in Rs:
            for t in Ts:
                loop_specs.append([(tdir, t, sdir, r)])
                keys.append(f"R{r}_T{t}")

    if not loop_specs:
        return

    values = g.qcd.gauge.rectangle(
        U_use,
        loop_specs,
        field=sampler.enabled,
        trace=True,
        real=False,
    )
    if isinstance(values, tuple):
        pass
    elif isinstance(values, list):
        values = tuple(values)
    else:
        try:
            values = tuple(values)
        except TypeError:
            values = (values,)

    for key, val in zip(keys, values):
        if sampler.enabled:
            val = sampler.mean(val)
        if key not in loops_acc:
            loops_acc[key] = []
        loops_acc[key].append(scalar_complex(val))


def measure_flux_profile_for_tdir(
    U_use,
    tdir,
    nd,
    flux_r,
    flux_t,
    flux_r_perp_max,
    orientations_for_tdir,
    sampler,
    profiles_acc,
):
    p_field = g.qcd.gauge.plaquette(U_use, field=True)
    avg_p = sampler.mean(p_field)

    x_mid = flux_r // 2
    t_mid = flux_t // 2

    for sdir in orientations_for_tdir:
        w_field = g.qcd.gauge.rectangle(
            U_use,
            [[(tdir, flux_t, sdir, flux_r)]],
            field=True,
            trace=True,
            real=False,
        )

        avg_w = sampler.mean(w_field)
        avg_w_re = float(avg_w.real)
        if abs(avg_w_re) < 1e-20:
            continue

        perp_axes = [mu for mu in range(nd) if mu not in (tdir, sdir)]
        
        # Structure: map shift_vector -> list of r_perp indices that use it
        shift_map = {}
        
        # r_perp = 0 (on axis)
        s0 = make_shift(nd, sdir, x_mid)
        s0[tdir] = t_mid # center in time as well
        s0_key = tuple(s0)
        if s0_key not in shift_map: shift_map[s0_key] = []
        shift_map[s0_key].append(0)

        for r_perp in range(1, flux_r_perp_max + 1):
             for axis in perp_axes:
                # Positive direction
                svp = make_shift(nd, sdir, x_mid)
                svp[axis] = +r_perp
                svp[tdir] = t_mid
                svp_key = tuple(svp)
                if svp_key not in shift_map: shift_map[svp_key] = []
                shift_map[svp_key].append(r_perp)
                
                # Negative direction
                svm = make_shift(nd, sdir, x_mid)
                svm[axis] = -r_perp
                svm[tdir] = t_mid
                svm_key = tuple(svm)
                if svm_key not in shift_map: shift_map[svm_key] = []
                shift_map[svm_key].append(r_perp)

        # Accumulators for this specific tube orientation
        # profile_vals[r_perp] = list of connected correlator values
        profile_vals = [[] for _ in range(flux_r_perp_max + 1)]

        for s_key, r_indices in shift_map.items():
            p_shift = shifted(p_field, s_key)
            wp = sampler.mean(w_field * p_shift)
            connected = (wp / avg_w) - avg_p
            val = float(connected.real)
            for r_idx in r_indices:
                profile_vals[r_idx].append(val)
        
        # Average over rotationally equivalent points for this tube
        averaged_profile = [mean(vals) for vals in profile_vals]
        profiles_acc.append(averaged_profile)


def mean_measurement_items(items):
    if len(items) == 1:
        return items[0]

    out = {
        "plaquette": mean([x["plaquette"] for x in items]),
        "loops": {},
        "flux_profile_r_perp": [],
    }

    keys = list(items[0]["loops"].keys())
    for k in keys:
        out["loops"][k] = {
            "re": mean([x["loops"][k]["re"] for x in items]),
            "im": mean([x["loops"][k]["im"] for x in items]),
        }

    m = len(items[0]["flux_profile_r_perp"])
    for j in range(m):
        out["flux_profile_r_perp"].append(mean([x["flux_profile_r_perp"][j] for x in items]))

    return out


def estimate_veff_from_means(measurements, Rs, Ts):
    mean_loops = {}
    for r in Rs:
        for t in Ts:
            key = f"R{r}_T{t}"
            vals_re = [m["loops"][key]["re"] for m in measurements]
            vals_im = [m["loops"][key]["im"] for m in measurements]
            mean_loops[key] = {
                "re": mean(vals_re),
                "im": mean(vals_im),
            }

    veff = {}
    for r in Rs:
        for t in Ts:
            if (t + 1) not in Ts:
                continue
            a = mean_loops[f"R{r}_T{t}"]["re"]
            b = mean_loops[f"R{r}_T{t+1}"]["re"]
            if a > 0.0 and b > 0.0:
                veff[f"R{r}_T{t}to{t+1}"] = -math.log(b / a)
    return mean_loops, veff


def binned_measurements(measurements, bin_size):
    if bin_size <= 1:
        return list(measurements)
    n = len(measurements)
    nb = n // bin_size
    out = []
    for b in range(nb):
        chunk = measurements[b * bin_size : (b + 1) * bin_size]
        out.append(mean_measurement_items(chunk))
    rem = n % bin_size
    if rem > 0 and rem >= max(2, bin_size // 2):
        out.append(mean_measurement_items(measurements[-rem:]))
    return out


def estimate_veff_with_errors(measurements, Rs, Ts, bin_size):
    mean_loops, veff = estimate_veff_from_means(measurements, Rs, Ts)
    veff_err = {}
    veff_nbins = {}

    binned = binned_measurements(measurements, bin_size)
    nbin = len(binned)
    if nbin < 2:
        return mean_loops, veff, veff_err, veff_nbins, nbin

    for r in Rs:
        for t in Ts:
            if (t + 1) not in Ts:
                continue
            k0 = f"R{r}_T{t}"
            k1 = f"R{r}_T{t+1}"
            a = [m["loops"][k0]["re"] for m in binned]
            b = [m["loops"][k1]["re"] for m in binned]
            a_mean = mean(a)
            b_mean = mean(b)
            if not (a_mean > 0.0 and b_mean > 0.0):
                continue
            jk_vals = []
            for leave in range(nbin):
                aa = mean([x for i, x in enumerate(a) if i != leave])
                bb = mean([x for i, x in enumerate(b) if i != leave])
                if not (aa > 0.0 and bb > 0.0):
                    jk_vals = []
                    break
                jk_vals.append(-math.log(bb / aa))
            if len(jk_vals) != nbin:
                continue
            vjk = mean(jk_vals)
            var = sum((x - vjk) * (x - vjk) for x in jk_vals)
            sem = math.sqrt((nbin - 1.0) / nbin * var)
            key = f"R{r}_T{t}to{t+1}"
            veff_err[key] = sem
            veff_nbins[key] = nbin

    return mean_loops, veff, veff_err, veff_nbins, nbin


def estimate_flux_profile_with_errors(measurements, flux_r_perp_max, bin_size):
    m = flux_r_perp_max + 1
    mean_flux = []
    for r_perp in range(m):
        vals = [x["flux_profile_r_perp"][r_perp] for x in measurements]
        mean_flux.append(mean(vals))

    binned = binned_measurements(measurements, bin_size)
    nbin = len(binned)
    if nbin < 2:
        return mean_flux, [0.0] * m, nbin

    err = [0.0] * m
    for r_perp in range(m):
        vals = [x["flux_profile_r_perp"][r_perp] for x in binned]
        total = sum(vals)
        jk_vals = [(total - vals[leave]) / (nbin - 1) for leave in range(nbin)]
        vjk = mean(jk_vals)
        var = sum((x - vjk) * (x - vjk) for x in jk_vals)
        err[r_perp] = math.sqrt((nbin - 1.0) / nbin * var)

    return mean_flux, err, nbin


def comparable(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, (list, tuple)):
        return tuple(comparable(x) for x in value)
    return value


def checkpoint_mismatches(ckpt_params, current_params, keys):
    out = []
    for k in keys:
        if k not in ckpt_params:
            continue
        if comparable(ckpt_params.get(k)) != comparable(current_params.get(k)):
            out.append((k, ckpt_params.get(k), current_params.get(k)))
    return out


def main():
    seed = g.default.get("--seed", "petrus-su2-signal")
    beta = g.default.get_float("--beta", 2.4)
    ntherm = g.default.get_int("--ntherm", 200)
    nmeas = g.default.get_int("--nmeas", 200)
    nskip = g.default.get_int("--nskip", 5)

    L = parse_list_int(g.default.get("--L", "16,16,16,16"))
    Rs = sorted(parse_list_int(g.default.get("--R", "2,3,4,6,8,12")))
    Ts = sorted(parse_list_int(g.default.get("--T", "2,3,4,5,6")))

    flux_r = g.default.get_int("--flux-r", 6)
    flux_t = g.default.get_int("--flux-t", 4)
    flux_r_perp_max = g.default.get_int("--flux-rperp-max", 6)

    # Improved defaults for signal quality.
    time_dirs = parse_time_dirs(g.default.get("--time-dirs", "all"), len(L))
    avg_space_dirs = g.default.get_int("--avg-space-dirs", 1) != 0
    smear_steps = g.default.get_int("--smear-steps", 12)
    smear_rho = g.default.get_float("--smear-rho", 0.10)
    smear_spatial_only = g.default.get_int("--smear-spatial-only", 1) != 0

    multilevel_blocks = max(1, g.default.get_int("--multilevel-blocks", 2))
    multilevel_sweeps = max(0, g.default.get_int("--multilevel-sweeps", 1))
    multihit_samples = max(1, g.default.get_int("--multihit-samples", 2))
    multihit_temporal_sweeps = max(0, g.default.get_int("--multihit-temporal-sweeps", 1))

    flux_vacuum_mode = (g.default.get("--flux-vacuum-mode", "tail_mean") or "none").strip().lower()
    if flux_vacuum_mode not in {"none", "tail_mean"}:
        flux_vacuum_mode = "tail_mean"
    flux_vacuum_tail = max(0, g.default.get_int("--flux-vacuum-tail", 2))

    a_fm = g.default.get_float("--a-fm", 0.0)
    sample_sites = g.default.get_int("--sample-sites", 0)
    save_cfg_every = g.default.get_int("--save-cfg-every", 1)
    checkpoint_every = g.default.get_int("--checkpoint-every", 20)
    progress_every = max(1, g.default.get_int("--progress-every", 20))
    resume = g.default.get_int("--resume", 1) != 0
    resume_force = g.default.get_int("--resume-force", 0) != 0
    max_lag = max(10, g.default.get_int("--autocorr-max-lag", 200))

    precision = g.default.get("--precision", "double")
    prec = g.single if precision == "single" else g.double

    out_dir = g.default.get("--out", "results/su2_signal_scan")
    cli_out_dir = cli_arg_value("--out")
    if cli_out_dir:
        out_dir = cli_out_dir
    out_dir = os.path.expanduser(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    progress_file = os.path.join(out_dir, f"progress_{seed}.json")
    live_file = os.path.join(out_dir, f"live_{seed}.json")
    checkpoint_file = os.path.join(out_dir, f"checkpoint_{seed}.json")
    checkpoint_cfg_file = os.path.join(out_dir, f"checkpoint_{seed}.cfg")
    cfg_dir = os.path.join(out_dir, f"configs_{seed}")
    if save_cfg_every > 0:
        os.makedirs(cfg_dir, exist_ok=True)

    grid = g.grid(L, prec)
    grid_eo = g.grid(L, prec, g.redblack)
    rng = g.random(seed)
    meas_rng = g.random(f"{seed}:meas")
    estimator_rng = g.random(f"{seed}:estimator")
    sampler = SiteSampler(grid, meas_rng, sample_sites)
    U = g.qcd.gauge.unit(grid)
    Nd = len(U)

    # Ensure time dirs are legal in case L/nd differs from expectation.
    time_dirs[:] = [mu for mu in time_dirs if 0 <= mu < Nd]
    if not time_dirs:
        time_dirs[:] = [Nd - 1]

    orientations = loop_orientations(Nd, time_dirs, avg_space_dirs)
    all_mu_dirs = list(range(Nd))

    mask_rb = g.complex(grid_eo)
    mask_rb[:] = 1
    mask = g.complex(grid)

    action = g.qcd.gauge.action.wilson(beta)
    hb_chain = g.algorithms.markov.su2_heat_bath(rng)
    hb_estimator = g.algorithms.markov.su2_heat_bath(estimator_rng)

    smear_ops = build_smear_ops(time_dirs, smear_steps, smear_rho, smear_spatial_only)

    g.default.push_verbose("su2_heat_bath", False)
    g.message(f"Lattice={L}, precision={precision}, beta={beta}")
    if a_fm > 0.0:
        g.message(f"Lattice spacing a={a_fm:.6f} fm (a^-1={(0.1973269804 / a_fm):.3f} GeV)")
    g.message(f"Therm={ntherm}, meas={nmeas}, skip={nskip}")
    g.message(f"R={Rs}, T={Ts}, flux(R,T)=({flux_r},{flux_t}), r_perp_max={flux_r_perp_max}")
    if sampler.enabled:
        g.message(
            f"Measurement estimator=random-site sampling ({sampler.sample_sites}/{sampler.gsites} sites per observable)"
        )
    else:
        g.message("Measurement estimator=full-volume averaging")
    g.message(
        "Loop estimator improvements: "
        f"time_dirs={time_dirs}, avg_space_dirs={int(avg_space_dirs)}, "
        f"smear_steps={smear_steps}, smear_rho={smear_rho}, smear_spatial_only={int(smear_spatial_only)}"
    )
    g.message(
        "Variance reduction: "
        f"multilevel_blocks={multilevel_blocks}, multilevel_sweeps={multilevel_sweeps}, "
        f"multihit_samples={multihit_samples}, multihit_temporal_sweeps={multihit_temporal_sweeps}"
    )
    g.message(
        f"Flux vacuum subtraction: mode={flux_vacuum_mode}, tail={flux_vacuum_tail}"
    )
    g.message(f"Progress file: {progress_file}")
    g.message(f"Live file: {live_file}")
    g.message(
        f"Checkpoint every={checkpoint_every}, save_cfg_every={save_cfg_every}, "
        f"progress_every={progress_every}, resume={int(resume)}"
    )
    g.message(f"Autocorr analysis: max_lag={max_lag}")

    run_start = time.time()
    total_sweeps = ntherm + nmeas * nskip
    sweeps_done = 0
    last_plaquette = None
    last_loop_re = None
    last_flux0 = None
    therm_start = 0
    meas_start = 0
    measurements = []
    stop_requested = False

    def handle_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        g.message(f"Received signal {signum}; will checkpoint and stop.")

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    live_file_jsonl = os.path.join(out_dir, f"live_{seed}.jsonl")

    def write_live(new_item=None):
        # Update the metadata/status file (lightweight)
        # We no longer store the full 'measurements' list here to avoid O(N^2) I/O.
        meta_payload = {
            "meta": {
                "seed": seed,
                "beta": beta,
                "L": L,
                "R": Rs,
                "T": Ts,
                "a_fm": a_fm if a_fm > 0.0 else None,
                "flux_r": flux_r,
                "flux_t": flux_t,
                "flux_r_perp_max": flux_r_perp_max,
                "sample_sites": sample_sites,
                "sampling_mode": "random" if sampler.enabled else "full",
                "time_dirs": time_dirs,
                "avg_space_dirs": int(avg_space_dirs),
                "smear_steps": smear_steps,
                "smear_rho": smear_rho,
                "smear_spatial_only": int(smear_spatial_only),
                "multilevel_blocks": multilevel_blocks,
                "multilevel_sweeps": multilevel_sweeps,
                "multihit_samples": multihit_samples,
                "multihit_temporal_sweeps": multihit_temporal_sweeps,
                "flux_vacuum_mode": flux_vacuum_mode,
                "flux_vacuum_tail": flux_vacuum_tail,
                "jsonl_path": os.path.basename(live_file_jsonl),
            },
            # "measurements": [], # Removed to save space
        }
        write_json(live_file, meta_payload)

        # Append new measurement to JSONL
        if new_item is not None:
            with open(live_file_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(new_item) + "\n")

    def save_checkpoint(phase, therm_done, meas_done):
        if checkpoint_every <= 0:
            return
        g.save(checkpoint_cfg_file, U)
        write_json(
            checkpoint_file,
            {
                "seed": seed,
                "phase": phase,
                "therm_done": therm_done,
                "meas_done": meas_done,
                "sweeps_done": sweeps_done,
                "last_plaquette": last_plaquette,
                "last_loop_re": last_loop_re,
                "last_flux0": last_flux0,
                "measurements": measurements,
                "params": {
                    "beta": beta,
                    "L": L,
                    "ntherm": ntherm,
                    "nmeas": nmeas,
                    "nskip": nskip,
                    "R": Rs,
                    "T": Ts,
                    "a_fm": a_fm if a_fm > 0.0 else None,
                    "flux_r": flux_r,
                    "flux_t": flux_t,
                    "flux_r_perp_max": flux_r_perp_max,
                    "sample_sites": sample_sites,
                    "sampling_mode": "random" if sampler.enabled else "full",
                    "precision": precision,
                    "time_dirs": time_dirs,
                    "avg_space_dirs": int(avg_space_dirs),
                    "smear_steps": smear_steps,
                    "smear_rho": smear_rho,
                    "smear_spatial_only": int(smear_spatial_only),
                    "multilevel_blocks": multilevel_blocks,
                    "multilevel_sweeps": multilevel_sweeps,
                    "multihit_samples": multihit_samples,
                    "multihit_temporal_sweeps": multihit_temporal_sweeps,
                    "flux_vacuum_mode": flux_vacuum_mode,
                    "flux_vacuum_tail": flux_vacuum_tail,
                },
            },
        )

    def single_measurement(U_field):
        if sampler.enabled:
            plaq_re, _ = scalar_complex(sampler.mean(g.qcd.gauge.plaquette(U_field, field=True)))
            plaq = plaq_re
        else:
            plaq = float(g.qcd.gauge.plaquette(U_field))
    
        loops_acc = {}  # Key: "R{r}_T{t}", Value: list of (re, im) tuples
        flux_profiles_acc = []  # List of [val_at_r0, val_at_r1, ...]
    
        # Iterate one time-direction at a time to save memory
        for tdir, U_use in iter_measurement_fields(
            U_field, time_dirs, smear_steps, smear_ops, smear_spatial_only
        ):
            # Filter orientations for this tdir
            orientations_for_tdir = [sdir for (td, sdir) in orientations if td == tdir]
            if not orientations_for_tdir:
                continue
    
            measure_loops_for_tdir(
                U_use, tdir, Rs, Ts, orientations_for_tdir, sampler, loops_acc
            )
            measure_flux_profile_for_tdir(
                U_use,
                tdir,
                Nd,
                flux_r,
                flux_t,
                flux_r_perp_max,
                orientations_for_tdir,
                sampler,
                flux_profiles_acc,
            )
    
        # Aggregate loops
        loops = {}
        for k, v in loops_acc.items():
            re, im = mean_complex(v)
            loops[k] = {"re": re, "im": im}
    
        # Aggregate flux profiles
        if not flux_profiles_acc:
            final_flux_profile = [0.0] * (flux_r_perp_max + 1)
        else:
            final_flux_profile = []
            for j in range(flux_r_perp_max + 1):
                final_flux_profile.append(mean([p[j] for p in flux_profiles_acc]))
    
        # Apply vacuum subtraction if needed (globally averaged profile)
        if (
            flux_vacuum_mode == "tail_mean"
            and flux_vacuum_tail > 0
            and len(final_flux_profile) >= flux_vacuum_tail
        ):
            tail = final_flux_profile[-flux_vacuum_tail:]
            vac = mean(tail)
            final_flux_profile = [x - vac for x in final_flux_profile]
    
        return {
            "plaquette": plaq,
            "loops": loops,
            "flux_profile_r_perp": final_flux_profile,
        }
    def measure_with_variance_reduction(U_field):
        if multilevel_blocks == 1 and multihit_samples == 1:
            return single_measurement(U_field)

        blocks = []
        U_block = g.copy(U_field)
        for ib in range(multilevel_blocks):
            if ib > 0 and multilevel_sweeps > 0:
                for _ in range(multilevel_sweeps):
                    one_sweep(U_block, hb_estimator, action, mask, mask_rb, all_mu_dirs)

            hits = []
            U_hit = g.copy(U_block)
            for ih in range(multihit_samples):
                if ih > 0 and multihit_temporal_sweeps > 0:
                    for _ in range(multihit_temporal_sweeps):
                        one_sweep(U_hit, hb_estimator, action, mask, mask_rb, time_dirs)
                hits.append(single_measurement(U_hit))

            blocks.append(mean_measurement_items(hits))

        return mean_measurement_items(blocks)

    if resume and os.path.exists(checkpoint_file) and os.path.exists(checkpoint_cfg_file):
        ckpt = read_json(checkpoint_file)
        if ckpt and ckpt.get("seed") == seed:
            current_params = {
                "beta": beta,
                "L": L,
                "ntherm": ntherm,
                "nmeas": nmeas,
                "nskip": nskip,
                "R": Rs,
                "T": Ts,
                "precision": precision,
                "flux_r": flux_r,
                "flux_t": flux_t,
                "flux_r_perp_max": flux_r_perp_max,
                "sample_sites": sample_sites,
                "time_dirs": time_dirs,
                "avg_space_dirs": int(avg_space_dirs),
                "smear_steps": smear_steps,
                "smear_rho": smear_rho,
                "smear_spatial_only": int(smear_spatial_only),
                "multilevel_blocks": multilevel_blocks,
                "multilevel_sweeps": multilevel_sweeps,
                "multihit_samples": multihit_samples,
                "multihit_temporal_sweeps": multihit_temporal_sweeps,
                "flux_vacuum_mode": flux_vacuum_mode,
                "flux_vacuum_tail": flux_vacuum_tail,
            }
            mm = checkpoint_mismatches(
                ckpt.get("params", {}) or {},
                current_params,
                keys=list(current_params.keys()),
            )
            if mm and not resume_force:
                details = "; ".join(f"{k}: ckpt={a} current={b}" for k, a, b in mm[:6])
                raise SystemExit(
                    "Checkpoint parameter mismatch. Refusing resume. "
                    f"{details}. Pass --resume-force 1 to override."
                )
            if mm and resume_force:
                g.message(f"Resume forced despite {len(mm)} parameter mismatches.")

            g.message(f"Resuming from checkpoint {checkpoint_file}")
            loaded = g.load(checkpoint_cfg_file)
            for mu in range(min(len(U), len(loaded))):
                g.copy(U[mu], loaded[mu])
            therm_start = int(ckpt.get("therm_done", 0))
            meas_start = int(ckpt.get("meas_done", 0))
            sweeps_done = int(ckpt.get("sweeps_done", therm_start + meas_start * nskip))
            measurements = list(ckpt.get("measurements", []))
            if len(measurements) < meas_start:
                meas_start = len(measurements)
            last_plaquette = ckpt.get("last_plaquette")
            last_loop_re = ckpt.get("last_loop_re")
            last_flux0 = ckpt.get("last_flux0")
            g.message(
                f"Resume state: therm={therm_start}/{ntherm}, meas={meas_start}/{nmeas}, sweeps={sweeps_done}/{total_sweeps}"
            )
            
            # Rewrite JSONL to ensure consistency with checkpoint
            if measurements:
                try:
                    with open(live_file_jsonl, "w", encoding="utf-8") as f:
                        for m in measurements:
                            f.write(json.dumps(m) + "\n")
                except Exception as e:
                    g.message(f"Warning: failed to rewrite JSONL on resume: {e}")

        else:
            g.message("Checkpoint found but seed mismatch; starting fresh.")

    write_live()

    write_json(
        progress_file,
        make_progress_payload(
            seed=seed,
            out_dir=out_dir,
            phase="initializing",
            ntherm=ntherm,
            nmeas=nmeas,
            therm_done=therm_start,
            meas_done=meas_start,
            sweeps_done=sweeps_done,
            total_sweeps=total_sweeps,
            elapsed_sec=0.0,
            last_plaquette=last_plaquette,
            last_loop_re=last_loop_re,
            last_flux0=last_flux0,
            done=False,
        ),
    )

    therm_log_every = max(1, ntherm // 10)
    therm_done = therm_start
    meas_done = meas_start
    for i in range(therm_start, ntherm):
        one_sweep(U, hb_chain, action, mask, mask_rb, all_mu_dirs)
        sweeps_done += 1
        therm_done = i + 1
        if (i + 1) % therm_log_every == 0:
            last_plaquette = float(g.qcd.gauge.plaquette(U))
            g.message(f"Thermalization {i+1}/{ntherm}, P={last_plaquette}")

        if ((i + 1) % progress_every) == 0 or (i + 1) == ntherm or stop_requested:
            write_json(
                progress_file,
                make_progress_payload(
                    seed=seed,
                    out_dir=out_dir,
                    phase="thermalization",
                    ntherm=ntherm,
                    nmeas=nmeas,
                    therm_done=i + 1,
                    meas_done=meas_start,
                    sweeps_done=sweeps_done,
                    total_sweeps=total_sweeps,
                    elapsed_sec=time.time() - run_start,
                    last_plaquette=last_plaquette,
                    last_loop_re=last_loop_re,
                    last_flux0=last_flux0,
                    done=False,
                ),
            )
        if checkpoint_every > 0 and (((i + 1) % checkpoint_every) == 0 or (i + 1) == ntherm):
            save_checkpoint("thermalization", i + 1, meas_start)
        if stop_requested:
            break

    for i in range(meas_start, nmeas):
        if stop_requested:
            break
        for _ in range(nskip):
            one_sweep(U, hb_chain, action, mask, mask_rb, all_mu_dirs)
            sweeps_done += 1

        measured = measure_with_variance_reduction(U)

        item = {
            "idx": i,
            "plaquette": measured["plaquette"],
            "loops": measured["loops"],
            "flux_profile_r_perp": measured["flux_profile_r_perp"],
        }
        measurements.append(item)
        meas_done = i + 1

        last_plaquette = item["plaquette"]
        last_loop_re = item["loops"].get(f"R{flux_r}_T{flux_t}", {"re": 0.0})["re"]
        last_flux0 = item["flux_profile_r_perp"][0]

        g.message(
            f"Meas {i+1}/{nmeas}: P={item['plaquette']:.8f}, "
            f"W(R={flux_r},T={flux_t})={last_loop_re:.6e}, "
            f"flux(r_perp=0)={last_flux0:.6e}"
        )

        write_live(new_item=item)
        write_json(
            progress_file,
            make_progress_payload(
                seed=seed,
                out_dir=out_dir,
                phase="production",
                ntherm=ntherm,
                nmeas=nmeas,
                therm_done=ntherm,
                meas_done=i + 1,
                sweeps_done=sweeps_done,
                total_sweeps=total_sweeps,
                elapsed_sec=time.time() - run_start,
                last_plaquette=last_plaquette,
                last_loop_re=last_loop_re,
                last_flux0=last_flux0,
                done=False,
            ),
        )

        if save_cfg_every > 0 and ((i + 1) % save_cfg_every) == 0:
            cfg_file = os.path.join(cfg_dir, f"cfg_{seed}_{i+1:05d}.cfg")
            g.save(cfg_file, U)
        if checkpoint_every > 0 and (((i + 1) % checkpoint_every) == 0 or (i + 1) == nmeas):
            save_checkpoint("production", ntherm, i + 1)

    if stop_requested:
        write_live()
        write_json(
            progress_file,
            make_progress_payload(
                seed=seed,
                out_dir=out_dir,
                phase="interrupted",
                ntherm=ntherm,
                nmeas=nmeas,
                therm_done=therm_done,
                meas_done=meas_done,
                sweeps_done=sweeps_done,
                total_sweeps=total_sweeps,
                elapsed_sec=time.time() - run_start,
                last_plaquette=last_plaquette,
                last_loop_re=last_loop_re,
                last_flux0=last_flux0,
                done=False,
            ),
        )
        save_checkpoint("interrupted", therm_done, meas_done)
        g.message("Interrupted; checkpoint saved.")
        return

    plaq_series = [m["plaquette"] for m in measurements]
    tau_plaq = estimate_tau_int(plaq_series, max_lag=max_lag) if plaq_series else 0.5
    loop_key = f"R{flux_r}_T{flux_t}"
    loop_series = [m["loops"].get(loop_key, {"re": 0.0})["re"] for m in measurements]
    tau_loop = estimate_tau_int(loop_series, max_lag=max_lag) if loop_series else 0.5
    bin_size = auto_bin_size(max(tau_plaq, tau_loop))
    g.message(
        f"Autocorr: tau_int(plaq)={tau_plaq:.3f}, tau_int({loop_key})={tau_loop:.3f}, bin_size={bin_size}"
    )

    mean_loops, veff, veff_err, veff_nbins, n_binned = estimate_veff_with_errors(
        measurements, Rs, Ts, bin_size
    )

    mean_flux, mean_flux_err, flux_n_binned = estimate_flux_profile_with_errors(
        measurements, flux_r_perp_max, bin_size
    )

    output = {
        "meta": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "seed": seed,
            "beta": beta,
            "L": L,
            "precision": precision,
            "ntherm": ntherm,
            "nmeas": nmeas,
            "nskip": nskip,
            "R": Rs,
            "T": Ts,
            "a_fm": a_fm if a_fm > 0.0 else None,
            "flux_r": flux_r,
            "flux_t": flux_t,
            "flux_r_perp_max": flux_r_perp_max,
            "sample_sites": sample_sites,
            "sampling_mode": "random" if sampler.enabled else "full",
            "time_dirs": time_dirs,
            "avg_space_dirs": int(avg_space_dirs),
            "smear_steps": smear_steps,
            "smear_rho": smear_rho,
            "smear_spatial_only": int(smear_spatial_only),
            "multilevel_blocks": multilevel_blocks,
            "multilevel_sweeps": multilevel_sweeps,
            "multihit_samples": multihit_samples,
            "multihit_temporal_sweeps": multihit_temporal_sweeps,
            "flux_vacuum_mode": flux_vacuum_mode,
            "flux_vacuum_tail": flux_vacuum_tail,
            "autocorr_max_lag": max_lag,
            "tau_int_plaquette": tau_plaq,
            "tau_int_loop_re": tau_loop,
            "autocorr_bin_size": bin_size,
            "veff_n_binned": n_binned,
            "flux_n_binned": flux_n_binned,
        },
        "mean_plaquette": mean([m["plaquette"] for m in measurements]),
        "mean_loops": mean_loops,
        "veff": veff,
        "veff_err": veff_err,
        "veff_nbins": veff_nbins,
        "mean_flux_profile_r_perp": mean_flux,
        "mean_flux_profile_r_perp_err": mean_flux_err,
        "measurements": measurements,
    }

    out_file = os.path.join(out_dir, f"su2_2q_signal_{seed}.json")
    write_json(out_file, output)

    write_json(
        progress_file,
        make_progress_payload(
            seed=seed,
            out_dir=out_dir,
            phase="complete",
            ntherm=ntherm,
            nmeas=nmeas,
            therm_done=ntherm,
            meas_done=nmeas,
            sweeps_done=sweeps_done,
            total_sweeps=total_sweeps,
            elapsed_sec=time.time() - run_start,
            last_plaquette=last_plaquette,
            last_loop_re=last_loop_re,
            last_flux0=last_flux0,
            done=True,
        ),
    )
    save_checkpoint("complete", ntherm, nmeas)

    g.message(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
