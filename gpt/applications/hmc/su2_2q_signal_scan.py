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
import gc
import json
import math
import os
import signal
import sys
import time
from datetime import timezone, datetime
from pathlib import Path

import gpt as g

RUN_COMPUTE_META = {}


def clear_gpt_caches():
    """Clear GPT's internal stencil/transport caches to free C++ memory."""
    from gpt.qcd.gauge.loops import default_rectangle_cache
    from gpt.qcd.gauge.stencil.plaquette import default_plaquette_cache
    from gpt.qcd.gauge.stencil.staple import default_staple_cache
    from gpt.core.foundation.lattice.matrix.exp import default_exp_cache

    default_rectangle_cache.clear()
    default_plaquette_cache.clear()
    default_staple_cache.clear()
    default_exp_cache.clear()


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


def detect_grid_build_info():
    candidates = []
    try:
        here = Path(__file__).resolve()
        repo_root = here.parents[3]
        candidates.append(repo_root / "Grid" / "build" / "grid.configure.summary")
    except Exception:
        pass
    env_summary = os.environ.get("GRID_CONFIG_SUMMARY", "").strip()
    if env_summary:
        candidates.append(Path(env_summary))

    for path in candidates:
        try:
            if not path.exists():
                continue
            out = {
                "summary_path": str(path),
                "acceleration": None,
                "simd": None,
                "threading": None,
            }
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.rstrip("\n")
                    if ":" not in line:
                        continue
                    k, v = line.split(":", 1)
                    key = k.strip().lower()
                    val = v.strip()
                    if key == "acceleration":
                        out["acceleration"] = val
                    elif key == "simd":
                        out["simd"] = val
                    elif key == "threading":
                        out["threading"] = val
            return out
        except Exception:
            continue

    return {
        "summary_path": "",
        "acceleration": None,
        "simd": None,
        "threading": None,
    }


def detect_runtime_backend():
    mem = {}
    try:
        mem = g.mem_info() or {}
    except Exception:
        mem = {}

    build = detect_grid_build_info()
    accel_total = float(mem.get("accelerator_total") or 0.0)
    accel_available = float(mem.get("accelerator_available") or 0.0)
    backend = "gpu" if accel_total > 0 else "cpu"
    acceleration = str(build.get("acceleration") or "").strip()
    simd = str(build.get("simd") or "").strip()
    threading = str(build.get("threading") or "").strip()

    return {
        "backend": backend,
        "accelerator_total_bytes": int(max(0.0, accel_total)),
        "accelerator_available_bytes": int(max(0.0, accel_available)),
        "grid_acceleration": acceleration or None,
        "grid_simd": simd or None,
        "grid_threading": threading or None,
        "grid_summary_path": str(build.get("summary_path") or ""),
    }


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
    therm_sweep_substep_done=None,
    therm_sweep_substep_total=None,
    meas_cfg_index=None,
    meas_cfg_total=None,
    meas_cfg_stage=None,
    meas_cfg_substep_done=None,
    meas_cfg_substep_total=None,
    meas_cursor_kind=None,
    meas_cursor_tdir=None,
    meas_cursor_sdir=None,
    meas_cursor_r=None,
    meas_cursor_t=None,
    meas_cursor_r_perp=None,
    meas_cursor_shift=None,
    last_plaquette=None,
    last_loop_re=None,
    last_flux0=None,
    done=False,
):
    eta_sec = None
    if sweeps_done > 0 and sweeps_done < total_sweeps:
        eta_sec = elapsed_sec * (total_sweeps - sweeps_done) / sweeps_done
    sub_done = None
    sub_total = None
    sub_progress = None
    if isinstance(therm_sweep_substep_done, (int, float)) and isinstance(therm_sweep_substep_total, (int, float)):
        td = int(max(0, therm_sweep_substep_done))
        tt = int(max(0, therm_sweep_substep_total))
        if tt > 0:
            sub_done = min(td, tt)
            sub_total = tt
            sub_progress = sub_done / sub_total

    meas_sub_done = None
    meas_sub_total = None
    meas_sub_progress = None
    if isinstance(meas_cfg_substep_done, (int, float)) and isinstance(meas_cfg_substep_total, (int, float)):
        md = int(max(0, meas_cfg_substep_done))
        mt = int(max(0, meas_cfg_substep_total))
        if mt > 0:
            meas_sub_done = min(md, mt)
            meas_sub_total = mt
            meas_sub_progress = meas_sub_done / meas_sub_total

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
        "therm_sweep_substep_done": sub_done,
        "therm_sweep_substep_total": sub_total,
        "therm_sweep_substep_progress": sub_progress,
        "meas_cfg_index": meas_cfg_index,
        "meas_cfg_total": meas_cfg_total,
        "meas_cfg_stage": meas_cfg_stage,
        "meas_cfg_substep_done": meas_sub_done,
        "meas_cfg_substep_total": meas_sub_total,
        "meas_cfg_substep_progress": meas_sub_progress,
        "meas_cursor_kind": meas_cursor_kind,
        "meas_cursor_tdir": meas_cursor_tdir,
        "meas_cursor_sdir": meas_cursor_sdir,
        "meas_cursor_r": meas_cursor_r,
        "meas_cursor_t": meas_cursor_t,
        "meas_cursor_r_perp": meas_cursor_r_perp,
        "meas_cursor_shift": meas_cursor_shift,
        "progress": 0.0 if total_sweeps == 0 else sweeps_done / total_sweeps,
        "elapsed_sec": elapsed_sec,
        "eta_sec": eta_sec,
        "last_plaquette": last_plaquette,
        "last_loop_re": last_loop_re,
        "last_flux0": last_flux0,
    }
    if RUN_COMPUTE_META:
        payload.update(RUN_COMPUTE_META)
    return payload


def one_sweep(U_field, hb, action, mask, mask_rb, mu_dirs, step_cb=None):
    step_idx = 0
    step_total = max(1, 2 * len(mu_dirs))
    for cb in [g.even, g.odd]:
        mask[:] = 0
        mask_rb.checkerboard(cb)
        g.set_checkerboard(mask, mask_rb)
        for mu in mu_dirs:
            hb(U_field[mu], action.staple(U_field, mu), mask)
            step_idx += 1
            if step_cb is not None:
                step_cb(step_idx, step_total)


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


def wilson_loop_field(U, mu, L_mu, nu, L_nu):
    """Return Tr[W(R,T)] as a lattice complex field (one value per site)."""
    nd = len(U)
    W = g.copy(U[mu])
    for i in range(1, L_mu):
        W = g(W * g.cshift(U[mu], mu, i))
    for j in range(L_nu):
        tmp = U[nu]
        for d in range(nd):
            s = (L_mu if d == mu else 0) + (j if d == nu else 0)
            if s != 0:
                tmp = g.cshift(tmp, d, s)
        W = g(W * tmp)
    for i in range(L_mu - 1, -1, -1):
        tmp = g.adj(U[mu])
        for d in range(nd):
            s = (i if d == mu else 0) + (L_nu if d == nu else 0)
            if s != 0:
                tmp = g.cshift(tmp, d, s)
        W = g(W * tmp)
    for j in range(L_nu - 1, 0, -1):
        W = g(W * g.cshift(g.adj(U[nu]), nu, j))
    W = g(W * g.adj(U[nu]))
    ndim = U[0].otype.shape[0]
    return g(g.trace(W) / ndim)


def wilson_loop_trace(U, mu, L_mu, nu, L_nu):
    """Compute Tr[W(R,T)] averaged over the lattice using only g.cshift.

    Builds the closed rectangular path step-by-step so that at most two
    full-lattice matrix fields are alive at any time (~40 MB for SU(2)
    double on 24^4), avoiding the ~1 GB+ stencil allocations of
    g.qcd.gauge.rectangle / parallel_transport_matrix.
    """
    nd = len(U)
    # Forward mu: L_mu steps
    W = g.copy(U[mu])
    for i in range(1, L_mu):
        W = g(W * g.cshift(U[mu], mu, i))
    # Forward nu: L_nu steps
    for j in range(L_nu):
        shift = [0] * nd
        shift[mu] = L_mu
        shift[nu] = j
        u_shifted = W  # will be overwritten
        # Multi-dim shift via chained cshift
        tmp = U[nu]
        for d in range(nd):
            if shift[d] != 0:
                tmp = g.cshift(tmp, d, shift[d])
        W = g(W * tmp)
    # Backward mu: L_mu steps
    for i in range(L_mu - 1, -1, -1):
        shift = [0] * nd
        shift[mu] = i
        shift[nu] = L_nu
        tmp = g.adj(U[mu])
        for d in range(nd):
            if shift[d] != 0:
                tmp = g.cshift(tmp, d, shift[d])
        W = g(W * tmp)
    # Backward nu: L_nu steps
    for j in range(L_nu - 1, 0, -1):
        tmp = g.cshift(g.adj(U[nu]), nu, j)
        W = g(W * tmp)
    W = g(W * g.adj(U[nu]))
    # Trace and volume average
    ndim = U[0].otype.shape[0]
    tr = g.sum(g.trace(W))
    return tr / W.grid.gsites / ndim


def polyakov_loop_trace(U, mu, L_mu, sampler=None):
    """Compute average Polyakov loop Tr[prod U_mu] / N along direction mu."""
    if L_mu <= 0:
        return 0.0 + 0.0j
    P = g.copy(U[mu])
    for i in range(1, L_mu):
        P = g(P * g.cshift(U[mu], mu, i))
    ndim = U[0].otype.shape[0]
    tr_field = g(g.trace(P) / ndim)
    if sampler is not None and getattr(sampler, "enabled", False):
        return sampler.mean(tr_field)
    return g.sum(tr_field) / tr_field.grid.gsites


def measure_polyakov_loops(U_field, dirs, extents, sampler=None):
    out = {}
    for mu in dirs:
        if mu < 0 or mu >= len(U_field) or mu >= len(extents):
            continue
        l_mu = int(extents[mu])
        if l_mu <= 0:
            continue
        tr = polyakov_loop_trace(U_field, mu, l_mu, sampler=sampler)
        re, im = scalar_complex(tr)
        out[f"mu{mu}"] = {
            "re": re,
            "im": im,
            "phase": float(math.atan2(im, re)),
        }
    return out


def measure_loops_for_tdir(U_use, tdir, Rs, Ts, orientations_for_tdir, sampler, loops_acc, progress_cb=None):
    if not orientations_for_tdir:
        return

    for r in Rs:
        for t in Ts:
            key = f"R{r}_T{t}"
            for sdir in orientations_for_tdir:
                tr = wilson_loop_trace(U_use, tdir, t, sdir, r)
                if key not in loops_acc:
                    loops_acc[key] = []
                loops_acc[key].append(scalar_complex(tr))
                if progress_cb is not None:
                    progress_cb(
                        "measure_loops",
                        1,
                        cursor={
                            "kind": "loop",
                            "tdir": int(tdir),
                            "sdir": int(sdir),
                            "r": int(r),
                            "t": int(t),
                        },
                    )


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
    progress_cb=None,
):
    p_field = g.qcd.gauge.plaquette(U_use, field=True)
    avg_p = sampler.mean(p_field)

    x_mid = flux_r // 2
    t_mid = flux_t // 2

    for sdir in orientations_for_tdir:
        w_field = wilson_loop_field(U_use, tdir, flux_t, sdir, flux_r)

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
            if progress_cb is not None:
                cursor_r_perp = int(r_indices[0]) if r_indices else 0
                progress_cb(
                    "measure_flux",
                    1,
                    cursor={
                        "kind": "flux",
                        "tdir": int(tdir),
                        "sdir": int(sdir),
                        "r": int(flux_r),
                        "t": int(flux_t),
                        "r_perp": cursor_r_perp,
                        "shift": [int(x) for x in s_key],
                    },
                )
        
        # Average over rotationally equivalent points for this tube
        averaged_profile = [mean(vals) for vals in profile_vals]
        profiles_acc.append(averaged_profile)


def single_measurement_step_count(time_dirs, orientations, nd, Rs, Ts, flux_r_perp_max, polyakov_dirs_count=0):
    """Estimated fine-grained steps for one single_measurement() call."""
    if not time_dirs:
        return 1
    orientations_by_tdir = {tdir: 0 for tdir in time_dirs}
    for td, _ in orientations:
        if td in orientations_by_tdir:
            orientations_by_tdir[td] += 1

    loop_steps = 0
    flux_steps = 0
    flux_shift_count_per_orientation = 1 + 2 * max(0, nd - 2) * max(0, flux_r_perp_max)
    for tdir in time_dirs:
        n_or = orientations_by_tdir.get(tdir, 0)
        if n_or <= 0:
            continue
        loop_steps += n_or * len(Rs) * len(Ts)
        flux_steps += n_or * flux_shift_count_per_orientation
    poly_steps = max(0, int(polyakov_dirs_count))
    return max(1, loop_steps + flux_steps + poly_steps)


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

    poly_keys = sorted(
        {
            k
            for x in items
            for k in ((x.get("polyakov_loops") or {}).keys())
        }
    )
    if poly_keys:
        out["polyakov_loops"] = {}
        for k in poly_keys:
            vals = [x["polyakov_loops"][k] for x in items if isinstance(x.get("polyakov_loops", {}).get(k), dict)]
            if not vals:
                continue
            re = mean([float(v.get("re", 0.0)) for v in vals])
            im = mean([float(v.get("im", 0.0)) for v in vals])
            out["polyakov_loops"][k] = {
                "re": re,
                "im": im,
                "phase": float(math.atan2(im, re)),
            }

    if "profiling" in items[0]:
        out["profiling"] = {
            "loop_time": mean([max(0.0, x.get("profiling", {}).get("loop_time", 0.0)) for x in items]),
            "flux_time": mean([max(0.0, x.get("profiling", {}).get("flux_time", 0.0)) for x in items]),
        }

    return out


def estimate_veff_from_means(measurements, Rs, Ts):
    mean_loops = {}
    for r in Rs:
        for t in Ts:
            key = f"R{r}_T{t}"
            vals_re = []
            vals_im = []
            for m in measurements:
                loops = m.get("loops") if isinstance(m, dict) else None
                entry = loops.get(key) if isinstance(loops, dict) else None
                if not isinstance(entry, dict):
                    continue
                re_v = entry.get("re")
                im_v = entry.get("im")
                if not isinstance(re_v, (int, float)) or not math.isfinite(re_v):
                    continue
                if not isinstance(im_v, (int, float)) or not math.isfinite(im_v):
                    im_v = 0.0
                vals_re.append(float(re_v))
                vals_im.append(float(im_v))
            if not vals_re:
                continue
            mean_loops[key] = {
                "re": mean(vals_re),
                "im": mean(vals_im),
            }

    veff = {}
    for r in Rs:
        for t in Ts:
            if (t + 1) not in Ts:
                continue
            k0 = f"R{r}_T{t}"
            k1 = f"R{r}_T{t+1}"
            if k0 not in mean_loops or k1 not in mean_loops:
                continue
            a = mean_loops[k0]["re"]
            b = mean_loops[k1]["re"]
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
            pairs = []
            for m in binned:
                loops = m.get("loops") if isinstance(m, dict) else None
                e0 = loops.get(k0) if isinstance(loops, dict) else None
                e1 = loops.get(k1) if isinstance(loops, dict) else None
                if not isinstance(e0, dict) or not isinstance(e1, dict):
                    continue
                a0 = e0.get("re")
                b0 = e1.get("re")
                if (
                    isinstance(a0, (int, float))
                    and isinstance(b0, (int, float))
                    and math.isfinite(a0)
                    and math.isfinite(b0)
                ):
                    pairs.append((float(a0), float(b0)))
            npair = len(pairs)
            if npair < 2:
                continue
            a = [x[0] for x in pairs]
            b = [x[1] for x in pairs]
            a_mean = mean(a)
            b_mean = mean(b)
            if not (a_mean > 0.0 and b_mean > 0.0):
                continue
            jk_vals = []
            for leave in range(npair):
                aa = mean([x for i, x in enumerate(a) if i != leave])
                bb = mean([x for i, x in enumerate(b) if i != leave])
                if not (aa > 0.0 and bb > 0.0):
                    jk_vals = []
                    break
                jk_vals.append(-math.log(bb / aa))
            if len(jk_vals) != npair:
                continue
            vjk = mean(jk_vals)
            var = sum((x - vjk) * (x - vjk) for x in jk_vals)
            sem = math.sqrt((npair - 1.0) / npair * var)
            key = f"R{r}_T{t}to{t+1}"
            veff_err[key] = sem
            veff_nbins[key] = npair

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
    Rs = sorted(parse_list_int(g.default.get("--R", "1,2,3,4,6,8,12")))
    Ts = sorted(parse_list_int(g.default.get("--T", "1,2,3,4,5,6")))

    flux_r = g.default.get_int("--flux-r", 6)
    flux_t = g.default.get_int("--flux-t", 4)
    flux_r_perp_max = g.default.get_int("--flux-rperp-max", 6)

    # Improved defaults for signal quality.
    time_dirs = parse_time_dirs(g.default.get("--time-dirs", "all"), len(L))
    avg_space_dirs = g.default.get_int("--avg-space-dirs", 1) != 0
    smear_steps = g.default.get_int("--smear-steps", 12)
    smear_rho = g.default.get_float("--smear-rho", 0.10)
    smear_spatial_only = g.default.get_int("--smear-spatial-only", 1) != 0

    multilevel_blocks = max(1, g.default.get_int("--multilevel-blocks", 8))
    multilevel_sweeps = max(0, g.default.get_int("--multilevel-sweeps", 4))
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
    progress_substep_min_interval = max(0.05, g.default.get_float("--progress-substep-min-interval-sec", 0.2))
    resume = g.default.get_int("--resume", 1) != 0
    resume_force = g.default.get_int("--resume-force", 0) != 0
    skip_flux = g.default.get_int("--skip_flux", 0) != 0
    max_lag = max(10, g.default.get_int("--autocorr-max-lag", 200))
    pipeline_label = (g.default.get("--pipeline-label", "auto") or "auto").strip().lower()
    require_accelerator = g.default.get_int("--require-accelerator", 0) != 0

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
    runtime_backend = detect_runtime_backend()
    runtime_backend["pipeline"] = pipeline_label
    RUN_COMPUTE_META.clear()
    RUN_COMPUTE_META.update(
        {
            "compute_backend": runtime_backend.get("backend"),
            "compute_pipeline": pipeline_label,
            "grid_acceleration": runtime_backend.get("grid_acceleration"),
            "grid_simd": runtime_backend.get("grid_simd"),
            "accelerator_total_bytes": runtime_backend.get("accelerator_total_bytes"),
            "accelerator_available_bytes": runtime_backend.get("accelerator_available_bytes"),
        }
    )
    if require_accelerator and runtime_backend.get("backend") != "gpu":
        raise SystemExit(
            "Requested --require-accelerator 1 but no accelerator memory is available. "
            f"grid_acceleration={runtime_backend.get('grid_acceleration')}, "
            f"accelerator_total_bytes={runtime_backend.get('accelerator_total_bytes')}"
        )

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
    g.message(
        "Compute: "
        f"backend={runtime_backend.get('backend')}, pipeline={pipeline_label}, "
        f"grid_acceleration={runtime_backend.get('grid_acceleration') or 'unknown'}, "
        f"SIMD={runtime_backend.get('grid_simd') or 'unknown'}, "
        f"accelerator_total_bytes={runtime_backend.get('accelerator_total_bytes', 0)}"
    )
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
        f"progress_every={progress_every}, progress_substep_min_interval={progress_substep_min_interval:.3f}s, "
        f"resume={int(resume)}"
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
                "polyakov_dirs": all_mu_dirs,
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
                "compute_backend": runtime_backend.get("backend"),
                "compute_pipeline": pipeline_label,
                "grid_acceleration": runtime_backend.get("grid_acceleration"),
                "grid_simd": runtime_backend.get("grid_simd"),
                "grid_threading": runtime_backend.get("grid_threading"),
                "grid_summary_path": runtime_backend.get("grid_summary_path"),
                "accelerator_total_bytes": runtime_backend.get("accelerator_total_bytes"),
                "accelerator_available_bytes": runtime_backend.get("accelerator_available_bytes"),
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
                "runtime": {
                    "compute_backend": runtime_backend.get("backend"),
                    "compute_pipeline": pipeline_label,
                    "grid_acceleration": runtime_backend.get("grid_acceleration"),
                    "grid_simd": runtime_backend.get("grid_simd"),
                    "grid_threading": runtime_backend.get("grid_threading"),
                    "grid_summary_path": runtime_backend.get("grid_summary_path"),
                    "accelerator_total_bytes": runtime_backend.get("accelerator_total_bytes"),
                    "accelerator_available_bytes": runtime_backend.get("accelerator_available_bytes"),
                },
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
                    "polyakov_dirs": all_mu_dirs,
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

    def single_measurement(U_field, progress_cb=None):
        if sampler.enabled:
            plaq_re, _ = scalar_complex(sampler.mean(g.qcd.gauge.plaquette(U_field, field=True)))
            plaq = plaq_re
        else:
            plaq = float(g.qcd.gauge.plaquette(U_field))

        polyakov_loops = {}
        if not skip_flux:
            polyakov_loops = measure_polyakov_loops(U_field, all_mu_dirs, L, sampler=sampler)
        if progress_cb is not None:
            for mu in all_mu_dirs:
                progress_cb(
                    "measure_polyakov",
                    1,
                    cursor={
                        "kind": "polyakov",
                        "tdir": int(mu),
                    },
                )

        loops_acc = {}  # Key: "R{r}_T{t}", Value: list of (re, im) tuples
        flux_profiles_acc = []  # List of [val_at_r0, val_at_r1, ...]

        loop_time_total = 0.0
        flux_time_total = 0.0

        # Iterate one time-direction at a time to save memory
        for tdir, U_use in iter_measurement_fields(
            U_field, time_dirs, smear_steps, smear_ops, smear_spatial_only
        ):
            # Filter orientations for this tdir
            orientations_for_tdir = [sdir for (td, sdir) in orientations if td == tdir]
            if orientations_for_tdir:
                t0_loop = time.time()
                measure_loops_for_tdir(
                    U_use, tdir, Rs, Ts, orientations_for_tdir, sampler, loops_acc, progress_cb=progress_cb
                )
                loop_time_total += time.time() - t0_loop

                if not skip_flux:
                    t0_flux = time.time()
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
                        progress_cb=progress_cb,
                    )
                    flux_time_total += time.time() - t0_flux
            # Free C++ lattice temporaries from smearing/loops/flux for this tdir
            del U_use
            gc.collect()

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
            "polyakov_loops": polyakov_loops,
            "profiling": {
                "loop_time": loop_time_total,
                "flux_time": flux_time_total,
            },
        }

    def measure_with_variance_reduction(U_field, progress_cb=None):
        if multilevel_blocks == 1 and multihit_samples == 1:
            return single_measurement(U_field, progress_cb=progress_cb)

        blocks = []
        U_block = g.copy(U_field)
        for ib in range(multilevel_blocks):
            if ib > 0 and multilevel_sweeps > 0:
                for _ in range(multilevel_sweeps):
                    one_sweep(U_block, hb_estimator, action, mask, mask_rb, all_mu_dirs)
                    if progress_cb is not None:
                        progress_cb("multilevel_sweeps", 1)

            hits = []
            U_hit = g.copy(U_block)
            for ih in range(multihit_samples):
                if ih > 0 and multihit_temporal_sweeps > 0:
                    for _ in range(multihit_temporal_sweeps):
                        one_sweep(U_hit, hb_estimator, action, mask, mask_rb, time_dirs)
                        if progress_cb is not None:
                            progress_cb("multihit_temporal_sweeps", 1)
                hits.append(single_measurement(U_hit, progress_cb=progress_cb))
                gc.collect()

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
                "polyakov_dirs": all_mu_dirs,
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
            
            # Rewrite JSONL to ensure consistency with checkpoint.
            # Important: truncate even when measurements is empty, otherwise
            # stale points from prior runs can remain visible in the dashboard.
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
    last_substep_progress_write = 0.0
    for i in range(therm_start, ntherm):
        substate = {"done": 0, "total": max(1, 2 * len(all_mu_dirs))}

        def on_therm_substep(done, total):
            nonlocal last_substep_progress_write
            substate["done"] = int(done)
            substate["total"] = int(max(1, total))
            now = time.time()
            # Publish mid-sweep progress at a bounded rate to keep UI responsive.
            if (now - last_substep_progress_write) < progress_substep_min_interval and done < total:
                return
            write_json(
                progress_file,
                make_progress_payload(
                    seed=seed,
                    out_dir=out_dir,
                    phase="thermalization",
                    ntherm=ntherm,
                    nmeas=nmeas,
                    therm_done=i,
                    meas_done=meas_start,
                    sweeps_done=sweeps_done,
                    total_sweeps=total_sweeps,
                    elapsed_sec=time.time() - run_start,
                    therm_sweep_substep_done=substate["done"],
                    therm_sweep_substep_total=substate["total"],
                    last_plaquette=last_plaquette,
                    last_loop_re=last_loop_re,
                    last_flux0=last_flux0,
                    done=False,
                ),
            )
            last_substep_progress_write = now

        one_sweep(U, hb_chain, action, mask, mask_rb, all_mu_dirs, step_cb=on_therm_substep)
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

    # If thermalization is already complete (e.g. resumed at ntherm), publish
    # production phase immediately so dashboard reflects measuring threads
    # without waiting for the first full measurement to finish.
    if not stop_requested and meas_start < nmeas and therm_done >= ntherm:
        write_json(
            progress_file,
            make_progress_payload(
                seed=seed,
                out_dir=out_dir,
                phase="production",
                ntherm=ntherm,
                nmeas=nmeas,
                therm_done=therm_done,
                meas_done=meas_start,
                sweeps_done=sweeps_done,
                total_sweeps=total_sweeps,
                elapsed_sec=time.time() - run_start,
                meas_cfg_index=meas_start + 1,
                meas_cfg_total=nmeas,
                meas_cfg_stage="ready",
                meas_cfg_substep_done=0,
                meas_cfg_substep_total=1,
                last_plaquette=last_plaquette,
                last_loop_re=last_loop_re,
                last_flux0=last_flux0,
                done=False,
            ),
        )

    last_meas_substep_progress_write = 0.0
    for i in range(meas_start, nmeas):
        if stop_requested:
            break

        single_meas_steps = single_measurement_step_count(
            time_dirs,
            orientations,
            Nd,
            Rs,
            Ts,
            flux_r_perp_max,
            polyakov_dirs_count=len(all_mu_dirs),
        )
        meas_stage_steps = single_meas_steps * multilevel_blocks * multihit_samples
        meas_extra_sweeps = max(0, multilevel_blocks - 1) * multilevel_sweeps
        meas_extra_sweeps += multilevel_blocks * max(0, multihit_samples - 1) * multihit_temporal_sweeps
        skip_sweep_substeps = max(1, 2 * len(all_mu_dirs))
        meas_substate = {
            "done": 0,
            "total": max(1, nskip * skip_sweep_substeps + meas_extra_sweeps + meas_stage_steps),
            "stage": "skip_sweeps" if nskip > 0 else "measure_tdirs",
            "cursor": None,
        }

        def emit_meas_subprogress(force=False):
            nonlocal last_meas_substep_progress_write
            now = time.time()
            if (
                not force
                and (now - last_meas_substep_progress_write) < progress_substep_min_interval
                and meas_substate["done"] < meas_substate["total"]
            ):
                return
            cursor = meas_substate.get("cursor") or {}
            write_json(
                progress_file,
                make_progress_payload(
                    seed=seed,
                    out_dir=out_dir,
                    phase="production",
                    ntherm=ntherm,
                    nmeas=nmeas,
                    therm_done=ntherm,
                    meas_done=i,
                    sweeps_done=sweeps_done,
                    total_sweeps=total_sweeps,
                    elapsed_sec=now - run_start,
                    meas_cfg_index=i + 1,
                    meas_cfg_total=nmeas,
                    meas_cfg_stage=meas_substate["stage"],
                    meas_cfg_substep_done=meas_substate["done"],
                    meas_cfg_substep_total=meas_substate["total"],
                    meas_cursor_kind=cursor.get("kind"),
                    meas_cursor_tdir=cursor.get("tdir"),
                    meas_cursor_sdir=cursor.get("sdir"),
                    meas_cursor_r=cursor.get("r"),
                    meas_cursor_t=cursor.get("t"),
                    meas_cursor_r_perp=cursor.get("r_perp"),
                    meas_cursor_shift=cursor.get("shift"),
                    last_plaquette=last_plaquette,
                    last_loop_re=last_loop_re,
                    last_flux0=last_flux0,
                    done=False,
                ),
            )
            last_meas_substep_progress_write = now

        def advance_meas_subprogress(stage, inc=1, force=False, cursor=None, absolute_done=None):
            meas_substate["stage"] = stage
            if cursor is not None:
                meas_substate["cursor"] = cursor
            if absolute_done is not None:
                meas_substate["done"] = min(
                    meas_substate["total"],
                    max(0, int(absolute_done)),
                )
            elif inc > 0:
                meas_substate["done"] = min(meas_substate["total"], meas_substate["done"] + int(inc))
            emit_meas_subprogress(force=force)

        emit_meas_subprogress(force=True)
        skip_mu_counts = [0 for _ in range(max(1, len(all_mu_dirs)))]
        for skip_idx in range(nskip):
            skip_base_done = int(meas_substate["done"])
            saw_skip_substep = False
            mu_count = max(1, len(all_mu_dirs))
            skip_step_seen = 0

            def on_skip_substep(done, total):
                nonlocal saw_skip_substep, skip_step_seen
                saw_skip_substep = True
                d = int(max(0, min(total, done)))
                if d <= skip_step_seen:
                    return
                for local_step in range(skip_step_seen, d):
                    mu_local = int(local_step % mu_count)
                    skip_mu_counts[mu_local] += 1
                skip_step_seen = d
                mu_idx = int((max(0, d - 1)) % mu_count)
                advance_meas_subprogress(
                    "skip_sweeps",
                    inc=0,
                    cursor={
                        "kind": "skip",
                        "tdir": mu_idx,
                        "r": int(skip_idx + 1),
                        "t": int(nskip),
                        "shift": [int(x) for x in skip_mu_counts],
                    },
                    absolute_done=skip_base_done + d,
                )

            one_sweep(U, hb_chain, action, mask, mask_rb, all_mu_dirs, step_cb=on_skip_substep)
            sweeps_done += 1
            if not saw_skip_substep:
                for local_step in range(skip_sweep_substeps):
                    mu_local = int(local_step % mu_count)
                    skip_mu_counts[mu_local] += 1
                mu_idx = int((max(0, skip_sweep_substeps - 1)) % mu_count)
                advance_meas_subprogress(
                    "skip_sweeps",
                    inc=skip_sweep_substeps,
                    cursor={
                        "kind": "skip",
                        "tdir": mu_idx,
                        "r": int(skip_idx + 1),
                        "t": int(nskip),
                        "shift": [int(x) for x in skip_mu_counts],
                    },
                )

        measured = measure_with_variance_reduction(U, progress_cb=advance_meas_subprogress)
        meas_substate["done"] = meas_substate["total"]
        advance_meas_subprogress("finalize", inc=0, force=True, cursor={"kind": "finalize"})

        # Free C++ lattice temporaries from smearing/loops/flux measurements.
        # The grid stays the same so caches stabilize, but gc.collect() ensures
        # reference-cycled C++ objects are freed promptly.
        gc.collect()

        item = {
            "idx": i,
            "plaquette": measured["plaquette"],
            "loops": measured["loops"],
            "flux_profile_r_perp": measured["flux_profile_r_perp"],
            "polyakov_loops": measured.get("polyakov_loops", {}),
            "profiling": measured.get("profiling", {}),
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

        checkpoint_now = checkpoint_every > 0 and (((i + 1) % checkpoint_every) == 0 or (i + 1) == nmeas)
        if checkpoint_now:
            # Persist completed measurement state first so resume does not drop back.
            save_checkpoint("production", ntherm, i + 1)

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
                meas_cfg_index=i + 1,
                meas_cfg_total=nmeas,
                meas_cfg_stage="complete",
                meas_cfg_substep_done=meas_substate["total"],
                meas_cfg_substep_total=meas_substate["total"],
                last_plaquette=last_plaquette,
                last_loop_re=last_loop_re,
                last_flux0=last_flux0,
                done=False,
            ),
        )

        if save_cfg_every > 0 and ((i + 1) % save_cfg_every) == 0:
            cfg_file = os.path.join(cfg_dir, f"cfg_{seed}_{i+1:05d}.cfg")
            g.save(cfg_file, U)

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

    mean_polyakov = {}
    poly_keys = sorted(
        {
            k
            for m in measurements
            for k in ((m.get("polyakov_loops") or {}).keys())
        }
    )
    for k in poly_keys:
        vals = [m["polyakov_loops"][k] for m in measurements if isinstance(m.get("polyakov_loops", {}).get(k), dict)]
        if not vals:
            continue
        re = mean([float(v.get("re", 0.0)) for v in vals])
        im = mean([float(v.get("im", 0.0)) for v in vals])
        mean_polyakov[k] = {
            "re": re,
            "im": im,
            "phase": float(math.atan2(im, re)),
        }

    output = {
        "meta": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
            "polyakov_dirs": all_mu_dirs,
            "autocorr_max_lag": max_lag,
            "tau_int_plaquette": tau_plaq,
            "tau_int_loop_re": tau_loop,
            "autocorr_bin_size": bin_size,
            "veff_n_binned": n_binned,
            "flux_n_binned": flux_n_binned,
            "compute_backend": runtime_backend.get("backend"),
            "compute_pipeline": pipeline_label,
            "grid_acceleration": runtime_backend.get("grid_acceleration"),
            "grid_simd": runtime_backend.get("grid_simd"),
            "grid_threading": runtime_backend.get("grid_threading"),
            "grid_summary_path": runtime_backend.get("grid_summary_path"),
            "accelerator_total_bytes": runtime_backend.get("accelerator_total_bytes"),
            "accelerator_available_bytes": runtime_backend.get("accelerator_available_bytes"),
        },
        "mean_plaquette": mean([m["plaquette"] for m in measurements]),
        "mean_loops": mean_loops,
        "mean_polyakov_loops": mean_polyakov,
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
