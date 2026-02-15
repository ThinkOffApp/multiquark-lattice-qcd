#!/usr/bin/env python3
#
# Minimal SU(2) pure-gauge scan for:
# 1) 2q potential from Wilson loops W(R,T)
# 2) 2q flux-tube signal from connected plaquette-Wilson-loop correlator
#
import json
import math
import os
import time
from datetime import datetime, UTC

import gpt as g


def parse_list_int(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


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


def measure_flux_profile(U, flux_r, flux_t, flux_r_perp_max):
    # One fixed orientation: time=mu3, spatial separation along x=mu0
    w_field = g.qcd.gauge.rectangle(
        U, [[(3, flux_t, 0, flux_r)]], field=True, trace=True, real=False
    )
    p_field = g.qcd.gauge.plaquette(U, field=True)
    avg_w = g.sum(w_field) / w_field.grid.gsites
    avg_p = g.sum(p_field) / p_field.grid.gsites
    avg_w_re = float(avg_w.real)

    if abs(avg_w_re) < 1e-20:
        raise RuntimeError("Wilson-loop average is too small for stable flux normalization")

    x_mid = flux_r // 2
    t_mid = flux_t // 2

    profile = []
    for r_perp in range(flux_r_perp_max + 1):
        # Average over +/- y and +/- z at fixed x_mid, t_mid.
        shifts = unique_shifts(
            [
                [x_mid, +r_perp, 0, t_mid],
                [x_mid, -r_perp, 0, t_mid],
                [x_mid, 0, +r_perp, t_mid],
                [x_mid, 0, -r_perp, t_mid],
            ]
        )
        corr_vals = []
        for s in shifts:
            p_shift = shifted(p_field, s)
            wp = g.sum(w_field * p_shift) / w_field.grid.gsites
            connected = (wp / avg_w) - avg_p
            corr_vals.append(float(connected.real))
        profile.append(sum(corr_vals) / len(corr_vals))
    return profile


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
    save_cfg_every = g.default.get_int("--save-cfg-every", 1)
    checkpoint_every = g.default.get_int("--checkpoint-every", 20)
    resume = g.default.get_int("--resume", 1) != 0

    precision = g.default.get("--precision", "single")
    prec = g.single if precision == "single" else g.double

    out_dir = g.default.get("--out", "results/su2_signal_scan")
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
    U = g.qcd.gauge.unit(grid)
    Nd = len(U)

    mask_rb = g.complex(grid_eo)
    mask_rb[:] = 1
    mask = g.complex(grid)

    action = g.qcd.gauge.action.wilson(beta)
    hb = g.algorithms.markov.su2_heat_bath(rng)

    g.default.push_verbose("su2_heat_bath", False)
    g.message(f"Lattice={L}, precision={precision}, beta={beta}")
    g.message(f"Therm={ntherm}, meas={nmeas}, skip={nskip}")
    g.message(f"R={Rs}, T={Ts}, flux(R,T)=({flux_r},{flux_t}), r_perp_max={flux_r_perp_max}")
    g.message(f"Progress file: {progress_file}")
    g.message(f"Live file: {live_file}")
    g.message(
        f"Checkpoint every={checkpoint_every}, save_cfg_every={save_cfg_every}, resume={int(resume)}"
    )

    run_start = time.time()
    total_sweeps = ntherm + nmeas * nskip
    sweeps_done = 0
    last_plaquette = None
    last_loop_re = None
    last_flux0 = None
    therm_start = 0
    meas_start = 0
    measurements = []

    def write_live():
        write_json(
            live_file,
            {
                "meta": {
                    "seed": seed,
                    "beta": beta,
                    "L": L,
                    "R": Rs,
                    "T": Ts,
                    "flux_r": flux_r,
                    "flux_t": flux_t,
                    "flux_r_perp_max": flux_r_perp_max,
                },
                "measurements": measurements,
            },
        )

    def one_sweep():
        for cb in [g.even, g.odd]:
            mask[:] = 0
            mask_rb.checkerboard(cb)
            g.set_checkerboard(mask, mask_rb)
            for mu in range(Nd):
                hb(U[mu], action.staple(U, mu), mask)

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
                    "flux_r": flux_r,
                    "flux_t": flux_t,
                    "flux_r_perp_max": flux_r_perp_max,
                    "precision": precision,
                },
            },
        )

    if resume and os.path.exists(checkpoint_file) and os.path.exists(checkpoint_cfg_file):
        ckpt = read_json(checkpoint_file)
        if ckpt and ckpt.get("seed") == seed:
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
        else:
            g.message("Checkpoint found but seed mismatch; starting fresh.")

    # Always rewrite live state at startup so stale points from older runs are not shown.
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
    for i in range(therm_start, ntherm):
        one_sweep()
        sweeps_done += 1
        if (i + 1) % therm_log_every == 0:
            last_plaquette = float(g.qcd.gauge.plaquette(U))
            g.message(f"Thermalization {i+1}/{ntherm}, P={last_plaquette}")
        # Always write thermalization progress so dashboard reflects exact sweep count.
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

    for i in range(meas_start, nmeas):
        for _ in range(nskip):
            one_sweep()
            sweeps_done += 1

        plaq = float(g.qcd.gauge.plaquette(U))
        loops = {}
        for r in Rs:
            for t in Ts:
                # Fix temporal direction to mu=3 and average over spatial nu=0,1,2
                val = g.qcd.gauge.rectangle(U, t, r, 3, trace=True, real=False)
                re, im = scalar_complex(val)
                loops[f"R{r}_T{t}"] = {"re": re, "im": im}

        flux_profile = measure_flux_profile(U, flux_r, flux_t, flux_r_perp_max)

        item = {
            "idx": i,
            "plaquette": plaq,
            "loops": loops,
            "flux_profile_r_perp": flux_profile,
        }
        measurements.append(item)
        last_plaquette = plaq
        last_loop_re = loops.get(f"R{flux_r}_T{flux_t}", {"re": 0.0})["re"]
        last_flux0 = flux_profile[0]
        g.message(
            f"Meas {i+1}/{nmeas}: P={plaq:.8f}, "
            f"W(R={flux_r},T={flux_t})={loops.get(f'R{flux_r}_T{flux_t}', {'re': 0.0})['re']:.6e}, "
            f"flux(r_perp=0)={flux_profile[0]:.6e}"
        )
        write_live()
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

    # Ensemble means and simple Veff estimates from mean loops.
    mean_loops = {}
    for r in Rs:
        for t in Ts:
            key = f"R{r}_T{t}"
            vals_re = [m["loops"][key]["re"] for m in measurements]
            vals_im = [m["loops"][key]["im"] for m in measurements]
            mean_loops[key] = {
                "re": sum(vals_re) / len(vals_re),
                "im": sum(vals_im) / len(vals_im),
            }

    veff = {}
    for r in Rs:
        for t in Ts:
            if (t + 1) in Ts:
                a = mean_loops[f"R{r}_T{t}"]["re"]
                b = mean_loops[f"R{r}_T{t+1}"]["re"]
                if a > 0.0 and b > 0.0:
                    veff[f"R{r}_T{t}to{t+1}"] = -math.log(b / a)

    mean_flux = []
    for r_perp in range(flux_r_perp_max + 1):
        vals = [m["flux_profile_r_perp"][r_perp] for m in measurements]
        mean_flux.append(sum(vals) / len(vals))

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
            "flux_r": flux_r,
            "flux_t": flux_t,
            "flux_r_perp_max": flux_r_perp_max,
        },
        "mean_plaquette": sum(m["plaquette"] for m in measurements) / len(measurements),
        "mean_loops": mean_loops,
        "veff": veff,
        "mean_flux_profile_r_perp": mean_flux,
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
