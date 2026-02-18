#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def progress_path(out_dir: Path, seed: str) -> Path:
    return out_dir / f"progress_{seed}.json"


def checkpoint_path(out_dir: Path, seed: str) -> Path:
    return out_dir / f"checkpoint_{seed}.json"


def all_done(out_dir: Path, seeds):
    status = []
    for seed in seeds:
        p = read_json(progress_path(out_dir, seed))
        done = bool(p and p.get("done"))
        phase = p.get("phase") if p else "missing"
        therm_done = p.get("therm_done") if p else None
        ntherm = p.get("ntherm") if p else None
        meas_done = p.get("meas_done") if p else None
        nmeas = p.get("nmeas") if p else None
        status.append((seed, done, phase, therm_done, ntherm, meas_done, nmeas))
    return status, all(x[1] for x in status)


def existing_or_running(out_dir: Path, seed: str):
    p = read_json(progress_path(out_dir, seed))
    if not p:
        return False
    return True


def build_job(seed, next_seed, ckpt, out_dir: Path):
    params = (ckpt or {}).get("params", {})
    beta = params.get("beta", 2.4)
    ntherm = int(params.get("ntherm", 200))
    nmeas = int(params.get("nmeas", 200))
    nskip = int(params.get("nskip", 5))
    Rs = params.get("R", [2, 3, 4, 6, 8, 12])
    Ts = params.get("T", [2, 3, 4, 5, 6])
    flux_r = int(params.get("flux_r", 6))
    flux_t = int(params.get("flux_t", 4))
    flux_r_perp_max = int(params.get("flux_r_perp_max", 6))
    precision = params.get("precision", "single")

    cmd = [
        "python3",
        "applications/hmc/su2_2q_signal_scan.py",
        "--seed",
        next_seed,
        "--L",
        "24,24,24,24",
        "--beta",
        str(beta),
        "--ntherm",
        str(ntherm),
        "--nmeas",
        str(nmeas),
        "--nskip",
        str(nskip),
        "--R",
        ",".join(str(x) for x in Rs),
        "--T",
        ",".join(str(x) for x in Ts),
        "--flux-r",
        str(flux_r),
        "--flux-t",
        str(flux_t),
        "--flux-rperp-max",
        str(flux_r_perp_max),
        "--save-cfg-every",
        "1",
        "--checkpoint-every",
        "20",
        "--resume",
        "0",
        "--precision",
        str(precision),
        "--out",
        str(out_dir),
    ]
    return cmd


def repo_root_from_script() -> Path:
    # tools/su2_chain_to_24.py -> repo root is parent of tools/
    return Path(__file__).resolve().parents[1]


def default_out_dir() -> str:
    env = os.environ.get("SU2_OUT_DIR", "").strip()
    if env:
        return env
    return str(repo_root_from_script() / "results" / "su2_signal_scan")


def default_gpt_dir() -> str:
    env = os.environ.get("SU2_GPT_DIR", "").strip()
    if env:
        return env
    return str(repo_root_from_script() / "gpt")


def launch_job(gpt_dir: Path, cmd, log_path: Path):
    shell_cmd = (
        f"cd {shlex.quote(str(gpt_dir))} && "
        "source lib/cgpt/build/source.sh && "
        + " ".join(shlex.quote(x) for x in cmd)
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = log_path.open("a", encoding="utf-8")
    p = subprocess.Popen(
        ["/usr/bin/env", "bash", "-lc", shell_cmd],
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return p.pid


def main():
    p = argparse.ArgumentParser(description="Auto-chain SU2 16^4 run to 24^4 once complete.")
    p.add_argument(
        "--out-dir",
        default=default_out_dir(),
        help="Output directory (default: $SU2_OUT_DIR or <repo>/results/su2_signal_scan)",
    )
    p.add_argument(
        "--gpt-dir",
        default=default_gpt_dir(),
        help="GPT directory (default: $SU2_GPT_DIR or <repo>/gpt)",
    )
    p.add_argument(
        "--seeds",
        default="petrus-su2-signal,petrus-su2-signal-b,petrus-su2-signal-c,petrus-su2-signal-d",
    )
    p.add_argument("--next-suffix", default="-L24")
    p.add_argument("--poll-sec", type=int, default=20)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    gpt_dir = Path(args.gpt_dir).resolve()
    seeds = [x.strip() for x in args.seeds.split(",") if x.strip()]
    next_seeds = [f"{s}{args.next_suffix}" for s in seeds]

    print(f"[chain] monitoring seeds: {', '.join(seeds)}")
    print(f"[chain] next seeds: {', '.join(next_seeds)}")
    print(f"[chain] out_dir={out_dir}")
    print(f"[chain] gpt_dir={gpt_dir}")

    while True:
        status, done = all_done(out_dir, seeds)
        summary = ", ".join(
            f"{s}: {('done' if d else f'{ph} {td}/{nt} meas {md}/{nm}')}"
            for s, d, ph, td, nt, md, nm in status
        )
        print(f"[chain] status: {summary}", flush=True)

        if done:
            print("[chain] all source runs complete; launching 24^4 jobs", flush=True)
            for seed, next_seed in zip(seeds, next_seeds):
                if existing_or_running(out_dir, next_seed):
                    print(f"[chain] skip {next_seed} (already exists)")
                    continue
                ckpt = read_json(checkpoint_path(out_dir, seed))
                cmd = build_job(seed, next_seed, ckpt, out_dir)
                log_path = out_dir / f"run_{next_seed}.log"
                pid = launch_job(gpt_dir, cmd, log_path)
                print(f"[chain] launched {next_seed} pid={pid} log={log_path}")
            print("[chain] done")
            return

        if args.once:
            return

        time.sleep(max(2, args.poll_sec))


if __name__ == "__main__":
    main()
