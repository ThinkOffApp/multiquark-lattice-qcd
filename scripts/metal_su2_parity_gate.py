import subprocess
import json
import sys
import numpy as np
import os
import shutil
import hashlib

def hash_file(filepath):
    h = hashlib.sha256()
    if os.path.isdir(filepath):
        for root, dirs, files in os.walk(filepath):
            for file in sorted(files):
                full_path = os.path.join(root, file)
                with open(full_path, 'rb') as f:
                    while chunk := f.read(8192):
                        h.update(chunk)
    else:
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
    return h.hexdigest()

LATTICE = "8,8,8,8"
BETA = "2.4"
SEED = "42"

def ensure_clean():
    for f in [f"results/su2_signal_scan/live_{SEED}.json",
              f"results/su2_signal_scan/live_{SEED}.jsonl",
              f"results/su2_signal_scan/progress_{SEED}.json",
              f"results/su2_signal_scan/checkpoint_{SEED}.json",
              f"results/su2_signal_scan/checkpoint_{SEED}.cfg"]:
        if os.path.exists(f):
            if os.path.isdir(f):
                shutil.rmtree(f)
            else:
                os.remove(f)

def run_generate():
    ensure_clean()
    print("--- Generating base CPU configuration for deterministic parity check ---")
    cmd = [
        "python3", "gpt/applications/hmc/su2_2q_signal_scan.py",
        "--L", LATTICE,
        "--beta", BETA,
        "--ntherm", "10",
        "--nmeas", "1",
        "--backend", "cpu",
        "--precision", "single",
        "--seed", SEED,
        "--save-cfg-every", "1",
        "--shm", "64",
        "--skip_flux", "1",
        "--resume", "0"
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = f"{os.getcwd()}/gpt/lib/cgpt/build-cpu-single:{os.getcwd()}/gpt/lib:" + env.get("PYTHONPATH", "")
    env["GRID_CONFIG_SUMMARY"] = f"{os.getcwd()}/Grid/build-cpu-single/grid.configure.summary"
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("ERROR: Base configuration generation failed!")
        sys.exit(1)
    
    cfg_path = f"results/su2_signal_scan/configs_{SEED}/cfg_{SEED}_00001.cfg"
    if not os.path.exists(cfg_path):
        print(f"ERROR: Expected configuration file {cfg_path} not found!")
        sys.exit(1)
    return cfg_path

def run_measure_only(backend, config_file, require_accel=False):
    ensure_clean()
    cmd = [
        "python3", "gpt/applications/hmc/su2_2q_signal_scan.py",
        "--L", LATTICE,
        "--beta", BETA,
        "--backend", backend,
        "--precision", "single",
        "--seed", SEED,
        "--measure-only", config_file,
        "--live-updates",
        "--shm", "64",
        "--R", "1,2,3",
        "--T", "1,2",
        "--skip_flux", "1"
    ]
    if require_accel:
        cmd.extend(["--require-accelerator", "1"])
    print(f"--- Running measure-only deterministic evaluation: [{backend.upper()}] ---")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    
    if backend == "cpu":
        build_dir = "build-cpu-single"
    else:
        build_dir = "build-metal-single"
        
    env["PYTHONPATH"] = f"{os.getcwd()}/gpt/lib/cgpt/{build_dir}:{os.getcwd()}/gpt/lib:" + env.get("PYTHONPATH", "")
    env["GRID_CONFIG_SUMMARY"] = f"{os.getcwd()}/Grid/{build_dir}/grid.configure.summary"
    result = subprocess.run(cmd, env=env)
    
    if result.returncode != 0:
        print(f"ERROR: Measure-only run on {backend} failed!")
        sys.exit(1)
        
    measurements = []
    meta = {}
    with open(f"results/su2_signal_scan/live_{SEED}.jsonl", "r") as f:
        for line in f:
            measurements.append(json.loads(line))
            
    progress_file = f"results/su2_signal_scan/live_{SEED}.json"
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            meta = json.load(f).get("meta", {})

    ensure_clean()
    return measurements, meta

print("==================================================")
print(" Deterministic IEEE-754 Metal Parity Validator    ")
print("==================================================")

base_cfg = run_generate()
cfg_hash = hash_file(base_cfg)

cpu_data, cpu_meta = run_measure_only("cpu", base_cfg, require_accel=False)
gpu_data, gpu_meta = run_measure_only("metal", base_cfg, require_accel=True)

if not cpu_data or not gpu_data:
    print("FATAL: Failed to parse measurement output from evaluation hooks.")
    sys.exit(1)

cpu_final = cpu_data[-1]
gpu_final = gpu_data[-1]

print("\n--- Runtime Environment ---")
print(f"Shared Gauge Config: {base_cfg}")
print(f"Config SHA-256 Hash: {cfg_hash}")
print(f"Backend Reported [CPU]: {cpu_meta.get('compute_backend')}  |  [GPU]: {gpu_meta.get('compute_backend')}")
print(f"Acceleration     [CPU]: {cpu_meta.get('grid_acceleration')}  |  [GPU]: {gpu_meta.get('grid_acceleration')}")
print(f"Total Accel Mem  [CPU]: {cpu_meta.get('accelerator_total_bytes')}  |  [GPU]: {gpu_meta.get('accelerator_total_bytes')}")
print(f"Loaded .so       [CPU]: {cpu_meta.get('loaded_cgpt_so')}  |  [GPU]: {gpu_meta.get('loaded_cgpt_so')}")

cpu_accel = cpu_meta.get('accelerator_total_bytes', 0)
gpu_accel = gpu_meta.get('accelerator_total_bytes', 0)

if gpu_accel == 0:
    print("\n[FAIL] Accelerator total bytes is 0 on GPU leg. Metal GPU was not actually engaged.")
    sys.exit(1)

if cpu_accel != 0:
    print(f"\n[FAIL] Accelerator total bytes is {cpu_accel} on CPU leg. True isolation failed.")
    sys.exit(1)
    
if cpu_meta.get('loaded_cgpt_so') == gpu_meta.get('loaded_cgpt_so'):
    print("\n[FAIL] Both legs loaded the identical cgpt.so library. True isolation failed.")
    sys.exit(1)

cpu_accel_name = (cpu_meta.get('grid_acceleration') or "").lower()
gpu_accel_name = (gpu_meta.get('grid_acceleration') or "").lower()
if cpu_accel_name != "none":
    print(f"\n[FAIL] CPU leg grid_acceleration='{cpu_accel_name}', expected 'none'.")
    sys.exit(1)
if gpu_accel_name != "metal":
    print(f"\n[FAIL] Metal leg grid_acceleration='{gpu_accel_name}', expected 'metal'.")
    sys.exit(1)

print("\n--- Build Hashes (manifest) ---")
for label, meta, build_dir in (("CPU", cpu_meta, "build-cpu-single"),
                               ("METAL", gpu_meta, "build-metal-single")):
    so_path = meta.get('loaded_cgpt_so') or ""
    so_hash = hash_file(so_path) if so_path and os.path.exists(so_path) else "n/a"
    summary_path = f"{os.getcwd()}/Grid/{build_dir}/grid.configure.summary"
    summary_hash = hash_file(summary_path) if os.path.exists(summary_path) else "n/a"
    print(f"  [{label}] cgpt.so sha256={so_hash}  path={so_path}")
    print(f"  [{label}] grid.configure.summary sha256={summary_hash}  path={summary_path}")

print("\n--- Final Observables Parity Vector ---")
print(f"Plaquette      [CPU]: {cpu_final['plaquette']}  |  [GPU]: {gpu_final['plaquette']}")

w_keys = ["R1_T1", "R2_T1", "R3_T1", "R2_T2", "R3_T2"]
diffs = []
plaq_diff = abs(cpu_final['plaquette'] - gpu_final['plaquette'])
diffs.append(("Plaquette", plaq_diff))

for k in w_keys:
    if k in cpu_final['loops'] and k in gpu_final['loops']:
        cv = cpu_final['loops'][k]['re']
        gv = gpu_final['loops'][k]['re']
        diff = abs(cv - gv)
        diffs.append((f"W({k})", diff))
        print(f"W({k})  [CPU]: {cv}  |  [GPU]: {gv}  |  Diff: {diff:e}")

print("\n--- Error Analysis ---")
failed = False
TOLERANCE = 1e-5
for name, d in diffs:
    print(f"{name:12} Variance: {d:e}")
    if d > TOLERANCE:
        failed = True

if failed:
    print(f"\n[FAIL] Observables diverged beyond safety threshold ({TOLERANCE}) on shared configuration!")
    sys.exit(1)

print(f"\n[SUCCESS] On a shared gauge configuration, Metal-built and CPU-built measure-only legs produced plaquette and Wilson-loop observables agreeing within {TOLERANCE:.0e}. This is a pure-gauge SU(2) parity check; it does not exercise Wilson Dslash (covered separately by the Wilson Dslash regression).")
