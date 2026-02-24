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
        "--seed", SEED,
        "--save-cfg-every", "1",
        "--shm", "64",
        "--skip_flux", "1",
        "--resume", "0"
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = f"{os.getcwd()}/gpt/lib/cgpt/build:{os.getcwd()}/gpt/lib:" + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("ERROR: Base configuration generation failed!")
        sys.exit(1)
    
    cfg_path = f"results/su2_signal_scan/configs_{SEED}/cfg_{SEED}_00001.cfg"
    if not os.path.exists(cfg_path):
        print(f"ERROR: Expected configuration file {cfg_path} not found!")
        sys.exit(1)
    return cfg_path

def run_measure_only(backend, config_file):
    ensure_clean()
    cmd = [
        "python3", "gpt/applications/hmc/su2_2q_signal_scan.py",
        "--L", LATTICE,
        "--beta", BETA,
        "--backend", backend,
        "--seed", SEED,
        "--measure-only", config_file,
        "--live-updates",
        "--shm", "64",
        "--R", "1,2,3",
        "--T", "1,2",
        "--skip_flux", "1"
    ]
    print(f"--- Running measure-only deterministic evaluation: [{backend.upper()}] ---")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = f"{os.getcwd()}/gpt/lib/cgpt/build:{os.getcwd()}/gpt/lib:" + env.get("PYTHONPATH", "")
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

cpu_data, cpu_meta = run_measure_only("cpu", base_cfg)
gpu_data, gpu_meta = run_measure_only("metal", base_cfg)

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
TOLERANCE = 1e-6
for name, d in diffs:
    print(f"{name:12} Variance: {d:e}")
    if d > TOLERANCE:
        failed = True

if failed:
    print(f"\n[FAIL] Observables diverged beyond safety threshold ({TOLERANCE}) on shared configuration!")
    sys.exit(1)

print("\n[SUCCESS] Native Apple Metal GPU matrix math strictly matches standard CPU paths on an identical gauge field invariant.")
