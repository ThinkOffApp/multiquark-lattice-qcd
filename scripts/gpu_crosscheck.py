import subprocess
import json
import sys
import numpy as np
import os
import shutil

# This cross-check fulfills @ether's request to assert key observables
# automatically matching inside a tiny lattice deterministic run.
LATTICE = "4.4.4.4"
BETA = "2.2"
THERM = "10"
MEAS = "5"
SEED = "42"

def ensure_clean():
    if os.path.exists(f"live_{SEED}.jsonl"):
        os.remove(f"live_{SEED}.jsonl")
    if os.path.exists(f"su2_config_{SEED}.bin"):
        os.remove(f"su2_config_{SEED}.bin")

def run_backend(backend):
    ensure_clean()
    cmd = [
        "python3", "gpt/applications/hmc/su2_2q_signal_scan.py",
        "--grid", LATTICE,
        "--beta", BETA,
        "--therm", THERM,
        "--meas", MEAS,
        "--backend", backend,
        "--seed", SEED,
        "--live-updates"
    ]
    print(f"Running SU(2) benchmark with {backend} backend...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {backend} run failed:\n{result.stderr}")
        sys.exit(1)
        
    measurements = []
    with open(f"live_{SEED}.jsonl", "r") as f:
        for line in f:
            measurements.append(json.loads(line))
            
    # Clean up to not interfere with the next run
    ensure_clean()
    return measurements

print("========================================")
print(" CPU vs GPU Observables Cross-Check     ")
print("========================================")

cpu_data = run_backend("cpu")
gpu_data = run_backend("metal")

if len(cpu_data) != len(gpu_data):
    print("FATAL: Mismatched measurement counts!")
    sys.exit(1)

# Extract final physics values
cpu_final = cpu_data[-1]
gpu_final = gpu_data[-1]

print("\n--- Final Observables Parity ---")
print(f"Plaquette [CPU]: {cpu_final['plaquette']}  |  [GPU]: {gpu_final['plaquette']}")
print(f"W(R=2,T=2) [CPU]: {cpu_final['wilson_loop']}  |  [GPU]: {gpu_final['wilson_loop']}")

plaq_diff = abs(cpu_final['plaquette'] - gpu_final['plaquette'])
wilson_diff = abs(cpu_final['wilson_loop'] - gpu_final['wilson_loop'])

print(f"\nPlaquette Difference:  {plaq_diff:e}")
print(f"Wilson Difference:     {wilson_diff:e}")

# The threshold allows for floating-point accumulated variance over the HMC integrator trajectories.
# Since HMC trajectories diverge structurally over millions of arithmetic ops, they won't be exactly 1e-8.
# However, if the underlying gauge force generated is entirely broken, it will deviate hugely immediately.
TOLERANCE = 1e-4

if plaq_diff > TOLERANCE or wilson_diff > TOLERANCE:
    print(f"\n[FAIL] Observables diverged beyond safety threshold ({TOLERANCE})!")
    sys.exit(1)

print("\n[SUCCESS] Native Apple Metal GPU physics evaluation precisely matches standard CPU generation.")
