#!/bin/bash

# Ensure we are in the Grid build directory or use the right path
BENCHMARK="./benchmarks/Benchmark_wilson --grid 8.8.8.8"

echo "======================================"
echo "    Metal Threadgroup Autotuning      "
echo "======================================"

for TGS in 32 64 128 256 512 1024 2048; do
    echo "======================================"
    echo "Testing Threadgroup Occupancy: $TGS"
    echo "======================================"
    
    # Run the benchmark and capture the mflop/s and memory bandwidth
    export GRID_METAL_THREADGROUP=$TGS
    $BENCHMARK | grep -E "mflop/s|GiB/s"
    
    echo ""
done

echo "Autotuning Sweep Complete."
