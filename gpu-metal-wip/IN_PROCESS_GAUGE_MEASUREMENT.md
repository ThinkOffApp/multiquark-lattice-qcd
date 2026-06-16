# In-process Metal gauge measurement WIP

Checkpoint for moving pure-gauge measurement from the Swift/file-roundtrip
parity harness into Grid/C++.

## Added API

`Grid/Grid/qcd/utils/WilsonLoops.h` now has opt-in Metal methods behind
`#ifdef GRID_METAL`:

- `WilsonLoops<Gimpl>::avgPlaquetteMetal(Umu)`
- `WilsonLoops<Gimpl>::avgWilsonLoopMetal(Umu, R1, R2)`

These do not replace the default CPU methods. They are intended for parity and
timing gates first.

## Dispatch Shape

- Grid Cshift aligns physical neighbours on the host side.
- `PlaquettePlane` and `WilsonLoopPath` run the local SU(3) matrix products on
  Metal.
- ReTr output is reduced on CPU from a shared Metal buffer.
- Wilson loops pack the shifted path fields into one shared Metal buffer and
  use a per-link dagger flag buffer, avoiding the old 32-link bitmask limit.

## Current Verification

- `Grid/Grid/qcd/action/fermion/WilsonKernels.metal` compiles locally to
  `default.metallib` with the new kernels.
- `git diff --check` passes.

## Not Yet Claimed

- Full Grid/cgpt compile has not been run in this worktree because it has no
  configured Grid build.
- CPU-vs-Metal parity for the new in-process API still needs to run on the
  configured mini Metal build.
- No end-to-end speedup is claimed until that parity pass and timing are pushed.
