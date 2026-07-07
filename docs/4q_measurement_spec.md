# 4Q + Off-Axis Measurement Extension — Specification

Status: ACCEPTED by Petrus 2026-07-07 (decisions in §7); technical review by
claudeMB pending on this PR. Gauge group: **SU(2)** throughout —
same as the existing production pipeline and the PGM program this extends.
Target lattice: 24⁴ at β=2.4 (dev runs on 16⁴/8⁴), on the existing
generation → measurement → live-dashboard chain.

## 1. Physics goals

1. **Off-axis 2Q potential** V(R⃗) on non-axis separations — fills the gaps
   between integer R, tests restoration of rotational invariance (lattice
   artifacts show as scatter off the Cornell curve at small R), and anchors
   the same R values the tetrahedron uses (R = d√2).
2. **4Q binding energies** B(geometry, d) = E₄Q − E₂Q,pair for
   **square** (d = 2…8) and **tetrahedron** (d = 2…6, edge R = d√2 ≈ 2.8…8.5)
   geometries — the multiquark question proper: is the 4-body ground state
   below two independent flux tubes, and by how much.
3. (Later phase) 4Q flux distributions on the symmetry-reduced wedges — where
   does the field rearrange when binding turns on.

## 2. Geometries and quark positions (lattice sites)

| Geometry | Positions | Scan | Point group | Reduction |
|---|---|---|---|---|
| off-axis 2Q | 0 → R⃗ for R⃗ classes (n,n,0), (n,n,n), (2n,n,0) | \|R\| ≤ ~8.5 | axial C₂ᵥ-type | orbit-avg |
| square | (0,0,0),(d,0,0),(0,d,0),(d,d,0) | d = 2…8 | D₄h (order 16) | 1/16 |
| tetra | (0,0,0),(d,d,0),(d,0,d),(0,d,d) | d = 2…6 | T_d (order 24) | 1/24 |

All tetra vertices are lattice sites for every integer d (verified). The
fundamental-domain conditions in `tools/geometry_viewer.html` are verified
exact (20k-orbit brute-force test, 2026-07-07) and are reused verbatim for
flux sampling.

**Weights (hard requirement):** flux/observable sums over a reduced wedge must
weight each site by `|G| / |stab(site)|` — NOT flat ×16/×24. On-wall sites are
14–50% of the wedge; flat weighting overcounts sums by ~56% at d=6. The
reduction helper must return `(site, weight)` pairs and a unit test must check
`Σ weights == full-volume count` for every geometry and d (both parities).

## 3. Operators

### 3.1 Off-axis 2Q
Wilson loop with spatial legs built as canonical staircase paths
(e.g. R⃗=(2,1,0): x,x,y — fixed lexicographic step order so the operator is
well-defined and identical across translations). Same smearing (12-step
spatial), same T set {1…6}, same V_eff plateau extraction as on-axis.
Cubic-orbit averaging: measure all distinct orientations of each R⃗ class per
time direction (e.g. (n,n,0): 12 orientations ÷ path-reversal symmetry) and
average before the T-fit — same statistical role that time-direction
averaging plays today.

### 3.2 4Q correlation matrix (PGM formalism)
Four static quarks admit two independent pairings into SU(2) singlets:
A = (Q₁Q₂)(Q₃Q₄), B = (Q₁Q₃)(Q₂Q₄). Measure the 2×2 matrix

```
W_ij(T) = ⟨ P_i(0) · [4 temporal lines] · P_j(T)† ⟩,  i,j ∈ {A,B}
```

- **Diagonal W_AA, W_BB**: products of two rectangular (or staircase) Wilson
  loops — existing loop machinery, run twice per pairing.
- **Off-diagonal W_AB**: one closed contour — spatial connectors of pairing A
  at t=0, the four temporal lines, spatial connectors of pairing B at t=T.
  New but small: the path builder already composes arbitrary link products;
  this is a single trace over a longer path.
- Extract E₄Q via the generalized eigenvalue problem on W(T)/W(T₀) (2×2 GEVP,
  closed-form for 2×2 — no numerics dependency), then the same V_eff plateau
  fit per eigenvalue. Binding: B(d) = E₄Q^(0)(d) − 2·V(R_pair) with V from the
  SAME configs (correlated errors partially cancel — jackknife over configs).

Note the square has a third pairing (diagonals) that mixes only weakly; PGM
used the 2×2 basis and so do we (documented limitation).

## 4. Estimator reuse (nothing new invented)

- 12-step spatial smearing on connectors (ground-state overlap).
- Multilevel blocks × multihit on temporal lines — identical mechanics;
  the 4Q observable is linear in each temporal line's sub-average exactly like
  2Q, so blocks×hits factorization carries over unchanged.
- Full-volume translation averaging via the existing sampler.
- Time-direction (×4) orientation averaging.
- Same live JSON/JSONL schema: loop keys extended to
  `G<geom>_d<d>_P<ij>_T<t>` (e.g. `Gsq_d4_PAB_T3`); off-axis 2Q keys
  `Rv<class>_<n>_T<t>` (e.g. `Rv110_3_T4`). Dashboard plateau/loop panels key
  off these transparently (R-selector becomes a geometry+size selector).

## 5. Implementation plan (phased, each lands runnable)

- **P1 — off-axis 2Q** (smallest diff, validates the path builder):
  staircase spatial paths + orbit averaging + dashboard keys. Gate: off-axis
  V(R) points lie on the Cornell fit through on-axis points within 2σ.
- **P2 — square 4Q**: pairing connectors, 2×2 matrix, GEVP, binding output.
  Gate: at large pair separation the ground state → 2·V(d) within errors;
  W_AB → 0 as separation grows.
- **P3 — tetra 4Q**: same machinery, tetra positions/pairings.
- **P4 — 4Q flux profiles** on the verified wedges with orbit weights.

Each phase: parity test at 8⁴ against a slow reference implementation
(direct link-product evaluation, no multilevel) before production, same
discipline as PR #24's bit-identical check.

## 6. Cost estimate (24⁴, per config)

Off-axis 2Q ≈ 1.5× current 2Q loop cost (more orientations, same T set).
4Q square/tetra: 3 matrix elements × 7 (5) sizes × 6 T × 4 tdirs ≈ 2–3× the
2Q loop budget at equal multilevel settings — well inside the current
per-measurement envelope; flux phases dominate cost and get the 16×/24×
wedge reduction. Run tiering follows the leak situation (2x1 now, 8x2 when
the reuse-pool/census-v3 or process-split lands).

## 7. Decisions (Petrus, 2026-07-07, thinkoff-development)

1. **Off-axis classes: use the exact PGM set.** P1 starts by extracting the
   off-axis separation class list from the PGM paper tables and encoding it as
   the canonical `OFF_AXIS_CLASSES` constant (with a doc pointer to the source
   table); the (n,n,0)/(n,n,n)/(2n,n,0) list in §2 is a placeholder until that
   extraction and is replaced, not merged with it.
2. **Square pairings: keep PGM's 2×2 basis.** No third (diagonal) pairing; the
   omission stays a documented limitation as in §3.2.
3. **Scope: all of it.** On-axis 2Q (existing) + off-axis 2Q + square + tetra
   are all in scope — the P1→P4 order in §5 is a landing sequence, not a
   selection.
4. **GPU-resident throughout** ("implement all the listed measurements using
   gpu fully"): every new measurement inner loop — staircase/connector path
   products, 4Q contour traces, multilevel sub-averages — runs on the GPU
   (Metal), same as the existing plaquette/Wilson-loop kernels. CPU is
   orchestration, GEVP (closed-form 2×2), fits, and I/O only. The 8⁴ parity
   gates in §5 compare GPU results against the slow CPU reference.
