# 4Q Flux Measurement Operators — Square & Tetra (Design Spec)

Status: DESIGN for review (requested by Petrus 2026-07-08, Fable pass).
Extends `4q_measurement_spec.md` (P4 there) into a concrete operator design.
Gauge group **SU(2)** throughout. Program lineage: this is the 4Q flux
program of hep-lat/9804004 (squares + diagonal 2Q at β=2.4, R=2,4,6,8) with
the tetrahedron added — tetra flux was NOT measured in the published PGM
papers; it is new territory, on the T_d wedge verified 2026-07-07.

## 1. Physics goal

Map where the chromo-field rearranges when 4Q binding turns on:

    Δf(x) = f_4Q^(0)(x) − [ f_2Q^pair1(x) + f_2Q^pair2(x) ]

indexed by ABSOLUTE lattice position x — NOT each geometry's own
center-relative offset. A 2Q pair midpoint is displaced from the 4Q center
(e.g. a square side's midpoint sits d/2 off the square center), so equal
center-relative r would subtract fields at different physical points. Each
geometry's reduced wedge is unfolded to the common absolute frame and each
2Q pair field placed at its own pair center (§4) before subtraction. With
that alignment all three terms are on the SAME configurations so the vacuum
and much of the statistical noise cancel site-by-site. Ground state AND first
excited state (the flip-flop between pairings is visible in f^(1) − f^(0)
— both come free from the same raw data, §3). 9804004 measured exactly the
required reference set: on-axis 2Q, diagonal 2Q at √2·R, and the square —
consistent with the production decision that off-axis 2Q = diagonals only.

## 2. The operator

### 2.1 Flux correlator matrix

For each 4Q geometry (pairing basis i,j ∈ {A,B}, complete for SU(2)) and
probe offset r from the geometry center, plaquette plane class P:

    F_ij,P(r; T) = ⟨ Tr W_ij(x₀; T) · □_P(x₀ + c + r + (T/2)·t̂) ⟩_{x₀}

- W_ij(x₀; T): the 4Q correlation-matrix element as a per-site trace FIELD
  (base point x₀), exactly the P2/P3 objects before volume averaging.
- □_P: bare (UNSMEARED-field) plaquette field of plane class P — probe on
  the unsmeared links, source on the smeared links (PR #12 discipline).
- c: offset from base point to the geometry center (fixed per geometry).
- Probe timeslice: T/2 above the source's t=0 plane → T even for flux
  (T_flux = 4 default; the potential T-scan is unaffected).
- ⟨·⟩_{x₀}: full translation average — implemented as one chained cshift of
  the probe field by (r + c + T/2·t̂) then pointwise multiply and g.sum
  (the existing 2Q flux shift+product path, workspace-buffered).

Also store per measurement: W̄_ij(T) = ⟨Tr W_ij⟩ (already produced by
P2/P3) and the vacuum ⟨□_P⟩. Raw triplet (F, W̄, □̄) goes to disk; NOTHING
is subtracted or divided at measurement time (ensemble-level ratio + single
vacuum subtraction in postprocessing — PR #13/#14/#15 discipline).

### 2.2 State projection (postprocessing only)

Solve the 2×2 GEVP W(T) v = λ W(T₀) v from the same-run potential data.
For state n with eigenvector ṽⁿ:

    f_P^(n)(r) = [ Σ_ij ṽⁿ_i ṽⁿ_j F_ij,P(r; T_flux) ]
                 / [ Σ_ij ṽⁿ_i ṽⁿ_j W̄_ij(T_flux) ]  −  ⟨□_P⟩

Contract with ṽ⁰ → ground-state flux; ṽ¹ → excited state. Keeping the full
F_ij matrix (3 independent elements: AA, BB, AB) makes the projecting
eigenvector's (T₀, T) choice a postprocessing stability check, not a
burned-in decision. Note F is stored only at T_flux, so this freedom lives
in the potential-side GEVP eigenvector; the flux sink time itself is fixed
(a genuine flux T-contamination check needs F at a second T — §10 Q3).
Consistency check for free: D₄h (square) symmetry ⇒ F_AA = F_BB within
errors after frame mapping (the 90° rotation that exchanges the two
pairings).

### 2.3 Plane classes and E/B decomposition

Store all 6 plaquette orientations, keyed RELATIVE to the source frame:
E_a (plane spanned by tdir and spatial axis a) and B_a (spatial plane with
normal a), a = 1,2,3 in the geometry's canonical frame. Postprocessing
forms the Euclidean combinations: action density s ∝ Σ_P f_P, energy
density ε ∝ Σ_E f − Σ_B f (Michael sum-rule conventions). Because
measurement iterates over 4 time directions and geometry orientations, each
measured (tdir, lattice plane) pair is mapped back to the canonical frame
by the same signed permutation that generated the orientation, APPLIED AT
MEASUREMENT so §6 records land already in the canonical frame. This is a
sign-permutation relabel + average only (no ratio, no subtraction), so it
preserves the raw-triplet discipline; it gets its own round-trip unit test
(§7).

## 3. Probe region: boxes, wedges, weights

- Probe box: bounding box of the quark positions expanded by margin m
  (default m = 3) in every direction, rectangular, in plain center-relative
  integer coordinates. Flux sizes are even d (§4) ⇒ the geometry center is a
  lattice site, so there is no half-integer center to handle. (The viewer
  uses an integer floor-center for odd d and is NOT doubled; the earlier
  "doubled coordinates" convention was both unfounded — it is not what the
  viewer does — and moot for even d, and has been dropped.)
- Reduce the box by the geometry's full stabilizer group; each kept site
  carries weight |G| / |stab(site)|. This is the 56%-overcount trap: flat
  ×|G| weighting is forbidden. As a direct consequence, the number of
  representative sites EXCEEDS box-volume / |G| — sites on symmetry walls
  have |stab| > 1 and smaller orbits, so they contribute more than their
  1/|G| share of representatives (this is why the §8 counts are well above
  volume/|G|). The reduction helper returns (site, weight) pairs only.
- **Generic stabilizer engine** (new, replaces per-geometry hand-coded
  conditions): enumerate the 48 signed permutations, keep those mapping the
  quark position set to itself (for 2Q in SU(2), quark ≡ antiquark —
  pseudoreal — so endpoint swap is a symmetry), apply to probe sites
  directly. Expected group orders, asserted as unit tests:
    on-axis 2Q: 16 · diagonal 2Q (d,d,0): 8 · square: 16 (D₄h) ·
    tetra: 24 (T_d).
  The engine reduces the RECTANGULAR probe box, a different domain from the
  hand-verified viewer wedges (different margins and cutoff shapes — the
  square viewer uses fixed hMax=5 in the perpendicular, the tetra viewer a
  spherical cutoff ~1.1·D). So correctness is checked by predicate/orbit
  equivalence on a shared domain, NOT site-set equality with the viewer;
  see §7 gate 1.

## 4. Source set (all on the same configs)

| Source | Positions | Sizes (flux) | Reduction |
|---|---|---|---|
| 2Q on-axis | 0 → (d,0,0) | d = 2,4,6,8 | 1/16 |
| 2Q diagonal | 0 → (d,d,0), symmetrized-L connector (P1) | d = 2,4,6,8 (R=√2d ≈ 2.8–11.3) | 1/8 |
| square 4Q | side d, pairings A,B | d = 2,4,6,8 | 1/16 |
| tetra 4Q | (0,0,0),(d,d,0),(d,0,d),(0,d,d) | d = 2,4 (edge √2d) | 1/24 |

Flux sizes are the PGM 9804004 set (even d), NOT the full potential scan —
potentials at every d, flux at the even subset. Tetra capped at d = 4
initially (box grows as (2d+2m+1)³; d=6 pending cost data from d=4).
Both 2Q pairings of each 4Q geometry need a same-config reference. Square:
pairing A = on-axis sides (length d → on-axis 2Q at d), pairing B = (13)(24)
diagonals (length √2·d → diagonal 2Q at d). Tetrahedron: regular, both
pairings are edges (√2·d → diagonal 2Q at d), one reference covers both.
Diagonal 2Q therefore runs to d=8 so the d=8 square's B-pairing — and the
flip-flop f^(1)−f^(0) — has its reference. Δf is formed in ABSOLUTE lattice
coordinates (§1): each geometry's reduced wedge is unfolded to the common
frame and each 2Q pair field is placed at ITS pair center
(4Q_center + pair_offset), not at matched center-relative r. A §7 gate
checks the 2Q-pair and 4Q site sets coincide after re-centering.

## 5. Implementation structure

- Flux hooks into the P2/P3 measurement pass and REUSES the W_ij trace
  fields already built per (geometry, d, orientation) — marginal cost is
  probe cshifts × wedge sites × 6 planes only. One extra complex field per
  W_ij retained at T_flux (workspace buffer, ~5 MB at 24⁴).
- 6 probe plane fields built once per (config, tdir) from the unsmeared
  field (~32 MB total at 24⁴), reused across all sources and offsets.
- All shifts through the bounded workspace chain (Ca/Cb ping-pong); offsets
  are bounded by box size, so the cshift-plan cache stays bounded.
- **Hard prerequisite: the 24⁴ device-memory census** (--device-mem 32768
  vs 128 MB default). Flux adds working set on top of off-axis; it does NOT
  enter production until the census gate passes and the flag is standard.
- Same multilevel/multihit copies structure: F, W̄, □̄ accumulated per hit
  on the same field state, then averaged — the linearity argument that
  carries W_ij through hits/blocks applies to F unchanged.

## 6. Output schema (per measurement record)

```
"flux4q": {
  "Gsq_d4": {
    "T": 4, "probe_t": 2, "margin": 3,
    "sites":  [[rx, ry, rz, weight], ...],         # plain center-relative (even-d integer center)
    "planes": ["E1","E2","E3","B1","B2","B3"],     # canonical frame
    "wp": {"AA": [[site][plane]], "BB": ..., "AB": ...},   # ⟨Tr W · □⟩ raw
    "w":  {"AA": re, "BB": re, "AB": re},                  # ⟨Tr W⟩
    "p":  [6 vacuum ⟨□_P⟩]
  },
  "G2q_d4": { ... same shape, single "11" key instead of pairings ... },
  "G2qdiag_d4": { ... }
}
```
Arrays, not per-site JSON keys (a d=4 square is 84 wedge sites × 6 planes
× 3 matrix elements ≈ 1.5k floats/record — fine as arrays, hostile as keys).

## 7. Validation gates (in order, each blocks the next)

1. **Stabilizer engine unit tests.** (The earlier "reproduce the viewer
   wedges by site-set equality" demand was impossible — the viewer and the
   probe box are different domains — and is replaced by:)
   - (a) **group orders** 16 / 8 / 16 / 24 for on-axis 2Q / diagonal 2Q /
     square / tetra.
   - (b) **selector equivalence** on ONE common G-invariant domain (a union
     of full orbits): the engine's fundamental-domain membership selects the
     same representatives as the viewer predicates `sqSymLevel` / `tetSymLevel`
     (square/tetra) and `q2SymLevel` (on-axis 2Q). Diagonal 2Q has NO viewer
     predicate → check it against a from-scratch brute-force orbit
     enumeration instead.
   - (c) **orbit-cover partition** — this, not a Σweights checksum, is what
     tests the WEIGHTS: expand every kept (site, weight = |G|/|stab|) by its
     orbit under G and assert each box site is covered EXACTLY once. A
     Σweights == box-volume checksum passes under compensating |stab| errors;
     a partition test does not.
   - (d) the common test domain MUST contain wall sites of every stabilizer
     class (dx=0, dx=dy, h=0, and the four tetra walls) so |stab| > 1 weights
     are actually exercised. Flux is even-d (center on a lattice site, axis
     walls populated); the engine's odd-d path (potentials, out of scope
     here) has no valid viewer reference — the viewer is floor-centered and
     wrong for odd d — so it must be gated against an independent brute-force
     orbit reference, not the viewer.
   - (e) **frame-map round-trip** (§2.3).
   - (f) **re-centering coincidence** (§1/§4): after unfolding to the common
     absolute frame, the 2Q-pair site set and the 4Q site set coincide, so
     Δf is a true site-by-site subtraction.
2. **8⁴ parity**: F_ij,P per site vs an allocation-naive reference
   implementation; random-gauge-transform invariance (< 1e-11), same
   harness style as tools/test_offaxis_parity.py.
3. **Continuity gate**: 2Q on-axis flux through the NEW box machinery must
   reproduce the EXISTING perpendicular-profile numbers at matching offsets
   on the same configs (this is the regression test tying old and new
   pipelines together).
4. **Physics gates** (production, 16³×32 first — 9804004's own volume):
   large-d square f^(0) → sum of two independent 2Q tube profiles;
   action sum rule Σ_r s(r) vs dV/dR (loose, statistics permitting);
   profile shapes vs 9804004 at β=2.4, R=2,4,6,8.

## 8. Cost estimate (24⁴, per config, after device-mem fix)

Representative wedge sites per source at m=3 — EXACT orbit counts (brute-force
enumeration), NOT box-volume/|G|, which undercounts by ~55–65% because wall
sites have |stab|>1 (see §3):
- square d=2/4/6/8 = 60 / 84 / 112 / 144
- tetra  d=2/4     = 55 / 91
- on-axis 2Q d=2/4/6/8 = 50 / 60 / 70 / 80
- diagonal 2Q d=2/4/6/8 = 100 / 144 / 196 / 256

Full §4 set ≈ 1500 representative sites × 6 planes × (3 matrix elements for
4Q, 1 for 2Q) ≈ 15.5k shift+multiply+sum passes per tdir. The W_ij source
fields dominate and are SHARED with the potential measurement, so flux rides
on top of P2/P3 rather than adding a separate scan. Flux stays a per-run
opt-in flag (--flux4q 1) with per-source-class selection, mirroring
--offaxis.

## 9. Phasing

- **F1** — stabilizer/wedge/weights engine + unit tests (pure python, no
  lattice; lands independently).
- **F2** — box flux for 2Q on-axis + diagonal, continuity gate vs existing
  profiles. (Depends on P1 connectors; already merged machinery.)
- **F3** — square 4Q flux matrix + postprocessing contraction (ground +
  excited) + Δf maps. (Depends on P2 square W_ij fields.)
- **F4** — tetra 4Q flux. (Depends on P3; beyond published PGM.)
- Dashboard heat-map panel: separate task, fed by the §6 schema.

## 10. Open questions for Petrus

1. Flux size set: even-d PGM set (2,4,6,8 square; 2,4 tetra) as specced,
   or extend? (Potentials still cover every d.)
2. Probe margin m = 3 enough, or match a specific reach from the 90s runs?
3. T_flux = 4 with probe at T/2 = 2 (PGM convention), or also store T = 6
   for a contamination cross-check at ~2× flux cost?
4. Excited-state flux f^(1) is free in postprocessing — treat as a
   first-class deliverable (flip-flop map), or defer?
5. Tetra flux has no published PGM reference — gates rely on internal
   consistency (large-d factorization + sum rule). Comfortable, or do you
   want a 16³×32 square-flux reproduction milestone first as the anchor?
