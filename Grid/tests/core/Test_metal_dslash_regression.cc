/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: ./tests/core/Test_metal_dslash_regression.cc

    Copyright (C) 2026

Author: claudemm (Claude Opus 4.7) for ThinkOffApp/multiquark-lattice-qcd

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
/*  END LEGAL */

// Regression test for the Metal Wilson Dslash path (multiquark-lattice-qcd
// Apple-Silicon backend). Lands as a gate before the dispatch / Dag / Int / Ext
// kernel work so that any future change to WilsonKernels.metal or the
// KERNEL_CALLNB host-side dispatcher fails loudly rather than silently
// regressing solver correctness.
//
// Coverage:
//   A. Hermiticity:   <phi | D    | chi> == conj(<chi | D^dag | phi>)
//   B. Deo + Doe = D self-consistency on the unprec operator.
//   C. Determinism:   bitwise-equal result on two identical Dhop calls.
//   D. Float vs Double: WilsonImplF vs WilsonImplD agree to single-precision
//      tolerance on the same gauge + source. This catches the case where the
//      Metal kernel (hardcoded float4 / vComplexF) silently reinterprets a
//      double-precision spinor / gauge buffer.
//
// All tests run a 4^4 lattice with a hot random gauge for A/B/C and a fixed
// pRNG seed so the test is deterministic across runs. Tolerances:
//   single precision: 1e-5 absolute on per-site differences and norms
//   double precision: 1e-10 absolute (ample headroom for round-trip ops)
//   float-vs-double cross-check: 5e-5 (one ULP-ish at single precision over
//   the full Dslash stencil reduction)

#include <Grid/Grid.h>

using namespace Grid;

namespace {

constexpr double kSingleTol = 1e-5;
constexpr double kDoubleTol = 1e-10;
constexpr double kCrossPrecTol = 5e-5;

int g_failures = 0;

void Report(const std::string &name, bool ok, double measured, double tol)
{
  std::cout << GridLogMessage
            << (ok ? "[PASS] " : "[FAIL] ")
            << name
            << " measured=" << measured
            << " tol=" << tol
            << std::endl;
  if (!ok) g_failures++;
}

template <class Impl>
void TestHermiticity(GridCartesian &Grid_, GridRedBlackCartesian &RBGrid_,
                     GridParallelRNG &pRNG, double tol, const std::string &tag)
{
  using FermionField = typename Impl::FermionField;
  using GaugeField = typename Impl::GaugeField;

  GaugeField Umu(&Grid_);
  SU<Nc>::HotConfiguration(pRNG, Umu);

  FermionField phi(&Grid_); random(pRNG, phi);
  FermionField chi(&Grid_); random(pRNG, chi);
  FermionField Dchi(&Grid_); Dchi = Zero();
  FermionField Ddagphi(&Grid_); Ddagphi = Zero();

  RealD mass = 0.1;
  WilsonFermion<Impl> Dw(Umu, Grid_, RBGrid_, mass);

  Dw.Dhop(chi, Dchi, DaggerNo);
  Dw.Dhop(phi, Ddagphi, DaggerYes);

  ComplexD lhs = innerProduct(phi, Dchi);
  ComplexD rhs = innerProduct(chi, Ddagphi);
  double diff = std::abs(lhs - std::conj(rhs));

  Report("Hermiticity (" + tag + "): |<phi|D|chi> - conj(<chi|Ddag|phi>)|",
         diff < tol, diff, tol);
}

template <class Impl>
void TestEvenOddConsistency(GridCartesian &Grid_, GridRedBlackCartesian &RBGrid_,
                            GridParallelRNG &pRNG, double tol, const std::string &tag)
{
  using FermionField = typename Impl::FermionField;
  using GaugeField = typename Impl::GaugeField;

  GaugeField Umu(&Grid_);
  SU<Nc>::HotConfiguration(pRNG, Umu);

  FermionField src(&Grid_); random(pRNG, src);
  FermionField ref(&Grid_); ref = Zero();
  FermionField r_eo(&Grid_); r_eo = Zero();

  FermionField src_e(&RBGrid_), src_o(&RBGrid_);
  FermionField r_e(&RBGrid_), r_o(&RBGrid_);

  pickCheckerboard(Even, src_e, src);
  pickCheckerboard(Odd, src_o, src);

  RealD mass = 0.1;
  WilsonFermion<Impl> Dw(Umu, Grid_, RBGrid_, mass);

  Dw.Meooe(src_e, r_o);
  Dw.Meooe(src_o, r_e);
  Dw.Dhop(src, ref, DaggerNo);

  setCheckerboard(r_eo, r_o);
  setCheckerboard(r_eo, r_e);

  FermionField err(&Grid_);
  err = ref - r_eo;
  double n = std::sqrt(norm2(err));

  Report("Deo+Doe=D consistency (" + tag + "): ||D - (Meo+Moe)||",
         n < tol, n, tol);
}

template <class Impl>
void TestDeterminism(GridCartesian &Grid_, GridRedBlackCartesian &RBGrid_,
                     GridParallelRNG &pRNG, const std::string &tag)
{
  using FermionField = typename Impl::FermionField;
  using GaugeField = typename Impl::GaugeField;

  GaugeField Umu(&Grid_);
  SU<Nc>::HotConfiguration(pRNG, Umu);

  FermionField src(&Grid_); random(pRNG, src);
  FermionField r1(&Grid_); r1 = Zero();
  FermionField r2(&Grid_); r2 = Zero();

  RealD mass = 0.1;
  WilsonFermion<Impl> Dw(Umu, Grid_, RBGrid_, mass);

  Dw.Dhop(src, r1, DaggerNo);
  Dw.Dhop(src, r2, DaggerNo);

  FermionField diff(&Grid_);
  diff = r1 - r2;
  double n = std::sqrt(norm2(diff));

  // Determinism: two identical Dhop calls must produce bitwise-equal output.
  // Anything above 0 indicates a race in acceleratorMetalBufferMap or the
  // command-buffer pipeline. We allow exactly 0 (not a tolerance).
  Report("Determinism (" + tag + "): ||Dhop(src) - Dhop(src)||",
         n == 0.0, n, 0.0);
}

void TestFloatVsDouble(GridCartesian &Grid_F, GridRedBlackCartesian &RBGrid_F,
                       GridCartesian &Grid_D, GridRedBlackCartesian &RBGrid_D,
                       GridParallelRNG &pRNG_D)
{
  // Build a hot gauge in double precision on the D-shaped grid, project to
  // single precision on the F-shaped grid, run Dhop on each, and compare on
  // the F grid. precisionChange handles the SIMD-layout difference between
  // the two grids (different Nsimd for vComplexF vs vComplexD on NEON).
  // If the Metal kernel silently reinterprets a vComplexD buffer as float4,
  // the double-precision result will be wildly off and this assertion fires.

  LatticeGaugeFieldD UmuD(&Grid_D);
  SU<Nc>::HotConfiguration(pRNG_D, UmuD);

  LatticeGaugeFieldF UmuF(&Grid_F);
  precisionChange(UmuF, UmuD);

  LatticeFermionD srcD(&Grid_D); random(pRNG_D, srcD);
  LatticeFermionF srcF(&Grid_F);
  precisionChange(srcF, srcD);

  LatticeFermionD outD(&Grid_D); outD = Zero();
  LatticeFermionF outF(&Grid_F); outF = Zero();

  WilsonFermionD DwD(UmuD, Grid_D, RBGrid_D, 0.1);
  WilsonFermionF DwF(UmuF, Grid_F, RBGrid_F, 0.1);

  DwD.Dhop(srcD, outD, DaggerNo);
  DwF.Dhop(srcF, outF, DaggerNo);

  LatticeFermionF outDinF(&Grid_F);
  precisionChange(outDinF, outD);

  LatticeFermionF diff(&Grid_F);
  diff = outF - outDinF;
  double n = std::sqrt(norm2(diff)) / std::max(1.0, std::sqrt(norm2(outF)));

  Report("Float vs Double cross-check: relative ||D_F(src) - D_D(src)||",
         n < kCrossPrecTol, n, kCrossPrecTol);
}

}  // namespace

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  // Force a 4^4 lattice for fast deterministic regression. Override-able via
  // --grid X.Y.Z.T on the command line if a larger sweep is wanted.
  Coordinate latt_size = GridDefaultLatt();
  Coordinate mpi_layout = GridDefaultMpi();

  if (latt_size.size() < (size_t)Nd) {
    latt_size = Coordinate({4, 4, 4, 4});
  }

  // vComplexF and vComplexD have different Nsimd (e.g. on NEON: 2 vs 1).
  // Each Lattice<vobj> requires a Grid whose simd_layout matches its vobj's
  // Nsimd, so we build two grids and route F-typed and D-typed tests to the
  // matching one. Using a single grid with vComplex::Nsimd() (the build-time
  // default) crashes peekLocalSite with an `sizeof(sobj)*Nsimd ==
  // sizeof(vobj)` assertion the moment the wrong-shape field is touched.
  Coordinate simd_F = GridDefaultSimd(Nd, vComplexF::Nsimd());
  Coordinate simd_D = GridDefaultSimd(Nd, vComplexD::Nsimd());

  GridCartesian Grid_F(latt_size, simd_F, mpi_layout);
  GridCartesian Grid_D(latt_size, simd_D, mpi_layout);
  GridRedBlackCartesian RBGrid_F(&Grid_F);
  GridRedBlackCartesian RBGrid_D(&Grid_D);

  GridParallelRNG pRNG_F(&Grid_F);
  GridParallelRNG pRNG_D(&Grid_D);
  pRNG_F.SeedFixedIntegers(std::vector<int>({45, 12, 81, 9}));
  pRNG_D.SeedFixedIntegers(std::vector<int>({45, 12, 81, 9}));

  std::cout << GridLogMessage
            << "============================================================" << std::endl
            << "  Metal Wilson Dslash regression test ("
            << latt_size[0] << "^" << Nd << ")" << std::endl
            << "============================================================" << std::endl;

  // Single-precision suite. This is the only Impl currently exercised by the
  // Metal kernel, so most failures will surface here first.
  TestHermiticity<WilsonImplF>(Grid_F, RBGrid_F, pRNG_F, kSingleTol, "WilsonImplF");
  TestEvenOddConsistency<WilsonImplF>(Grid_F, RBGrid_F, pRNG_F, kSingleTol, "WilsonImplF");
  TestDeterminism<WilsonImplF>(Grid_F, RBGrid_F, pRNG_F, "WilsonImplF");

  // Double-precision suite. Under the current Metal backend, Codex flagged
  // that vComplexD will be misinterpreted as float4 if the kernel dispatch
  // does not gate on Impl precision. With PR #2's MetalWilsonImplOK trait
  // landed these should pass (D Impl falls through to CPU); without it they
  // will fail loudly.
  TestHermiticity<WilsonImplD>(Grid_D, RBGrid_D, pRNG_D, kDoubleTol, "WilsonImplD");
  TestEvenOddConsistency<WilsonImplD>(Grid_D, RBGrid_D, pRNG_D, kDoubleTol, "WilsonImplD");
  TestDeterminism<WilsonImplD>(Grid_D, RBGrid_D, pRNG_D, "WilsonImplD");

  TestFloatVsDouble(Grid_F, RBGrid_F, Grid_D, RBGrid_D, pRNG_D);

  std::cout << GridLogMessage
            << "============================================================" << std::endl
            << "  Result: " << (g_failures == 0 ? "ALL PASSED" : "FAILED")
            << " (" << g_failures << " failure" << (g_failures == 1 ? "" : "s") << ")" << std::endl
            << "============================================================" << std::endl;

  Grid_finalize();
  return g_failures == 0 ? 0 : 1;
}
