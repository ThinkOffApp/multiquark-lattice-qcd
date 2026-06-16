#include <metal_stdlib>
using namespace metal;

// Mirrors antigravity's WilsonKernels.metal conventions.
// NEON Nsimd=2 for ComplexF: vComplexF = float4(lane0_re, lane0_im, lane1_re, lane1_im).
// An SU(3) link matrix = 9 complex = 9 x float4 (row-major: data[row*3 + col]).
struct SU3Matrix { float4 data[9]; };

// per-lane complex multiply on the packed 2-lane float4
inline float4 cmul(float4 a, float4 b) {
    return float4(a.x*b.x - a.y*b.y,
                  a.x*b.y + a.y*b.x,
                  a.z*b.z - a.w*b.w,
                  a.z*b.w + a.w*b.z);
}
inline float4 cadd(float4 a, float4 b) { return a + b; }
// complex conjugate per lane
inline float4 cconj(float4 a) { return float4(a.x, -a.y, a.z, -a.w); }

// C = A * B   (3x3 complex, SIMD over 2 lanes)
inline SU3Matrix mm(SU3Matrix A, SU3Matrix B) {
    SU3Matrix C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float4 s = float4(0.0f);
            for (int k = 0; k < 3; ++k)
                s = cadd(s, cmul(A.data[i*3+k], B.data[k*3+j]));
            C.data[i*3+j] = s;
        }
    return C;
}
// C = A * B^dagger
inline SU3Matrix mm_dag(SU3Matrix A, SU3Matrix B) {
    SU3Matrix C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float4 s = float4(0.0f);
            for (int k = 0; k < 3; ++k)
                // (B^dag)_{kj} = conj(B_{jk})
                s = cadd(s, cmul(A.data[i*3+k], cconj(B.data[j*3+k])));
            C.data[i*3+j] = s;
        }
    return C;
}
// Re Tr(A), summed over the 2 SIMD lanes (returns float2: {lane0, lane1})
inline float2 reTrace(SU3Matrix A) {
    float4 t = A.data[0] + A.data[4] + A.data[8]; // diagonal 00,11,22
    return float2(t.x, t.z); // real parts of the two lanes
}

// Local plaquette: inputs are already neighbour-aligned by a CPU Cshift, so
// per vSite this is pure local arithmetic (no stencil / no lane permute).
// For one (mu,nu) plane: P = Umu(x) * Unu(x+mu) * Umu(x+nu)^dag * Unu(x)^dag.
// Buffers hold the four aligned fields for this plane.
kernel void PlaquettePlane(
    device const SU3Matrix* Umu_x        [[buffer(0)]], // U_mu(x)
    device const SU3Matrix* Unu_xpmu     [[buffer(1)]], // U_nu(x+mu)  (pre-shifted)
    device const SU3Matrix* Umu_xpnu     [[buffer(2)]], // U_mu(x+nu)  (pre-shifted)
    device const SU3Matrix* Unu_x        [[buffer(3)]], // U_nu(x)
    device float2*          outReTr      [[buffer(4)]], // per-vSite Re Tr (2 lanes)
    constant uint32_t&      nVSite       [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= nVSite) return;
    SU3Matrix p = mm(Umu_x[id], Unu_xpmu[id]);   // U_mu * U_nu(x+mu)
    p = mm_dag(p, Umu_xpnu[id]);                  // * U_mu(x+nu)^dag
    p = mm_dag(p, Unu_x[id]);                     // * U_nu(x)^dag
    outReTr[id] = reTrace(p);
}
