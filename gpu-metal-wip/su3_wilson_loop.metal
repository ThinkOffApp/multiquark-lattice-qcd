#include <metal_stdlib>
using namespace metal;

// Wilson-loop path-product kernel (ladder step 3), generalizing su3_plaquette.metal.
// A rectangular R x T Wilson loop in a (mu,nu) plane is the ordered product of
// 2R+2T links around the rectangle, with the two return edges daggered:
//   W = [R links +mu] [T links +nu] [R links -mu]^dag [T links -nu]^dag
// The plaquette is the R=T=1 case (N=4). As in the plaquette kernel the link
// fields are already neighbour-aligned by a CPU Cshift, so per vSite this is
// pure local arithmetic with no stencil / no lane permute.
//
// Conventions mirror antigravity's WilsonKernels.metal exactly:
// NEON Nsimd=2, vComplexF = float4(lane0_re, lane0_im, lane1_re, lane1_im),
// SU(3) link = 9 x float4 row-major (data[row*3 + col]).
struct SU3Matrix { float4 data[9]; };

inline float4 cmul(float4 a, float4 b) {
    return float4(a.x*b.x - a.y*b.y,
                  a.x*b.y + a.y*b.x,
                  a.z*b.z - a.w*b.w,
                  a.z*b.w + a.w*b.z);
}
inline float4 cadd(float4 a, float4 b) { return a + b; }
inline float4 cconj(float4 a) { return float4(a.x, -a.y, a.z, -a.w); }

// C = A * B
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
// C = A * B^dagger    ((B^dag)_{kj} = conj(B_{jk}))
inline SU3Matrix mm_dag(SU3Matrix A, SU3Matrix B) {
    SU3Matrix C;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float4 s = float4(0.0f);
            for (int k = 0; k < 3; ++k)
                s = cadd(s, cmul(A.data[i*3+k], cconj(B.data[j*3+k])));
            C.data[i*3+j] = s;
        }
    return C;
}
inline float2 reTrace(SU3Matrix A) {
    float4 t = A.data[0] + A.data[4] + A.data[8];
    return float2(t.x, t.z);
}
// per-lane identity (1+0i on the diagonal for both SIMD lanes)
inline SU3Matrix identity() {
    SU3Matrix I;
    for (int n = 0; n < 9; ++n) I.data[n] = float4(0.0f);
    I.data[0] = float4(1.0f, 0.0f, 1.0f, 0.0f);
    I.data[4] = float4(1.0f, 0.0f, 1.0f, 0.0f);
    I.data[8] = float4(1.0f, 0.0f, 1.0f, 0.0f);
    return I;
}

// Ordered path product around an arbitrary loop. `links` holds nLink
// pre-aligned SU(3) fields packed as links[k*nVSite + id], in path order.
// `dagger` is a per-link flag (dagger[k] != 0 => link k enters as U^dag, the
// return edges). A per-link buffer, not a bitmask: a rectangular loop has
// 2R+2T links and production shapes (e.g. R=12,T=6 -> 36 links) exceed the
// 32 bits a uint mask could hold.
kernel void WilsonLoopPath(
    device const SU3Matrix* links   [[buffer(0)]],
    device float2*          outReTr [[buffer(1)]],
    constant uint32_t&      nLink   [[buffer(2)]],
    device const uchar*     dagger  [[buffer(3)]],
    constant uint32_t&      nVSite  [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= nVSite) return;
    SU3Matrix acc = identity();
    for (uint k = 0; k < nLink; ++k) {
        SU3Matrix Uk = links[k * nVSite + id];
        if (dagger[k]) acc = mm_dag(acc, Uk);
        else           acc = mm(acc, Uk);
    }
    outReTr[id] = reTrace(acc);
}
