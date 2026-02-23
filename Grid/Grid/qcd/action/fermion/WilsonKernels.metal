#include <metal_stdlib>
using namespace metal;

struct StencilEntry {
    uint offset;
    uint is_local;
    uint permute;
    uint around_the_world; // Grid adds a fourth 32-bit int padding
};

// Target OS is macOS (M-series), Grid utilizes NEON SIMD (Nsimd=2 for float complex).
// 1 vComplexF = float4(lane0_real, lane0_imag, lane1_real, lane1_imag).
struct vComplexF { float4 v; };

struct SiteHalfSpinor { float4 data[6]; };
struct SiteSpinor { float4 data[12]; };
struct SU3Matrix { float4 data[9]; };

// SIMD Algebra Math
inline float4 timesI(float4 a) { return float4(-a.y, a.x, -a.w, a.z); }
inline float4 timesMinusI(float4 a) { return float4(a.y, -a.x, a.w, -a.z); }
inline float4 multComplex(float4 a, float4 b) {
    float4 r;
    r.x = a.x*b.x - a.y*b.y; r.y = a.x*b.y + a.y*b.x;
    r.z = a.z*b.z - a.w*b.w; r.w = a.z*b.w + a.w*b.z;
    return r;
}
inline float4 permute_lanes(float4 a) { return a.zwxy; }

// SU(3) multiplies a 6-component Half Spinor
inline SiteHalfSpinor multLink(SU3Matrix U, SiteHalfSpinor chi) {
    SiteHalfSpinor res;
    for(int s=0; s<2; s++) {
        for(int c=0; c<3; c++) {
            float4 sum = float4(0.0f);
            for(int k=0; k<3; k++) {
                sum += multComplex(U.data[c*3 + k], chi.data[s*3 + k]);
            }
            res.data[s*3 + c] = sum;
        }
    }
    return res;
}

// Xp projector (1 - gamma_x)
inline SiteHalfSpinor spProjXp(SiteSpinor fspin, uint perm) {
    SiteHalfSpinor hspin;
    for(int c=0; c<3; ++c) {
        hspin.data[0*3+c] = fspin.data[0*3+c] + timesI(fspin.data[3*3+c]);
        hspin.data[1*3+c] = fspin.data[1*3+c] + timesI(fspin.data[2*3+c]);
        if(perm) { hspin.data[0*3+c] = permute_lanes(hspin.data[0*3+c]); hspin.data[1*3+c] = permute_lanes(hspin.data[1*3+c]); }
    }
    return hspin;
}
inline SiteHalfSpinor spProjXm(SiteSpinor fspin, uint perm) {
    SiteHalfSpinor hspin;
    for(int c=0; c<3; ++c) {
        hspin.data[0*3+c] = fspin.data[0*3+c] - timesI(fspin.data[3*3+c]);
        hspin.data[1*3+c] = fspin.data[1*3+c] - timesI(fspin.data[2*3+c]);
        if(perm) { hspin.data[0*3+c] = permute_lanes(hspin.data[0*3+c]); hspin.data[1*3+c] = permute_lanes(hspin.data[1*3+c]); }
    }
    return hspin;
}
inline SiteHalfSpinor spProjYp(SiteSpinor fspin, uint perm) {
    SiteHalfSpinor hspin;
    for(int c=0; c<3; ++c) {
        hspin.data[0*3+c] = fspin.data[0*3+c] - fspin.data[3*3+c];
        hspin.data[1*3+c] = fspin.data[1*3+c] + fspin.data[2*3+c];
        if(perm) { hspin.data[0*3+c] = permute_lanes(hspin.data[0*3+c]); hspin.data[1*3+c] = permute_lanes(hspin.data[1*3+c]); }
    }
    return hspin;
}
inline SiteHalfSpinor spProjYm(SiteSpinor fspin, uint perm) {
    SiteHalfSpinor hspin;
    for(int c=0; c<3; ++c) {
        hspin.data[0*3+c] = fspin.data[0*3+c] + fspin.data[3*3+c];
        hspin.data[1*3+c] = fspin.data[1*3+c] - fspin.data[2*3+c];
        if(perm) { hspin.data[0*3+c] = permute_lanes(hspin.data[0*3+c]); hspin.data[1*3+c] = permute_lanes(hspin.data[1*3+c]); }
    }
    return hspin;
}
inline SiteHalfSpinor spProjZp(SiteSpinor fspin, uint perm) {
    SiteHalfSpinor hspin;
    for(int c=0; c<3; ++c) {
        hspin.data[0*3+c] = fspin.data[0*3+c] + timesI(fspin.data[2*3+c]);
        hspin.data[1*3+c] = fspin.data[1*3+c] - timesI(fspin.data[3*3+c]);
        if(perm) { hspin.data[0*3+c] = permute_lanes(hspin.data[0*3+c]); hspin.data[1*3+c] = permute_lanes(hspin.data[1*3+c]); }
    }
    return hspin;
}
inline SiteHalfSpinor spProjZm(SiteSpinor fspin, uint perm) {
    SiteHalfSpinor hspin;
    for(int c=0; c<3; ++c) {
        hspin.data[0*3+c] = fspin.data[0*3+c] - timesI(fspin.data[2*3+c]);
        hspin.data[1*3+c] = fspin.data[1*3+c] + timesI(fspin.data[3*3+c]);
        if(perm) { hspin.data[0*3+c] = permute_lanes(hspin.data[0*3+c]); hspin.data[1*3+c] = permute_lanes(hspin.data[1*3+c]); }
    }
    return hspin;
}
inline SiteHalfSpinor spProjTp(SiteSpinor fspin, uint perm) {
    SiteHalfSpinor hspin;
    for(int c=0; c<3; ++c) {
        hspin.data[0*3+c] = fspin.data[0*3+c] + fspin.data[2*3+c];
        hspin.data[1*3+c] = fspin.data[1*3+c] + fspin.data[3*3+c];
        if(perm) { hspin.data[0*3+c] = permute_lanes(hspin.data[0*3+c]); hspin.data[1*3+c] = permute_lanes(hspin.data[1*3+c]); }
    }
    return hspin;
}
inline SiteHalfSpinor spProjTm(SiteSpinor fspin, uint perm) {
    SiteHalfSpinor hspin;
    for(int c=0; c<3; ++c) {
        hspin.data[0*3+c] = fspin.data[0*3+c] - fspin.data[2*3+c];
        hspin.data[1*3+c] = fspin.data[1*3+c] - fspin.data[3*3+c];
        if(perm) { hspin.data[0*3+c] = permute_lanes(hspin.data[0*3+c]); hspin.data[1*3+c] = permute_lanes(hspin.data[1*3+c]); }
    }
    return hspin;
}

// Reconstructors
inline void spReconXp(thread SiteSpinor& out, SiteHalfSpinor hspin) {
    for(int c=0; c<3; ++c) {
        out.data[0*3+c] = hspin.data[0*3+c];
        out.data[1*3+c] = hspin.data[1*3+c];
        out.data[2*3+c] = timesMinusI(hspin.data[1*3+c]);
        out.data[3*3+c] = timesMinusI(hspin.data[0*3+c]);
    }
}
inline void accumReconXp(thread SiteSpinor& out, SiteHalfSpinor hspin) {
    for(int c=0; c<3; ++c) {
        out.data[0*3+c] += hspin.data[0*3+c];
        out.data[1*3+c] += hspin.data[1*3+c];
        out.data[2*3+c] -= timesI(hspin.data[1*3+c]);
        out.data[3*3+c] -= timesI(hspin.data[0*3+c]);
    }
}
inline void accumReconYp(thread SiteSpinor& out, SiteHalfSpinor hspin) {
    for(int c=0; c<3; ++c) {
        out.data[0*3+c] += hspin.data[0*3+c];
        out.data[1*3+c] += hspin.data[1*3+c];
        out.data[2*3+c] += hspin.data[1*3+c];
        out.data[3*3+c] -= hspin.data[0*3+c];
    }
}
inline void accumReconZp(thread SiteSpinor& out, SiteHalfSpinor hspin) {
    for(int c=0; c<3; ++c) {
        out.data[0*3+c] += hspin.data[0*3+c];
        out.data[1*3+c] += hspin.data[1*3+c];
        out.data[2*3+c] -= timesI(hspin.data[0*3+c]);
        out.data[3*3+c] += timesI(hspin.data[1*3+c]);
    }
}
inline void accumReconTp(thread SiteSpinor& out, SiteHalfSpinor hspin) {
    for(int c=0; c<3; ++c) {
        out.data[0*3+c] += hspin.data[0*3+c];
        out.data[1*3+c] += hspin.data[1*3+c];
        out.data[2*3+c] += hspin.data[0*3+c];
        out.data[3*3+c] += hspin.data[1*3+c];
    }
}
inline void accumReconXm(thread SiteSpinor& out, SiteHalfSpinor hspin) {
    for(int c=0; c<3; ++c) {
        out.data[0*3+c] += hspin.data[0*3+c];
        out.data[1*3+c] += hspin.data[1*3+c];
        out.data[2*3+c] += timesI(hspin.data[1*3+c]);
        out.data[3*3+c] += timesI(hspin.data[0*3+c]);
    }
}
inline void accumReconYm(thread SiteSpinor& out, SiteHalfSpinor hspin) {
    for(int c=0; c<3; ++c) {
        out.data[0*3+c] += hspin.data[0*3+c];
        out.data[1*3+c] += hspin.data[1*3+c];
        out.data[2*3+c] -= hspin.data[1*3+c];
        out.data[3*3+c] += hspin.data[0*3+c];
    }
}
inline void accumReconZm(thread SiteSpinor& out, SiteHalfSpinor hspin) {
    for(int c=0; c<3; ++c) {
        out.data[0*3+c] += hspin.data[0*3+c];
        out.data[1*3+c] += hspin.data[1*3+c];
        out.data[2*3+c] += timesI(hspin.data[0*3+c]);
        out.data[3*3+c] -= timesI(hspin.data[1*3+c]);
    }
}
inline void accumReconTm(thread SiteSpinor& out, SiteHalfSpinor hspin) {
    for(int c=0; c<3; ++c) {
        out.data[0*3+c] += hspin.data[0*3+c];
        out.data[1*3+c] += hspin.data[1*3+c];
        out.data[2*3+c] -= hspin.data[0*3+c];
        out.data[3*3+c] -= hspin.data[1*3+c];
    }
}

// Kernel to execute the Wilson Dslash
kernel void GenericDhopSite(
    device const SiteSpinor* in_spinor [[buffer(0)]],
    device SiteSpinor* out_spinor [[buffer(1)]],
    device const SU3Matrix* gauge_field [[buffer(2)]],
    device const StencilEntry* stencil [[buffer(3)]],
    constant uint32_t& Ls [[buffer(4)]],
    constant uint32_t& Nsite [[buffer(5)]],
    device const SiteHalfSpinor* buf [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= Nsite * Ls) return;

    uint sF = id; // Spinor site index
    uint sU = id / Ls; // Gauge field site index (if Ls=1 this is the same)
    
    SiteSpinor result;
    for(int i=0; i<12; i++) result.data[i] = float4(0.0f);
    
    // 8-Way Stencil Execution (Xp, Yp, Zp, Tp, Xm, Ym, Zm, Tm)
    // Dir = 0 (Xp)
    {
        StencilEntry SE = stencil[0 * Nsite + sU];
        SiteHalfSpinor hs = spProjXp(in_spinor[SE.offset], SE.permute);
        SU3Matrix U = gauge_field[sU * 8 + 0]; // +0 for Xp (Grid layout: U[Xp, Yp, Zp, Tp, Xm, Ym, Zm, Tm])
        SiteHalfSpinor chi = multLink(U, hs);
        spReconXp(result, chi);
    }
    // Dir = 1 (Yp)
    {
        StencilEntry SE = stencil[1 * Nsite + sU];
        SiteHalfSpinor hs = spProjYp(in_spinor[SE.offset], SE.permute);
        SU3Matrix U = gauge_field[sU * 8 + 1];
        SiteHalfSpinor chi = multLink(U, hs);
        accumReconYp(result, chi);
    }
    // Dir = 2 (Zp)
    {
        StencilEntry SE = stencil[2 * Nsite + sU];
        SiteHalfSpinor hs = spProjZp(in_spinor[SE.offset], SE.permute);
        SU3Matrix U = gauge_field[sU * 8 + 2];
        SiteHalfSpinor chi = multLink(U, hs);
        accumReconZp(result, chi);
    }
    // Dir = 3 (Tp)
    {
        StencilEntry SE = stencil[3 * Nsite + sU];
        SiteHalfSpinor hs = spProjTp(in_spinor[SE.offset], SE.permute);
        SU3Matrix U = gauge_field[sU * 8 + 3];
        SiteHalfSpinor chi = multLink(U, hs);
        accumReconTp(result, chi);
    }
    // Dir = 4 (Xm)
    {
        StencilEntry SE = stencil[4 * Nsite + sU];
        SiteHalfSpinor hs = spProjXm(in_spinor[SE.offset], SE.permute);
        SU3Matrix U = gauge_field[sU * 8 + 4];
        SiteHalfSpinor chi = multLink(U, hs);
        accumReconXm(result, chi);
    }
    // Dir = 5 (Ym)
    {
        StencilEntry SE = stencil[5 * Nsite + sU];
        SiteHalfSpinor hs = spProjYm(in_spinor[SE.offset], SE.permute);
        SU3Matrix U = gauge_field[sU * 8 + 5];
        SiteHalfSpinor chi = multLink(U, hs);
        accumReconYm(result, chi);
    }
    // Dir = 6 (Zm)
    {
        StencilEntry SE = stencil[6 * Nsite + sU];
        SiteHalfSpinor hs = spProjZm(in_spinor[SE.offset], SE.permute);
        SU3Matrix U = gauge_field[sU * 8 + 6];
        SiteHalfSpinor chi = multLink(U, hs);
        accumReconZm(result, chi);
    }
    // Dir = 7 (Tm)
    {
        StencilEntry SE = stencil[7 * Nsite + sU];
        SiteHalfSpinor hs = spProjTm(in_spinor[SE.offset], SE.permute);
        SU3Matrix U = gauge_field[sU * 8 + 7];
        SiteHalfSpinor chi = multLink(U, hs);
        accumReconTm(result, chi);
    }

    out_spinor[sF] = result;
}
