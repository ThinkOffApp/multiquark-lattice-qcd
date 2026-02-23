#include <metal_stdlib>
using namespace metal;

// In Grid, SiteSpinor and SiteHalfSpinor are heavily templated.
// For Metal, we define the memory layout for SU(Nc) Nc=3, Nd=4.
// Spinor has 4 spin components, each with 3 color components.
// Each component is a complex number (2 floats or 2 doubles).
// For simplicity in Phase 2, we assume single precision floats (or we can use macros for precision).

// Complex number structure
struct Complex {
    float real;
    float imag;
};

// HalfSpinor: 2 spin components * 3 colors = 6 Complex numbers
struct SiteHalfSpinor {
    Complex data[6];
};

// Spinor: 4 spin components * 3 colors = 12 Complex numbers
struct SiteSpinor {
    Complex data[12];
};

// SU(3) Matrix: 3x3 Complex numbers
struct SU3Matrix {
    Complex data[9];
};

// Gauge field link has 4 directions * SU(3) Matrix per site
// Wait, DoubledGaugeField stores U[site][dir] in a specific layout.
// StencilEntry contains offsets and permutation flags.
struct StencilEntry {
    uint32_t offset;
    uint32_t is_local;
    uint32_t permute;
};

// Kernel to execute the Wilson Dslash
kernel void GenericDhopSite(
    device const SiteSpinor* in_spinor [[buffer(0)]],
    device SiteSpinor* out_spinor [[buffer(1)]],
    device const SU3Matrix* gauge_field [[buffer(2)]],
    device const StencilEntry* stencil [[buffer(3)]],
    constant uint32_t& Ls [[buffer(4)]],
    constant uint32_t& Nsite [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= Nsite * Ls) return;

    // TODO: Implement the 8-way stencil hops mapping to spProj, multLink, Recon
    // For now, this serves as the foundational shader compile target.
    
    // out_spinor[id] = in_spinor[id];   // Basic passthrough for testing pipeline
}
