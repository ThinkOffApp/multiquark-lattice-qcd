#
# su2_hb_gpu.py — ctypes bridge to the Metal SU(2) heatbath generator
# (libcwsu2gen.dylib built from su2gen_gpu.swift).
#
# Packs gpt SU(2) gauge lattices into the quaternion float4/link layout the
# GPU kernel uses (site = x + L0*(y + L1*(z + L2*t)), link index site*4+mu,
# quat (a0,a1,a2,a3) with U = [[a0+i a3, a2+i a1], [-a2+i a1, a0-i a3]]),
# runs sweeps on the GPU, and unpacks back into the gpt lattices.
#
import ctypes
import os

import numpy as np

import gpt as g

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        path = os.environ.get(
            "SU2GEN_DYLIB",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "libcwsu2gen.dylib"),
        )
        L = ctypes.CDLL(path)
        L.su2gen_init.argtypes = [ctypes.c_int32] * 4 + [ctypes.c_uint64]
        L.su2gen_init.restype = ctypes.c_int32
        L.su2gen_load.argtypes = [ctypes.POINTER(ctypes.c_float)]
        L.su2gen_load.restype = ctypes.c_int32
        L.su2gen_store.argtypes = [ctypes.POINTER(ctypes.c_float)]
        L.su2gen_store.restype = ctypes.c_int32
        L.su2gen_sweep.argtypes = [ctypes.c_int32, ctypes.c_float, ctypes.c_uint32]
        L.su2gen_sweep.restype = ctypes.c_int32
        L.su2gen_plaquette.argtypes = []
        L.su2gen_plaquette.restype = ctypes.c_double
        L.su2gen_staple_dump.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_float)]
        L.su2gen_staple_dump.restype = ctypes.c_int32
        L.su2gen_set_counter.argtypes = [ctypes.c_uint64]
        L.su2gen_set_counter.restype = ctypes.c_int32
        L.su2gen_get_counter.argtypes = []
        L.su2gen_get_counter.restype = ctypes.c_uint64
        L.su2gen_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int32]
        L.su2gen_device_name.restype = ctypes.c_int32
        _LIB = L
    return _LIB


def mat_to_quat(arr):
    """(N,2,2) complex -> (N,4) float32 quaternion."""
    q = np.empty((arr.shape[0], 4), dtype=np.float32)
    q[:, 0] = arr[:, 0, 0].real
    q[:, 3] = arr[:, 0, 0].imag
    q[:, 2] = arr[:, 0, 1].real
    q[:, 1] = arr[:, 0, 1].imag
    return q


def quat_to_mat(q, dtype):
    """(N,4) float -> (N,2,2) complex."""
    arr = np.empty((q.shape[0], 2, 2), dtype=dtype)
    arr[:, 0, 0] = q[:, 0] + 1j * q[:, 3]
    arr[:, 0, 1] = q[:, 2] + 1j * q[:, 1]
    arr[:, 1, 0] = -q[:, 2] + 1j * q[:, 1]
    arr[:, 1, 1] = q[:, 0] - 1j * q[:, 3]
    return arr


class su2_gpu_generator:
    def __init__(self, U, seed, sweep_counter=0):
        grid = U[0].grid
        self.L = [int(x) for x in grid.fdimensions]
        assert len(self.L) == 4 and len(U) == 4
        self.nsites = int(np.prod(self.L))
        self.coords = g.coordinates(U[0].grid)  # keep native dtype for lattice indexing
        c = np.asarray(self.coords, dtype=np.int64)
        self.idx = (
            c[:, 0]
            + self.L[0] * (c[:, 1] + self.L[1] * (c[:, 2] + self.L[2] * c[:, 3]))
        ).astype(np.int64)
        self.dtype = U[0][:].dtype  # complex64 (single) or complex128 (double)
        lib = _lib()
        r = lib.su2gen_init(*[ctypes.c_int32(x) for x in self.L], ctypes.c_uint64(int(seed) & (2**64 - 1)))
        if r != 0:
            raise RuntimeError("su2gen_init failed (no Metal device?)")
        if sweep_counter:
            lib.su2gen_set_counter(ctypes.c_uint64(int(sweep_counter)))
        buf = ctypes.create_string_buffer(256)
        lib.su2gen_device_name(buf, 256)
        self.device_name = buf.value.decode()
        self.buf = np.empty((self.nsites, 4, 4), dtype=np.float32)

    def _cptr(self):
        return self.buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def pack(self, U):
        for mu in range(4):
            arr = U[mu][self.coords]
            arr = np.ascontiguousarray(arr.reshape(self.nsites, 2, 2))
            self.buf[self.idx, mu, :] = mat_to_quat(arr)
        _lib().su2gen_load(self._cptr())

    def unpack(self, U):
        _lib().su2gen_store(self._cptr())
        for mu in range(4):
            q = self.buf[self.idx, mu, :]
            U[mu][self.coords] = np.ascontiguousarray(quat_to_mat(q, self.dtype))

    def sweep_gpu_only(self, n, beta, mu_dirs=(0, 1, 2, 3)):
        mask = 0
        for mu in mu_dirs:
            mask |= 1 << int(mu)
        r = _lib().su2gen_sweep(ctypes.c_int32(int(n)), ctypes.c_float(float(beta)), ctypes.c_uint32(mask))
        if r < 0:
            raise RuntimeError("su2gen_sweep failed")

    def sweep(self, U, n, beta, mu_dirs=(0, 1, 2, 3)):
        """pack -> n GPU sweeps -> unpack (always leaves gpt U current)."""
        self.pack(U)
        self.sweep_gpu_only(n, beta, mu_dirs)
        self.unpack(U)

    def plaquette(self):
        return float(_lib().su2gen_plaquette())

    def staple_dump(self, mu):
        out = np.empty((self.nsites, 4), dtype=np.float32)
        _lib().su2gen_staple_dump(
            ctypes.c_uint32(int(mu)), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return out

    def sweep_counter(self):
        return int(_lib().su2gen_get_counter())
