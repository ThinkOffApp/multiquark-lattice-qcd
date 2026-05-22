#
#    GPT - Grid Python Toolkit
#    Copyright (C) 2023  Christoph Lehner (christoph.lehner@ur.de, https://github.com/lehner/gpt)
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License along
#    with this program; if not, write to the Free Software Foundation, Inc.,
#    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
import gpt as g


class matrix_padded:
    def __init__(self, lat, points, code, code_parallel_block_size=None):
        margin = [0] * lat.grid.nd
        for p in points:
            for i in range(lat.grid.nd):
                x = abs(p[i])
                if x > margin[i]:
                    margin[i] = x

        self.padding = g.padded_local_fields(lat, margin)
        self.local_stencil = g.local_stencil.matrix(
            self.padding(lat), points, code, code_parallel_block_size
        )
        self.write_fields = None
        self.verbose_performance = g.default.is_verbose("stencil_performance")
        # Reusable scratch buffers for the write/temporary (non-read) padded
        # fields. Layout is fully determined by `lat` + `margin` so the size
        # is invariant across calls. Reusing them avoids leaking
        # ~Nscratch * padded_lattice_bytes per call into Grid's allocator
        # cache, which under the Metal allocator does not evict aggressively
        # enough to bound RSS for repeated stencil applications (e.g. stout
        # smear in an HMC loop). The read-field padded buffers still get a
        # fresh padding() copy each call because their source data changes.
        self._scratch_padded = None
        self._scratch_n = None

    def data_access_hints(self, write_fields, read_fields, cache_fields):
        self.write_fields = write_fields
        self.read_fields = read_fields
        self.cache_fields = cache_fields

    def __call__(self, *fields):
        if self.write_fields is None:
            raise Exception(
                "Generalized matrix stencil needs more information.  Call stencil.data_access_hints."
            )
        if self.verbose_performance:
            t = g.timer("stencil.matrix")
            t("create fields")
        n = len(fields)
        # Lazy-init reusable padded buffers for ALL slots. Layout is fully
        # determined by `lat` + `margin` + `otype` so size is invariant
        # across calls. Allocating fresh each call leaked
        # ~Nfields * padded_lattice_bytes per call into Grid's allocator
        # cache, which under the Metal allocator does not evict aggressively
        # enough to bound RSS for repeated stencil applications (e.g. stout
        # smear inside an HMC sweep). For 16^4 SU(2) single + 4-mu staple
        # that was ~235 MB/iter, killing long HMC runs in <1 h.
        if (
            self._scratch_padded is None
            or self._scratch_n != n
        ):
            # Use any read field to size the padded layout.
            template_src = None
            for i in range(n):
                if i in self.read_fields:
                    template_src = fields[i]
                    break
            assert template_src is not None
            template_padded = self.padding(template_src)
            self._scratch_padded = [
                template_padded if i == 0
                # avoid double-allocating the template by reusing it for slot 0;
                # but only if slot 0 is a read slot (we refill below either way).
                else g.lattice(template_padded)
                for i in range(n)
            ]
            self._scratch_n = n
        padded_fields = self._scratch_padded
        # Refill read-field padded buffers from the current sources via
        # project (no new allocation).
        read_indices = [i for i in range(n) if i in self.read_fields]
        if read_indices:
            self.padding.domain.project(
                [padded_fields[i] for i in read_indices],
                [fields[i] for i in read_indices],
            )
        if self.verbose_performance:
            t("local stencil")
        self.local_stencil(*padded_fields)
        if self.verbose_performance:
            t("extract")
        for i in self.write_fields:
            self.padding.extract(fields[i], padded_fields[i])
        if self.verbose_performance:
            t()
            g.message(t)
        # todo: make use of cache_fields


def matrix(lat, points, code, code_parallel_block_size=None):
    # check if all points are cartesian
    for p in points:
        if len([s for s in p if s != 0]) > 1:
            return matrix_padded(lat, points, code, code_parallel_block_size)
    return g.local_stencil.matrix(lat, points, code, code_parallel_block_size, local=0)
