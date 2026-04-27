extra_sources=
extra_headers=

if BUILD_COMMS_MPI3
  extra_sources+=communicator/Communicator_mpi3.cc
  extra_sources+=communicator/Communicator_base.cc
  extra_sources+=communicator/SharedMemoryMPI.cc
  extra_sources+=communicator/SharedMemory.cc
endif

if BUILD_COMMS_NONE
  extra_sources+=communicator/Communicator_none.cc
  extra_sources+=communicator/Communicator_base.cc
  extra_sources+=communicator/SharedMemoryNone.cc
  extra_sources+=communicator/SharedMemory.cc
endif

if BUILD_HDF5
  extra_sources+=serialisation/Hdf5IO.cc 
  extra_headers+=serialisation/Hdf5IO.h
  extra_headers+=serialisation/Hdf5Type.h
endif


all: version-cache Version.h WilsonKernels.metallib

# Reproducible Metal compilation:
#   -std=metal3.0     pin language version (default tracks Xcode, breaks repro)
#   -O3               release optimisation
#   -frecord-sources  embed sources for crash logs / Xcode shader debugger
#   -Wall -Werror     don't ship a kernel with warnings
# We deliberately do NOT pass -ffast-math here: complex sums in Wilson Dslash
# rely on associative ordering for gauge-covariance to hold.
METAL_FLAGS = -std=metal3.0 -O3 -frecord-sources -Wall -Werror

WilsonKernels.metallib: $(top_srcdir)/Grid/qcd/action/fermion/WilsonKernels.metal
	@echo "Compiling Metal shaders..."
	xcrun -sdk macosx metal $(METAL_FLAGS) -c $(top_srcdir)/Grid/qcd/action/fermion/WilsonKernels.metal -o WilsonKernels.air
	xcrun -sdk macosx metallib WilsonKernels.air -o default.metallib

version-cache:
	@if [ `git status --porcelain | grep -v '??' | wc -l` -gt 0 ]; then\
		a="uncommited changes";\
	else\
		a="clean";\
	fi;\
	echo "`git log -n 1 --format=format:"#define GITHASH \\"%H:%d $$a\\"%n" HEAD`" > vertmp;\
	if [ -e version-cache ]; then\
		d=`diff vertmp version-cache`;\
		if [ "$${d}" != "" ]; then\
			mv vertmp version-cache;\
			rm -f Version.h;\
		fi;\
	else\
		mv vertmp version-cache;\
		rm -f Version.h;\
	fi;\
	rm -f vertmp

Version.h: version-cache
	cp version-cache Version.h

.PHONY: version-cache

#
# Libraries
#
include Make.inc
include Eigen.inc

extra_sources+=$(WILS_FERMION_FILES)
extra_sources+=$(STAG_FERMION_FILES)
if BUILD_ZMOBIUS
  extra_sources+=$(ZWILS_FERMION_FILES)
endif
if BUILD_GPARITY
  extra_sources+=$(GP_FERMION_FILES)
endif
if BUILD_FERMION_REPS
  extra_sources+=$(ADJ_FERMION_FILES)
  extra_sources+=$(TWOIND_FERMION_FILES)
endif
if BUILD_SP
    extra_sources+=$(SP_FERMION_FILES)
    extra_sources+=$(SP_TWOIND_FERMION_FILES)
endif

lib_LIBRARIES = libGrid.a

CCFILES += $(extra_sources)
HFILES  += $(extra_headers) Config.h Version.h

libGrid_a_SOURCES              = $(CCFILES)
libGrid_adir                   = $(includedir)/Grid
nobase_dist_pkginclude_HEADERS = $(HFILES) $(eigen_files) $(eigen_unsupp_files)
