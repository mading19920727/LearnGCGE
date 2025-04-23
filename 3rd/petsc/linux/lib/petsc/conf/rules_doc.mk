# -*- mode: makefile-gmake -*-
#
# This is included in this file so it may be used from any source code PETSc directory
libs: /home/zzy/deps/petsc_mumps_tutorial-3.22.3/lib/petsc/conf/files /home/zzy/deps/petsc_mumps_tutorial-3.22.3/tests/testfiles
	+@r=`echo "${MAKEFLAGS}" | grep ' -j'`; \
        if [ "$$?" = 0 ]; then make_j=""; else make_j="-j${MAKE_NP}"; fi; \
	r=`echo "${MAKEFLAGS}" | grep ' -l'`; \
        if [ "$$?" = 0 ]; then make_l=""; else make_l="-l${MAKE_LOAD}"; fi; \
        cmd="${OMAKE_PRINTDIR} -f gmakefile $${make_j} $${make_l} ${MAKE_PAR_OUT_FLG} V=${V} libs"; \
        cd /home/zzy/deps/petsc_mumps_tutorial-3.22.3 && echo $${cmd} && exec $${cmd}
