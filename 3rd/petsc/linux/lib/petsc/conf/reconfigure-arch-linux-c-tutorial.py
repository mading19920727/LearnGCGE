#!/usr/bin/python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-bison',
    '--download-hwloc',
    '--prefix=/home/zzy/deps/petsc_mumps_tutorial-3.22.3',
    '--with-debugging=0',
    '--with-mpi=1',
    '--with-mpiexec=/home/zzy/deps/mpich-4.2.3/bin/mpiexec',
    '--with-openmp',
    '--with-shared-libraries=0',
    '-download-metis',
    '-download-mumps',
    '-download-parmetis',
    '-download-ptscotch',
    '-download-scalapack',
    'COPTFLAGS=-O',
    'CXXOPTFLAGS=-O',
    'FOPTFLAGS=-O',
    'PETSC_ARCH=arch-linux-c-tutorial',
  ]
  configure.petsc_configure(configure_options)
