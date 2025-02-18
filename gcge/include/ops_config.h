
#ifndef  _OPS_CONFIG_H_
#define  _OPS_CONFIG_H_

#define  OPS_USE_HYPRE     0
#define  OPS_USE_INTEL_MKL 0
#define  OPS_USE_MATLAB    0
#define  OPS_USE_MEMWATCH  0
#define  OPS_USE_MPI       1
#define  OPS_USE_MUMPS     0
#define  OPS_USE_OMP       0
#define  OPS_USE_PHG       0
#define  OPS_USE_PETSC     0
#define  OPS_USE_SLEPC     0
#define  OPS_USE_UMFPACK   0

#define  PRINT_RANK    0

#if OPS_USE_MATLAB || OPS_USE_INTEL_MKL
#define FORTRAN_WRAPPER(x) x
#else
#define FORTRAN_WRAPPER(x) x ## _
#endif

#if OPS_USE_OMP
#define OMP_NUM_THREADS 2
#endif

//#if OPS_USE_INTEL_MKL
//#define MKL_NUM_THREADS 16
//#endif

#if OPS_USE_MEMWATCH
#include "../test/memwatch.h"
#endif

#endif  /* -- #ifndef _OPS_CONFIG_H_ -- */
