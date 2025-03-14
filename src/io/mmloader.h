#ifndef __SRC_IO_MMLOADER_H__
#define __SRC_IO_MMLOADER_H__
#ifdef __cplusplus
extern "C" {
#endif
#include <petscmat.h>
#include "mmio.h"
#include <petscsys.h>

PetscErrorCode MatCreateFromMTX(Mat *A, const char *filein, PetscBool aijonly);
#ifdef __cplusplus
}
#endif
#endif // __SRC_IO_MMLOADER_H__