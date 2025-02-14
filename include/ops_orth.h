
#ifndef  _OPS_ORTH_H_
#define  _OPS_ORTH_H_

#include    "ops.h"
#include    "app_lapack.h"

typedef struct ModifiedGramSchmidtOrth_ {
	int    block_size;    /* ������������С */ 
	int    max_reorth;
	double orth_zero_tol; /* ���������     */ 
	double reorth_tol;
	void   **mv_ws;       /* �����������ռ� */
	double *dbl_ws;      /* �����͹����ռ� */
} ModifiedGramSchmidtOrth;

typedef struct BinaryGramSchmidtOrth_ {
	int    block_size;    /* ������������С */ 
	int    max_reorth;
	double orth_zero_tol; /* ���������     */ 
	double reorth_tol;
	void   **mv_ws;       /* �����������ռ� */
	double *dbl_ws;      /* �����͹����ռ� */
} BinaryGramSchmidtOrth;

void MultiVecOrthSetup_ModifiedGramSchmidt(
	int block_size, int max_reorth, double orth_zero_tol, 
	void **mv_ws, double *dbl_ws, struct OPS_ *ops);
void MultiVecOrthSetup_BinaryGramSchmidt(
	int block_size, int max_reorth, double orth_zero_tol, 
	void **mv_ws, double *dbl_ws, struct OPS_ *ops);

#endif  /* -- #ifndef _OPS_ORTH_H_ -- */


