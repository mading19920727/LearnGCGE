#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <memory.h>
#include <time.h>
#include <float.h>
#include "ops_eig_sol_gcg.h"
#include "mmio_h.h"
#include "app_ccs.h"
#include    "ops.h"

#if OPS_USE_UMFPACK
#include "umfpack.h"
/*
  Create an application context to contain data needed by the
  application-provided call-back routines, ops->MultiLinearSolver().
*/
typedef struct {
   void   *Symbolic; 
   void   *Numeric;
   int    *Ap;
   int    *Ai;
   double *Ax;
   double *null;
   int    n;
} AppCtx;
static void AppCtxCreate(AppCtx *user, CCSMAT *ccs_mat)
{
   user->Ap   = ccs_mat->j_col;
   user->Ai   = ccs_mat->i_row;
   user->Ax   = ccs_mat->data;
   user->null = (double*)NULL;
   user->n    = ccs_mat->nrows;
   umfpack_di_symbolic(user->n, user->n, user->Ap, user->Ai, user->Ax, &(user->Symbolic), user->null, user->null);
   umfpack_di_numeric(user->Ap, user->Ai, user->Ax, user->Symbolic, &(user->Numeric), user->null, user->null);
   umfpack_di_free_symbolic(&(user->Symbolic));
   return;
}
static void AppCtxDestroy(AppCtx *user)
{
   umfpack_di_free_numeric(&(user->Numeric));
   user->Ap = NULL;
   user->Ai = NULL;
   user->Ax = NULL;
   return;
}
void UMFPACK_MultiLinearSolver(void *mat, void **b, void **x, int *start, int *end, struct OPS_ *ops)
{
   assert(end[0]-start[0]==end[1]-start[1]);
   AppCtx *user = (AppCtx*)ops->multi_linear_solver_workspace;
   LAPACKVEC *b_vec = (LAPACKVEC*)b, *x_vec = (LAPACKVEC*)x;
   double *b_array = b_vec->data+start[0]*b_vec->ldd; 
   double *x_array = x_vec->data+start[1]*x_vec->ldd; 
   int idx, ncols = end[0]-start[0];
   for (idx = 0; idx < ncols; ++idx) {
      umfpack_di_solve (UMFPACK_A, user->Ap, user->Ai, user->Ax, 
	    x_array, b_array, user->Numeric, user->null, user->null);
      b_array += b_vec->ldd; x_array += x_vec->ldd;
   }
   return;
}
#endif

static int CreateCCSFromMTX(CCSMAT *ccs_matA, char* file_matrix);
static int DestroyMatrixCCS(CCSMAT *ccs_matA);

int TestEigenSolverGCG   (void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops, double shift);

int TestEigenSolverGCG(void *A, void *B, int flag, int argc, char *argv[], struct OPS_ *ops, double shift) 
{

	int nevConv  = 100; //number of required eigenpairs 
	double gapMin = 1e-9, abs_tol, rel_tol;
	int nevGiven = 0, block_size = nevConv, nevMax = 2*nevConv, multiMax = 1, nevInit = nevMax;
	abs_tol = 1e-5; //abs tolerance
	rel_tol = 1e-5; //rel tolerance

	if (nevConv > 300)
	{
		block_size = nevConv / 10;
		nevInit = 5*block_size; 
		nevMax =  6*block_size + nevConv;
	}
#if 0
	nevConv = 1400;
	block_size = 140;
	nevInit = 560;
	nevMax = 2100;
#endif
	block_size = nevConv;
	nevInit = 2 * nevConv; 
	// nevMax = nevConv + nevInit;
	nevMax = 2 * nevConv;
	if (nevConv <= 50)
	{
		block_size = nevConv;
		nevInit = 2 * nevConv;
		nevMax = 2 * nevConv;
	}
	
	ops->GetOptionFromCommandLine("-nevConv"  ,'i',&nevConv   ,argc,argv,ops);
	ops->GetOptionFromCommandLine("-nevMax"   ,'i',&nevMax    ,argc,argv,ops);
	ops->GetOptionFromCommandLine("-blockSize",'i',&block_size,argc,argv,ops);
	ops->GetOptionFromCommandLine("-nevInit"  ,'i',&nevInit   ,argc,argv,ops);
	nevInit = nevInit<nevMax?nevInit:nevMax;
	int max_iter_gcg = 40; 
	double tol_gcg[2] = {1e-7,1e-8};
	tol_gcg[0] = abs_tol;
	tol_gcg[1] = rel_tol;

	double *eval; void **evec;
	eval = malloc(nevMax*sizeof(double));
	memset(eval,0,nevMax*sizeof(double));
	ops->MultiVecCreateByMat(&evec,nevMax,A,ops);
	ops->MultiVecSetRandomValue(evec,0,nevMax,ops);
		// ops->MultiVecView(evec,0,nevMax,ops);
	void **gcg_mv_ws[4]; double *dbl_ws; int *int_ws;
	ops->MultiVecCreateByMat(&gcg_mv_ws[0],nevMax+2*block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[0],0,nevMax+2*block_size,ops);
	ops->MultiVecCreateByMat(&gcg_mv_ws[1],block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[1],0,block_size,ops);
	ops->MultiVecCreateByMat(&gcg_mv_ws[2],block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[2],0,block_size,ops);
	ops->MultiVecCreateByMat(&gcg_mv_ws[3],block_size,A,ops);				
	ops->MultiVecSetRandomValue(gcg_mv_ws[3],0,block_size,ops);
	int sizeV = nevInit + 2*block_size;
	int length_dbl_ws = 2*sizeV*sizeV+10*sizeV
		+(nevMax+2*block_size)+(nevMax)*block_size;
	int length_int_ws = 6*sizeV+2*(block_size+3);
	dbl_ws = malloc(length_dbl_ws*sizeof(double));
	memset(dbl_ws,0,length_dbl_ws*sizeof(double));
	int_ws = malloc(length_int_ws*sizeof(int));
	memset(int_ws,0,length_int_ws*sizeof(int));

	srand(0);
	double time_start, time_interval;
	time_start = ops->GetWtime();
		
	ops->Printf("===============================================\n");
	ops->Printf("GCG Eigen Solver\n");
	EigenSolverSetup_GCG(multiMax,gapMin,nevInit,nevMax,block_size,
		tol_gcg,max_iter_gcg,flag,gcg_mv_ws,dbl_ws,int_ws,ops);
	
	int    check_conv_max_num    = 50   ;
		
	char   initX_orth_method[8]  = "mgs"; 
	int    initX_orth_block_size = 80   ; 
	int    initX_orth_max_reorth = 2    ; double initX_orth_zero_tol    = 2*DBL_EPSILON;//1e-12
	
	char   compP_orth_method[8]  = "mgs"; 
	int    compP_orth_block_size = -1   ; 
	int    compP_orth_max_reorth = 2    ; double compP_orth_zero_tol    = 2*DBL_EPSILON;//1e-12
	
	char   compW_orth_method[8]  = "mgs";
	int    compW_orth_block_size = 80   ; 	
	int    compW_orth_max_reorth = 2    ;  double compW_orth_zero_tol   = 2*DBL_EPSILON;//1e-12
	int    compW_bpcg_max_iter   = 300   ;  double compW_bpcg_rate       = 1e-2; 
	double compW_bpcg_tol        = 1e-14;  char   compW_bpcg_tol_type[8] = "abs";
	
	int    compRR_min_num        = -1   ;  double compRR_min_gap        = gapMin;
	double compRR_tol            = 2*DBL_EPSILON;
		
	EigenSolverSetParameters_GCG(
			check_conv_max_num   ,
			initX_orth_method    , initX_orth_block_size, 
			initX_orth_max_reorth, initX_orth_zero_tol  ,
			compP_orth_method    , compP_orth_block_size, 
			compP_orth_max_reorth, compP_orth_zero_tol  ,
			compW_orth_method    , compW_orth_block_size, 
			compW_orth_max_reorth, compW_orth_zero_tol  ,
			compW_bpcg_max_iter  , compW_bpcg_rate      , 
			compW_bpcg_tol       , compW_bpcg_tol_type  , 0, // without shift
			compRR_min_num       , compRR_min_gap       ,
			compRR_tol           , 
			ops);		

	EigenSolverSetParametersFromCommandLine_GCG(argc,argv,ops);
	ops->Printf("nevGiven = %d, nevConv = %d, nevMax = %d, block_size = %d, nevInit = %d\n",
			nevGiven,nevConv,nevMax,block_size,nevInit);
#if 1
	struct GCGSolver_ *gcgsolver;
    gcgsolver = (GCGSolver*)ops->eigen_solver_workspace;
	gcgsolver->tol[0] = 1e-3;
	gcgsolver->tol[1] = 1e-3;
	int initnev = nevConv;
	ops->EigenSolver(A,B,eval,evec,nevGiven,&initnev,ops);
	gcgsolver->tol[0] = tol_gcg[0];
	gcgsolver->tol[1] = tol_gcg[1];
	ops->EigenSolver(A,B,eval,evec,nevInit,&nevConv,ops); //Eigen solver
#else
	ops->EigenSolver(A,B,eval,evec,nevGiven,&nevConv,ops); //Eigen solver
#endif
	ops->Printf("numIter = %d, nevConv = %d\n",
			((GCGSolver*)ops->eigen_solver_workspace)->numIter, nevConv);
	ops->Printf("++++++++++++++++++++++++++++++++++++++++++++++\n");

	time_interval = ops->GetWtime() - time_start;
	ops->Printf("Time is %.3f\n", time_interval);

	ops->MultiVecDestroy(&gcg_mv_ws[0],nevMax+2*block_size,ops);
	ops->MultiVecDestroy(&gcg_mv_ws[1],block_size,ops);
	ops->MultiVecDestroy(&gcg_mv_ws[2],block_size,ops);
	ops->MultiVecDestroy(&gcg_mv_ws[3],block_size,ops);
	free(dbl_ws); free(int_ws);

#if 1
	ops->Printf("eigenvalues\n");
	int idx;
	for (idx = 0; idx < nevConv; ++idx) {
		eval[idx] -= shift;
		ops->Printf("%.10f\n",eval[idx]);
	}
#endif
	ops->MultiVecDestroy(&(evec),nevMax,ops);
	free(eval);
	return 0;
}

// argv[1]: 存储矩阵A的文件名
int TestAppCCS(int argc, char *argv[]) 
{
#if OPS_USE_MPI
   MPI_Init(&argc, &argv);
#endif

   OPS *ccs_ops = NULL;
   OPS_Create (&ccs_ops);
   OPS_CCS_Set (ccs_ops);
   OPS_Setup (ccs_ops);

   void *matA, *matB; OPS *ops;

   CCSMAT ccs_matA;
   char *filename = argv[1];
   CreateCCSFromMTX(&ccs_matA, filename);

   ops = ccs_ops; matA = (void*)(&ccs_matA); matB = NULL;

   int flag = 0;
   double sig = 0;

#if OPS_USE_UMFPACK 
   AppCtx user; flag = 1;
   if (flag>=1) {
      AppCtxCreate(&user, &ccs_matA);
      ops->multi_linear_solver_workspace = (void*)&user;
      ops->MultiLinearSolver = UMFPACK_MultiLinearSolver;
   }
#endif
   TestEigenSolverGCG(matA,matB,flag,argc,argv,ops,sig);

   OPS_Destroy (&ccs_ops);

#if OPS_USE_MPI
   MPI_Finalize();
#endif
   return 0;
}

/**
 * @brief 根据MTX文件创建CCS格式的稀疏矩阵
 * 
 * @param ccs_matA GCGE中定义个CCS矩阵的结构体对象
 * @param file_matrix mtx文件的路径
 * @return int 错误码
 */
static int CreateCCSFromMTX(CCSMAT *ccs_matA, char* file_matrix)
{
    // 此函数分两部分，一部分为读取MTX文件，获取数据到临时变量；另一部分为将读取的数据存入CCS矩阵结构体中
	int m, n, nnzA, isSymmetricA;
	int *row_ptr = NULL;
    int *col_idx = NULL;
	int read_matrix_base = 1;
    double *val = NULL;
    mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &read_matrix_base, &row_ptr, &col_idx, &val, file_matrix);
	ccs_matA->nrows = m; ccs_matA->ncols = n;
	ccs_matA->j_col = malloc((n+1)*sizeof(int));
	ccs_matA->i_row = malloc(nnzA*sizeof(int));
	ccs_matA->data  = malloc(nnzA*sizeof(double));
	int idx;
	for (idx = 0; idx < nnzA; idx++)
	{
		ccs_matA->data[idx] = val[idx];
	}
	for (idx = 0; idx < nnzA; idx++)
	{
		ccs_matA->i_row[idx] = col_idx[idx];
	}
	for (idx = 0; idx < n+1; idx++)
	{
		ccs_matA->j_col[idx] = row_ptr[idx];
	}
	free(row_ptr);
	free(val);
	free(col_idx);
	return 0;

} 

/**
 * @brief 销毁CCS格式的稀疏矩阵
 * 
 * @param ccs_matA CCS格式的稀疏矩阵
 * @return int 错误码
 */
static int DestroyMatrixCCS(CCSMAT *ccs_matA)
{
	free(ccs_matA->i_row); ccs_matA->i_row = NULL;
	free(ccs_matA->j_col); ccs_matA->j_col = NULL;
	free(ccs_matA->data) ; ccs_matA->data  = NULL;
	return 0;
}

int main(int argc, char *argv[]) 
{
    printf("start!!!!!!!!!!!!!!!!!!!!!");
	TestAppCCS(argc, argv);
   	return 0;
}


