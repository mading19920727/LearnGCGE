
#include	<stdio.h>
#include	<stdlib.h>
#include	<math.h>
#include	<assert.h>

#include    "ops.h"
#include    "app_lapack.h"

#define     DEBUG 0




void OPS_Create  (OPS **ops)
{
	*ops = malloc(sizeof(OPS));
	(*ops)->Printf                   = NULL;
	(*ops)->GetWtime                 = NULL;
	(*ops)->GetOptionFromCommandLine = NULL; 
	/* mat */
	(*ops)->MatAxpby                 = NULL;
	(*ops)->MatView                  = NULL;
	/* vec */
	(*ops)->VecCreateByMat           = NULL;
	(*ops)->VecCreateByVec           = NULL;
	(*ops)->VecDestroy               = NULL;
	(*ops)->VecView                  = NULL;
	(*ops)->VecInnerProd             = NULL;
	(*ops)->VecLocalInnerProd        = NULL;  
	(*ops)->VecSetRandomValue        = NULL;  
	(*ops)->VecAxpby                 = NULL;  
	(*ops)->MatDotVec                = NULL;  
	(*ops)->MatTransDotVec           = NULL;
	/* multi-vec */
	(*ops)->MultiVecCreateByMat      = NULL;  
	(*ops)->MultiVecCreateByVec      = NULL;  
	(*ops)->MultiVecCreateByMultiVec = NULL;  
	(*ops)->MultiVecDestroy          = NULL;  
	(*ops)->GetVecFromMultiVec       = NULL;  
	(*ops)->RestoreVecForMultiVec    = NULL;  
	(*ops)->MultiVecView             = NULL;  
	(*ops)->MultiVecLocalInnerProd   = NULL;  
	(*ops)->MultiVecInnerProd        = NULL;  
	(*ops)->MultiVecSetRandomValue   = NULL;  
	(*ops)->MultiVecAxpby            = NULL;  
	(*ops)->MultiVecLinearComb       = NULL;  
	(*ops)->MatDotMultiVec           = NULL; 
	(*ops)->MatTransDotMultiVec      = NULL; 
	(*ops)->MultiVecQtAP             = NULL;
	/* dense mat */
	(*ops)->lapack_ops               = NULL;
	(*ops)->DenseMatQtAP             = NULL;
	(*ops)->DenseMatOrth             = NULL;
	/* linear solver */
	(*ops)->LinearSolver             = NULL;	
	(*ops)->linear_solver_workspace  = NULL; 
	(*ops)->MultiLinearSolver        = NULL; 
	(*ops)->multi_linear_solver_workspace = NULL;
	/* orth */
	(*ops)->MultiVecOrth             = NULL; 
	(*ops)->orth_workspace           = NULL;
	/* muti gird */
	(*ops)->MultiGridCreate          = NULL;
	(*ops)->MultiGridDestroy         = NULL;
	(*ops)->VecFromItoJ              = NULL;
	(*ops)->MultiVecFromItoJ         = NULL;
	/* eigen solver */
	(*ops)->EigenSolver              = NULL; 
	(*ops)->eigen_solver_workspace   = NULL; 
	/* app ops for pas */
	(*ops)->app_ops                  = NULL;
	return;
}
void OPS_Setup   (OPS  *ops)
{
	if (ops->Printf == NULL) {
		ops->Printf = DefaultPrintf;
	}
	if (ops->GetWtime == NULL) {
		ops->GetWtime = DefaultGetWtime;
	}
	if (ops->GetOptionFromCommandLine == NULL) {
		ops->GetOptionFromCommandLine = DefaultGetOptionFromCommandLine;
	}
	if (ops->lapack_ops == NULL) {
		OPS_Create (&(ops->lapack_ops));
		OPS_LAPACK_Set (ops->lapack_ops);
	}
	if (ops->DenseMatQtAP == NULL) {
		ops->DenseMatQtAP = ops->lapack_ops->DenseMatQtAP;
	}
	if (ops->DenseMatOrth == NULL) {
		ops->DenseMatOrth = ops->lapack_ops->DenseMatOrth;
	}
	if (ops->VecFromItoJ == NULL) {
		ops->VecFromItoJ = DefaultVecFromItoJ;
	}
	if (ops->MultiVecFromItoJ == NULL) {
		ops->MultiVecFromItoJ = DefaultMultiVecFromItoJ;
	}
	if (ops->MultiVecInnerProd == NULL) {
		ops->MultiVecInnerProd = DefaultMultiVecInnerProd;
	}
	if (ops->MultiVecQtAP == NULL) {
		ops->MultiVecQtAP = DefaultMultiVecQtAP;
	}
	
#if 0	

	ops->MultiVecCreateByMat      = DefaultMultiVecCreateByMat     ;
	ops->MultiVecCreateByVec      = DefaultMultiVecCreateByVec     ;
	ops->MultiVecCreateByMultiVec = DefaultMultiVecCreateByMultiVec;
	ops->MultiVecDestroy          = DefaultMultiVecDestroy         ;
	ops->GetVecFromMultiVec       = DefaultGetVecFromMultiVec      ;
	ops->RestoreVecForMultiVec    = DefaultRestoreVecForMultiVec   ;
	ops->MultiVecView             = DefaultMultiVecView            ;
	ops->MultiVecLocalInnerProd   = DefaultMultiVecLocalInnerProd  ;
	ops->MultiVecInnerProd        = DefaultMultiVecInnerProd       ;
	ops->MultiVecSetRandomValue   = DefaultMultiVecSetRandomValue  ;
	ops->MultiVecAxpby            = DefaultMultiVecAxpby           ;
	ops->MultiVecLinearComb       = DefaultMultiVecLinearComb      ;
	ops->MatDotMultiVec           = DefaultMatDotMultiVec          ;
	ops->MatTransDotMultiVec      = DefaultMatTransDotMultiVec     ;
	ops->MultiVecQtAP             = DefaultMultiVecQtAP            ;

#endif
	
	return;
}
void OPS_Destroy (OPS **ops)
{
	if ((*ops)->lapack_ops!=NULL) {
		OPS_Destroy(&((*ops)->lapack_ops));
	}
	free(*ops); *ops = NULL;
	return;
}

/* ���ݷ��� */
static double *_array;
static int _comp(const void* a,const void* b){
  return (_array[*(int*)a]- _array[*(int*)b]>0)?-1:1;
}

/*
* @param  ad       'A' or 'D'
* @param  destin
* @param  length
* @param  min_gap
* @param  min_num
* @param  ntype     
* @param  displs
* @param  dbl_ws  length�� 
* @param  int_ws  length��
*
* @return 
*/
int SplitDoubleArray(double *destin, int length, 
		int num_group, double min_gap, int min_num, int* displs, 
		double *dbl_ws, int *int_ws)
{
	int i, j, k, num_non_empty_group; 
	double *dist = dbl_ws; int *idx = int_ws;	
	assert(num_group > 0 && length > 0);
	displs[0] = 0; 
	for (j = 0; j < num_group; ++j) {
		displs[j+1] = length; 
	}
	if (num_group == 1) return 0;

	dist[0] = 0.0;
#if DEBUG
	printf("dist[%d] = %f\n",0,dist[0]);
#endif
	for (k = 1; k < length; ++k) {
		dist[k] = fabs((destin[k] - destin[k-1])/(fabs(destin[k])==0.0?0.01:fabs(destin[k])));
#if DEBUG
		printf("dist[%d] = %f\n",k,dist[k]);
#endif
	}	
	for (k = 0; k < length; ++k) {
		idx[k] = k;
	}
	_array = dist;
	qsort(idx,length,sizeof(int),_comp);
#if DEBUG
	for (k = 0; k < length; ++k) {
		printf("idx[%d] = %d\n",k,idx[k]);
	}
#endif	
	num_non_empty_group = 1;
	min_num = min_num>=1?min_num:1;
	min_gap = min_gap>0.0?min_gap:0.0;
	for (k = 0; dist[idx[k]] > min_gap && k < length-1; ++k) {
		for (j = 0; j < num_group; ++j) {
			if (idx[k]-displs[j]>=min_num && displs[j+1]-idx[k]>=min_num) {			
				for (i = num_group-1; i > j; --i) {
					displs[i+1] = displs[i];
				}
				displs[j+1] = idx[k];
				++num_non_empty_group;
				break;		
			}
		}
		if (num_non_empty_group>=num_group) break;	
	}
#if DEBUG
	for (j = 0; j < num_group; ++j) {
		printf("[%d] %d <= j < %d\n",j,displs[j],displs[j+1]);
	}
#endif	
	return 0;
}


#if OPS_USE_MPI
/* �Ӿ���ͨѶ */
static int SUBMAT_TYPE_NROWS = 0; 
static int SUBMAT_TYPE_NCOLS = 0; 
static int SUBMAT_TYPE_LDA   = 0; 
static int SUBMAT_OP_USED    = 0;
/* �������Ͳ��� */
static void user_fn_submat_sum(double *in, double *inout, 
	int *len, MPI_Datatype* data_type)
{
	int i, j; double *a, *b;
	double one = 1.0; int inc = 1;
	for (i = 0; i < *len; ++i) {
		for (j = 0; j < SUBMAT_TYPE_NCOLS; ++j) {
			a = in   +SUBMAT_TYPE_LDA*j;
			b = inout+SUBMAT_TYPE_LDA*j; 
			daxpy(&SUBMAT_TYPE_NROWS,&one,a,&inc,b,&inc);
		}
	}
}
/* ����鴴�� */
int CreateMPIDataTypeSubMat(MPI_Datatype *data_type,
	int nrows, int ncols, int ldA)
{
	/* int MPI_Type_vector(
			int count, int blocklength, int stride,
    		MPI_Datatype oldtype, MPI_Datatype *newtype) */ 
	MPI_Type_vector(ncols,nrows,ldA,MPI_DOUBLE,data_type);
	MPI_Type_commit(data_type);
	if (SUBMAT_OP_USED == 0) {
		SUBMAT_TYPE_NROWS = nrows; SUBMAT_TYPE_LDA = ldA;
		SUBMAT_TYPE_NCOLS = ncols;		
	}
	return 0;
}
int CreateMPIOpSubMatSum(MPI_Op *op)
{
	assert(SUBMAT_OP_USED == 0);
	SUBMAT_OP_USED = 1;
	/* int commute = 1; */
	MPI_Op_create((MPI_User_function*)user_fn_submat_sum,1,op);
	return 0;
}
int DestroyMPIOpSubMatSum(MPI_Op *op)
{
	assert(SUBMAT_OP_USED == 1);
	SUBMAT_OP_USED = 0;
	MPI_Op_free(op);
	return 0;
}
/* ��������� */
int DestroyMPIDataTypeSubMat(MPI_Datatype *data_type)
{
	MPI_Type_free(data_type);
	if (SUBMAT_OP_USED == 0) {
		SUBMAT_TYPE_NROWS = 0; SUBMAT_TYPE_NCOLS = 0;
		SUBMAT_TYPE_LDA   = 0;
	}
	return 0;
}
#endif
