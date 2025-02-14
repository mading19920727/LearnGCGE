/**
 * @brief 基于CCS（压缩列存储）格式的矩阵和向量操作
 */
#ifndef  _APP_CCS_H_
#define  _APP_CCS_H_

#include	"ops.h"
#include	"app_lapack.h"
typedef struct CCSMAT_ {
	double *data ; 
	int    *i_row; int *j_col;
	int    nrows ; int ncols ;
} CCSMAT;


void OPS_CCS_Set  (struct OPS_ *ops);
	
#endif  /* -- #ifndef _APP_CCS_H_ -- */
