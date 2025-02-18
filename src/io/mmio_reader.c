/**
 * @brief 读取MTX文件的工具
 * @author zhangzy(zhiyu.zhang@cqbdri.pku.edu.cn)
 * @date 2025-02-18
 */

#include "mmio_reader.h"
#include "io/mmio_h.h"
#include "io/mmio.h"

int CreateCCSFromMTX(CCSMAT *ccs_matA, char* file_matrix)
{
    // 此函数分两部分，一部分为读取MTX文件，获取数据到临时变量；另一部分为将读取的数据存入CCS矩阵结构体中
	int m, n, nnzA, isSymmetricA;
	int *row_ptr = NULL;
    int *col_idx = NULL;
	int read_matrix_base = 1;
    double *val = NULL;
    // 调用mmio_allinone函数读取MTX文件，获取矩阵的行数、列数、非零元素数、是否对称等信息
    // 并将矩阵的数据存储到临时变量row_ptr、col_idx和val中
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
