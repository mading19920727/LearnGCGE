/**
 * @brief GCGE特征值求解器相关参数定义
 */

#ifndef __BEF_GCGE_SRC_GCGE_PARA_H__
#define __BEF_GCGE_SRC_GCGE_PARA_H__

#include <cfloat>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

extern "C" {
#include "app_ccs.h"
}

struct GcgeParam {
    int nevConv{5};                // 希望收敛到的特征值个数
    int block_size{nevConv};       // 块大小
    int nevInit{2 * nevConv};      // 初始X块的大小
    int max_iter_gcg{1000};        // 最大迭代次数
    int nevMax{2 * nevConv};       // 最大特征值个数
    double tol_gcg[2]{1e-1, 1e-5}; //精度，0是绝对， 1 相对
	double shift = 1;
    int nevGiven{0};
    int multiMax{1};
    int flag{0}; // 是否使用外部线性方程组求解器
    double gapMin{1e-5};
};

int eigenSolverGCG(void* A, void* B, std::vector<double>& eigenvalue, std::vector<std::vector<double>>& eigenvector,
                          struct GcgeParam* gcgeparam, struct OPS_* ops);
void multiVecReturn(LAPACKVEC* x, int end, std::vector<std::vector<double>>& eigenvector);
int destroyMatrixCCS(CCSMAT* ccs_matA, CCSMAT* ccs_matB);

#endif // __BEF_GCGE_SRC_GCGE_PARA_H__