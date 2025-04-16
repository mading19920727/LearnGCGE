/**
 * @brief 计算矩阵的负特征值数目
 * @author mading
 * @date 2025-03-24
 */

#include "count_eigen_num.h"

GcgeErrCode CountEigenNum::countEigenNum(void *A, void *B, double a, double b, int &numEigen) {
    printf("----countEigenNum\n");
    if (a > b) {    // [a, b]区间,应该 a =< b
        return GCGE_ERR_INPUT;  
    }
    
    /*计算cholesky(A - a * B),并获得惯性数值*/
    // 计算A - a * B
    Mat A_aB;   // 保存A - a * B
    MatDuplicate((Mat)B, MAT_COPY_VALUES, &A_aB);           // 拷贝B，作B = A - a * B
    MatAYPX(A_aB, -a, (Mat)A, DIFFERENT_NONZERO_PATTERN);   // MatAYPX(Y, a, X, ..)功能为 Y = a * Y + X 
    
    // 执行cholesky分解
    IS row, col;                // 分解排序索引
    MatFactorInfo info;         // 用于排序
    PetscInt nneg, nzero, npos; // 保存惯性数值
    Mat chol_AaB;   // cholesky分解后的矩阵

    // mumps参数设置，才能获得全局惯性
    PetscOptionsInsertString(NULL, "-mat_mumps_icntl_24 1"); 

    MatFactorInfoInitialize(&info);           
    MatGetFactor(A_aB, MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY, &chol_AaB); // mumps的cholesky分解
              
    MatGetOrdering(A_aB, MATORDERINGEXTERNAL, &row, &col);        // 排序
    MatCholeskyFactorSymbolic(chol_AaB, A_aB, row, &info);  // 符号分析
    MatCholeskyFactorNumeric(chol_AaB, A_aB, &info);        // 数值分解

    // 计算惯性指数
    MatGetInertia(chol_AaB, &numEigen, &nzero, &npos);
    MatDestroy(&A_aB);
    ISDestroy(&row);
    ISDestroy(&col);
    MatDestroy(&chol_AaB);
    // --------------------------------------------------------------------------------------

    /*计算cholesky(A - b * B),并获得惯性数值*/
    // 计算A - b * B
    Mat A_bB;   // 保存A - b * B
    MatDuplicate((Mat)B, MAT_COPY_VALUES, &A_bB); 
    MatAYPX(A_bB, -b, (Mat)A, DIFFERENT_NONZERO_PATTERN);   // 因为MatAYPX(Y, a, X, ..)功能为 Y = a * Y + X，不能复用A_aB   

    Mat chol_AbB;   // cholesky分解后的矩阵
    MatGetFactor(A_bB, MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY, &chol_AbB);
    MatGetOrdering(A_bB, MATORDERINGEXTERNAL, &row, &col);        // 排序
    MatCholeskyFactorSymbolic(chol_AbB, A_bB, row, &info);  // 复用了info
    MatCholeskyFactorNumeric(chol_AbB, A_bB, &info);

    // 计算惯性指数
    MatGetInertia(chol_AbB, &nneg, &nzero, &npos); 

    // 区间内特征值个数
    int tempN = numEigen;
    numEigen = nneg - numEigen;
    PetscPrintf(PETSC_COMM_WORLD, "    The number of eigenvalues in [a, b] is: %d = %d - %d\n", numEigen, nneg, tempN);

    MatDestroy(&A_bB);
    ISDestroy(&row);
    ISDestroy(&col);
    MatDestroy(&chol_AbB);

    return GCGE_SUCCESS;
}

GcgeErrCode CountEigenNum::processMatDAD(Mat &A, Mat &B) {
    Vec diag;                   // 向量
    MatCreateVecs(A, &diag, NULL);
    MatGetDiagonal(A, diag);    // 对角线向量
    VecSqrtAbs(diag);           // 开方运算
    VecReciprocal(diag);        // 倒数运算

    MatDiagonalScale(A, diag, diag);   // DAD
    MatDiagonalScale(B, diag, diag);   // DBD

    // MatView(A, PETSC_VIEWER_STDOUT_WORLD);    // 可视化DAD效果
    // MatView(B, PETSC_VIEWER_STDOUT_WORLD);    // 可视化DBD效果

    VecDestroy(&diag);
    return GCGE_SUCCESS;
} 