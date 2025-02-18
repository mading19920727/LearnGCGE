### 采用MPI运行方式
#### 编译前(编译方法不变)
 - 修改`gcge\include\ops_config.h`中`#define  OPS_USE_MPI       0`值为1
 - 取消`CMakeLists.txt`中注释` # include(linux_mpicc)`

#### 运行时
 - 命令行中执行`mpiexec -n <num> ./test.exe K.mtx M.mtx`, 其中<num>为期望采用的进程数量






# GCGE文件结构

app

config

src

include

# GCGE配置编译

## 外部包

1. 基础线性代数包 BLAS和LAPACK

目前调用的是openblas，如果调用其他库自行修改CMakeLists.txt

2. UMFPACK

直接法解法器，可以加速收敛

## 编译

```bash
cd GCGE1.0_linux/
mkdir build
cd build
cmake ..
make 
```

生成的库文件及头文件见GCGE1.0_linux/shared

## 调用

示例程序路径 GCGE1.0_linux/example

在CMakeLists.txt中已包含对该算例的编译，生成的可执行文件为GCGE1.0_linux/build/test

执行

```bash
./test test1.mtx
```

其中test1.mtx 即为提供的测试算例，补充了上三角部分（标准稀疏矩阵库mtx格式）


## 备注

该算例条件数较大，且矩阵比较病态，目前设置的参数并不适合一般矩阵的特征值问题（会影响效率）；

GCGE1.0_linux/include/ops_config.h 中包含该软件包涉及的一些计算环境的宏定义，注意同步修改这些宏定义；

目前包里提供的mtx转CSC的程序是通用的，注意矩阵文件需要是完整的；

计算结果已与主流特征值解法器SLEPC进行对比，即便是后面较大量级的特征值，在对应精度范围内小数点都是能对上的，且1的代数重数也是一样的