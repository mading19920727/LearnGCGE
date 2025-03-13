# 定义 PETSc 作为 INTERFACE 库
add_library(petsc INTERFACE)

set(PETSC_DIR ${PROJECT_SOURCE_DIR}/3rd/petsc/) # 不能用/f/zzy/petsc，否则会报错  

# 判断当前系统
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PETSC_INCLUDE_DIR ${PETSC_DIR}/windows/include)
    set(PETSC_LIB_DIR ${PETSC_DIR}/windows/lib)
    set(PETSC_LIB_NAME petsc)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PETSC_INCLUDE_DIR ${PETSC_DIR}/linux/include)
    set(PETSC_LIB_DIR ${PETSC_DIR}/linux/lib)
    set(PETSC_LIB_NAME petsc)
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# 设置头文件路径
target_include_directories(petsc INTERFACE ${PETSC_INCLUDE_DIR})

# 设置库文件搜索路径
target_link_directories(petsc INTERFACE ${PETSC_LIB_DIR})

# 链接 PETSc 库
target_link_libraries(petsc INTERFACE ${PETSC_LIB_NAME})
