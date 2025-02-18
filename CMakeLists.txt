# project
cmake_minimum_required(VERSION 3.16)
project (GCGE)
set(GCGE_VERSION_MAJOR 1)
set(GCGE_VERSION_MINOR 0)
set(CMAKE_VERBOSE_MAKEFILE ON)
# build type (set defualt type: RelWithDebInfo)
if(NOT CMAKE_BUILD_TYPE)
  message("Setting build type to 'RelWithDebInfo' as none was specified.")
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# include
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/config/)
include(linux_mpicc)

add_subdirectory(gcge)  # 添加 gcge 库的子目录
add_subdirectory(src)   # 添加 src 可执行文件的子目录
