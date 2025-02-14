# project
cmake_minimum_required(VERSION 3.5)
project (GCGE)
set(GCGE_VERSION_MAJOR 1)
set(GCGE_VERSION_MINOR 0)

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

# src
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/app SRC_APP)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)

# this project
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#UMFPACK
# 问题3：将umfpack的头文件路径放置在了环境变量中，通过添加环境变量将其添加到include_directories中，方式不好。
#        可以直接建立子文件夹，将umfpack的头文件放置在其中，然后通过include_directories添加。

include_directories($ENV{SUITESPARSE_HEADER_DIR})
# shared library
add_library(GCGE SHARED
    ${SRC_APP}
    ${SRCS}
)

# 问题1：没有指定openblas发布件和头文件的位置，因此需要保证openblas的so和头文件均在系统路径中，这样才找得到。
# 问题2：没有指定umfpack发布件的位置，因此需要保证umfpack的so在系统路径中，这样才找得到。

target_link_libraries(GCGE
    PUBLIC
        openblas
)

set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/shared/lib)
set(INCLUDE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/shared/include)
file(GLOB_RECURSE HEADER_ALL
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
)
file(COPY ${HEADER_ALL} DESTINATION ${INCLUDE_OUTPUT_PATH})

# test
add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/example/test_gcge_01.c)
target_link_libraries(test 
    PRIVATE 
        GCGE
)
