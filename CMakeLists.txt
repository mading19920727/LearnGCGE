# project
cmake_minimum_required(VERSION 2.8)
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
include_directories($ENV{SUITESPARSE_HEADER_DIR})
# shared library
add_library(GCGE SHARED
    ${SRC_APP}
    ${SRCS}
)

target_link_libraries(GCGE
    PUBLIC
        openblas
        umfpack
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
