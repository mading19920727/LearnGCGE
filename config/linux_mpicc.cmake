# MPI 
find_package(MPI REQUIRED)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_PATH})
    set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()
# complier
SET(CMAKE_C_COMPILER "mpicc")
SET(CMAKE_CXX_COMPILER "mpicpc")

# complie flag
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC  -std=c99")