# MPI 
find_package(MPI REQUIRED)

# complier
SET(CMAKE_C_COMPILER "mpicc")
SET(CMAKE_CXX_COMPILER "mpicpc")

# complie flag
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC  -std=c99")