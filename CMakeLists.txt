cmake_minimum_required(VERSION 3.10)
project(mpi_matrix_multiplication)

set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_FLAGS "-g3 -Wall -O0")
set(CMAKE_CXX_FLAGS "-std=c++17 -g3 -O3")

SET(CMAKE_CXX_COMPILER CC)


add_executable(matrixmul
        src/densematgen.cpp
        include/densematgen.h
        src/utils.cpp
        include/utils.h
        src/matrix.cpp
        include/matrix.h
        src/mpi.cpp
        include/mpi.h
        src/matmul.cpp
        include/matmul.h
        src/inner_abc.cpp
        src/col_a.cpp
        src/main.cpp)

add_executable(generator
        src/densematgen.cpp
        include/densematgen.h
        src/generator_main.cpp)

