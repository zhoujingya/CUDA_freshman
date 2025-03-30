cmake_minimum_required(VERSION 3.22)
project(cuda_freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=sm_80")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(01_hello_world)
add_subdirectory(02_vector_add)
add_subdirectory(03_matrix_multiplication)
