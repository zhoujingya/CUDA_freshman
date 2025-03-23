cmake_minimum_required(VERSION 3.22)
project(cuda_freshman CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-arch=sm_80")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(01_hello_world)
