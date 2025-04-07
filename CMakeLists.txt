cmake_minimum_required(VERSION 3.22)

find_package(CUDAToolkit)

set(CMAKE_CUDA_ARCHITECTURES 86)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

if(CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found")
    message(STATUS "CUDAToolkit_VERSION: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDAToolkit_INCLUDE_DIRS: ${CUDAToolkit_INCLUDE_DIRS}")
    project(cuda_freshman CXX C CUDA)
    set(CMAKE_CUDA_FLAGS "-arch=sm_80")

    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

    include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)
    # Get all directory names in the current directory
    file(GLOB DIR_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

    # Loop through all directories and add those starting with a number and underscore
    foreach(DIR ${DIR_LIST})
        if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${DIR})
            # Check if directory starts with a digit followed by underscore
            string(REGEX MATCH "^[0-9]+_" MATCH_RESULT ${DIR})
            if(MATCH_RESULT)
                message(STATUS "Adding subdirectory: ${DIR}")
                add_subdirectory(${DIR})
            endif()
        endif()
    endforeach()
endif()
