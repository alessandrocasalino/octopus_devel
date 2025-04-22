cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0104 NEW) # Set policy to handle CUDA_ARCHITECTURES

project(OctopusDevel LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the source directory
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

# Add the executable
add_executable(octopus_devel ${SRC_DIR}/main.cpp ${SRC_DIR}/kernels.cpp)

# Include the header directory
target_include_directories(octopus_devel PRIVATE ${SRC_DIR})

# Check for ROCm/HIP
find_package(HIP)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(octopus_devel PRIVATE -g -O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(octopus_devel PRIVATE -O3)
endif()

if(HIP_FOUND)
    message(STATUS "HIP found. Building with ROCm support.")
    set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
    add_definitions(-D__HIP_PLATFORM_HCC__)
    target_compile_options(octopus_devel PRIVATE -DHIP_ENABLED --offload-arch=gfx942)

    find_package(rocblas REQUIRED)
    target_link_libraries(octopus_devel PRIVATE roc::rocblas hip::device)
else()
    message(STATUS "HIP not found. Building with CUDA support.")
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)

    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set_source_files_properties(${SRC_DIR}/main.cpp ${SRC_DIR}/kernels.cpp PROPERTIES LANGUAGE CUDA)

    target_compile_options(octopus_devel PRIVATE
         "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -std=c++20>")
    set_target_properties(octopus_devel PROPERTIES
             CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(octopus_devel PRIVATE cuda CUDA::cublas CUDA::cuda_driver)
    set_property(TARGET octopus_devel PROPERTY CUDA_ARCHITECTURES "80")
endif()

