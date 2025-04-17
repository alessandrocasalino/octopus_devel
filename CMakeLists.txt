cmake_minimum_required(VERSION 3.16)
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

    # Find and link rocBLAS
    find_package(rocblas REQUIRED)
    target_link_libraries(octopus_devel PRIVATE roc::rocblas hip::device)
else()
    message(STATUS "HIP not found. Building with CUDA support.")
    find_package(CUDA REQUIRED)
    set(CMAKE_CXX_COMPILER ${CUDA_NVCC_EXECUTABLE})

    # Link cuBLAS
    target_link_libraries(octopus_devel PRIVATE cuda cublas)
endif()

# Link HIP or CUDA libraries
if(HIP_FOUND)
    target_link_libraries(octopus_devel PRIVATE hip::device)
else()
    target_link_libraries(octopus_devel PRIVATE cuda)
endif()

