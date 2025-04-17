#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_runtime.h>
#define cuMalloc hipMalloc
#define cuMemcpy hipMemcpy
#define cuMemset hipMemset
#define cuMemcpyHostToDevice hipMemcpyHostToDevice
#define cuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cuMallocAsync hipMallocAsync
#define cuMemsetAsync hipMemsetAsync
#define cuFreeAsync hipFreeAsync
#define cuFree hipFree
#define cuEvent_t hipEvent_t
#define cuEventCreate hipEventCreate
#define cuEventRecord hipEventRecord
#define cuEventSynchronize hipEventSynchronize
#define cuEventElapsedTime hipEventElapsedTime
#define cuEventDestroy hipEventDestroy
#define cuLaunchKernel hipLaunchKernelGGL
#define cuDeviceProp hipDeviceProp_t
#define cuGetDevice hipGetDevice
#define cuGetDeviceProperties hipGetDeviceProperties
#define cuDeviceSynchronize hipDeviceSynchronize
#define cuMemcpyKind hipMemcpyKind
#define cuMemcpyHostToDevice hipMemcpyHostToDevice
#define cuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cuGetLastError hipGetLastError
#else
#include <cuda_runtime.h>
#define cuMalloc cudaMalloc
#define cuMemcpy cudaMemcpy
#define cuMemset cudaMemset
#define cuMemcpyHostToDevice cudaMemcpyHostToDevice
#define cuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define cuMallocAsync cudaMallocAsync
#define cuMemsetAsync cudaMemsetAsync
#define cuFreeAsync cudaFreeAsync
#define cuFree cudaFree
#define cuEvent_t cudaEvent_t
#define cuEventCreate cudaEventCreate
#define cuEventRecord cudaEventRecord
#define cuEventSynchronize cudaEventSynchronize
#define cuEventElapsedTime cudaEventElapsedTime
#define cuEventDestroy cudaEventDestroy
#define cuLaunchKernel <<<grid, block>>>
#define cuDeviceProp cudaDeviceProp
#define cuGetDevice cudaGetDevice
#define cuGetDeviceProperties cudaGetDeviceProperties
#define cuDeviceSynchronize cudaDeviceSynchronize
#define cuMemcpyKind cudaMemcpyKind
#define cuMemcpyHostToDevice cudaMemcpyHostToDevice
#define cuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define cuGetLastError cudaGetLastError
#endif

#ifdef __HIP_PLATFORM_HCC__
#define cuStreamCreate(stream) hipStreamCreate(&(stream))
#define cuStreamDestroy(stream) hipStreamDestroy(stream)
#define cuStreamSynchronize(stream) hipStreamSynchronize(stream)
#define cuStream_t hipStream_t
#else
#define cuStreamCreate(stream) cudaStreamCreate(&(stream))
#define cuStreamDestroy(stream) cudaStreamDestroy(stream)
#define cuStreamSynchronize(stream) cudaStreamSynchronize(stream)
#define cuStream_t cudaStream_t
#endif


#ifdef __HIP_PLATFORM_HCC__
    #define CUDA_CHECK(call)                                                      \
        {                                                                         \
            hipError_t err = call;                                                \
            if (err != hipSuccess) {                                              \
                std::cerr << "HIP Error: " << hipGetErrorString(err)              \
                          << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
                exit(EXIT_FAILURE);                                               \
            }                                                                     \
        }
#else
    #define CUDA_CHECK(call)                                                      \
        {                                                                         \
            cudaError_t err = call;                                               \
            if (err != cudaSuccess) {                                             \
                std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                          << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
                exit(EXIT_FAILURE);                                               \
            }                                                                     \
        }
#endif

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

#define DEVICE_FUNC __host__ __device__
#define double2 Double2

class __align__(16) double2 {
public:
    double x;
    double y;
    DEVICE_FUNC __forceinline__ double2(const double &a = 0, const double &b = 0) : x(a), y(b) {}
};


DEVICE_FUNC __forceinline__ static Double2 operator*(const double &a, const Double2 &b) {
    Double2 c;
    c.x = a * b.x;
    c.y = a * b.y;
    return c;
}

DEVICE_FUNC __forceinline__ static Double2 operator*(const Double2 &a, const Double2 &b) {
    Double2 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    return c;
}

DEVICE_FUNC __forceinline__ static Double2 operator+(const Double2 &a, const Double2 &b) {
    Double2 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

DEVICE_FUNC __forceinline__ static Double2 operator-(const Double2 &a, const Double2 &b) {
    Double2 c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

DEVICE_FUNC __forceinline__ static Double2 operator+=(Double2 &a, const Double2 &b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

DEVICE_FUNC __forceinline__ static Double2 operator*=(Double2 &a, const double &b) {
    a.x *= b;
    a.y *= b;
    return a;
}

DEVICE_FUNC __forceinline__ static Double2 operator-=(Double2 &a, const Double2 &b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

DEVICE_FUNC __forceinline__ static Double2 operator/(const Double2 &a, const double &b) {
    Double2 c;
    c.x = a.x / b;
    c.y = a.y / b;
    return c;
}

DEVICE_FUNC inline double2 complex_mul(const double2 a, const double2 b) {
    return double2(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

DEVICE_FUNC inline double2 complex_conj(const double2 a) {
    return double2(a.x, -a.y);
}

__device__ __forceinline__ static size_t get_global_id(const int ii) {
    switch (ii) {
    case 0: return threadIdx.x + blockDim.x * blockIdx.x;
    case 1: return threadIdx.y + blockDim.y * blockIdx.y;
    case 2: return threadIdx.z + blockDim.z * blockIdx.z;
    }
    return 0;
}

__device__ __forceinline__ static size_t get_global_size(const int ii) {
    switch (ii) {
    case 0: return blockDim.x * gridDim.x;
    case 1: return blockDim.y * gridDim.y;
    case 2: return blockDim.z * gridDim.z;
    }
    return 0;
}

__device__ __forceinline__ static size_t get_local_id(const int ii) {
    switch (ii) {
    case 0: return threadIdx.x;
    case 1: return threadIdx.y;
    case 2: return threadIdx.z;
    }
    return 0;
}

__device__ __forceinline__ static size_t get_local_size(const int ii) {
    switch (ii) {
    case 0: return blockDim.x;
    case 1: return blockDim.y;
    case 2: return blockDim.z;
    }
    return 0;
}

__host__ double rand_double(double min = -1.0, double max = 1.0);

#define OFFSET_SIZE 6
#define IDX2D(i, j, ld) ((i) + ((j) << (ld)))  // i + j * 2^ld

using rtype = double2;
#define X(x)        z ## x
#define MUL(x, y)   complex_mul(x, y)
#define CONJ(x)     complex_conj(x)
#define REAL(x)     complex_real(x)
#define IMAG(x)     complex_imag(x)

#endif // DEFINITIONS_H
