#ifdef __HIP_PLATFORM_HCC__
#include <rocblas/rocblas.h>
#define cuBlas_handle rocblas_handle
#define cuBlasCreate(handle) rocblas_create_handle(&(handle))
#define cuBlasDestroy(handle) rocblas_destroy_handle(handle)
#define cuBlasSetStream(handle, stream) rocblas_set_stream((handle), (stream))
#define cuBlas_zgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) \
    rocblas_zgemm((handle), (transa), (transb), (m), (n), (k), (alpha), (A), (lda), (B), (ldb), (beta), (C), (ldc))
#define cuBlas_operation rocblas_operation
#define cuBlas_operation_transpose rocblas_operation_transpose
#define cuBlas_operation_conjugate_transpose rocblas_operation_conjugate_transpose
#define cuBlas_operation_none rocblas_operation_none
#define cuBlas_double_complex rocblas_double_complex
#else
#include <cublas_v2.h>
#define cuBlas_handle cublasHandle_t
#define cuBlasCreate(handle) cublasCreate(&(handle))
#define cuBlasDestroy(handle) cublasDestroy(handle)
#define cuBlasSetStream(handle, stream) cublasSetStream((handle), (stream))
#define cuBlas_zgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) \
    cublasZgemm((handle), (transa), (transb), (m), (n), (k), (alpha), (A), (lda), (B), (ldb), (beta), (C), (ldc))
#define cuBlas_operation cublasOperation_t
#define cuBlas_operation_transpose CUBLAS_OP_T
#define cuBlas_operation_conjugate_transpose CUBLAS_OP_C
#define cuBlas_operation_none CUBLAS_OP_N
#define cuBlas_double_complex cuDoubleComplex
#define cuBlasGetMathMode cublasGetMathMode
#define cuBlasMath_t cublasMath_t
#endif

#ifndef __HIP_PLATFORM_HCC__
template <typename T>
void cuBlas_GetMathMode(const T& handle) {

    cuBlasMath_t mathMode;

    auto status = cuBlasGetMathMode(handle, &mathMode);

    if (status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "Current math mode: ";
        switch (mathMode) {
            case CUBLAS_DEFAULT_MATH:
                std::cout << "CUBLAS_DEFAULT_MATH\n";
                break;
            case CUBLAS_TENSOR_OP_MATH:
                std::cout << "CUBLAS_TENSOR_OP_MATH\n";
                break;
            case CUBLAS_PEDANTIC_MATH:
                std::cout << "CUBLAS_PEDANTIC_MATH\n";
                break;
            case CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION:
                std::cout << "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION\n";
                break;
            default:
                std::cout << "Unknown mode\n";
        }
    }
    else {
        std::cerr << "Failed to get math mode: " << status << std::endl;
    }

}
#endif