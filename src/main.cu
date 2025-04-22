#include "definitions.hpp"
#include "kernels.hpp"
#include "blas.hpp"
#include <iostream>
#include <vector>
#include <array>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <random>
#include <algorithm>

constexpr bool debug {true};
constexpr double tolerance {1e-3};

void check_results(const auto &h_projection, const auto &h_projection_comparison) {
    // Compare with the previous results
    for(int i = 0; i < h_projection.size(); ++i){
        const auto& hp = h_projection[i];
        const auto& hpc = h_projection_comparison[i];
        // Compare elements with a tolerance
        if(h_projection[i].x > h_projection_comparison[i].x + tolerance || 
            h_projection[i].x < h_projection_comparison[i].x - tolerance ||
            h_projection[i].y > h_projection_comparison[i].y + tolerance ||
            h_projection[i].y < h_projection_comparison[i].y - tolerance){
                std::cerr << "Results do not match at index " << i << ": "
                << "h_projection = (" << h_projection[i].x << ", " << h_projection[i].y << "), "
                << "h_projection_comparison = (" << h_projection_comparison[i].x << ", " << h_projection_comparison[i].y << ")" << std::endl;
            std::cout << "Results do not match!" << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}


int launch_grid(dim3 grid, dim3 block, int ld, int nst_linear) {

    // Initialize Mersenne Twister random number generator
    std::mt19937 rng(42); // Seed with a fixed value for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0); // Range [-1.0, 1.0]

    // Number of executions
    constexpr int nexec = 10;

    const int nmat = 12;
    const int ldpsi = ld;
    const int ldprojection = ld;

    std::vector<int> h_offsets = {
        1573, 27, 0, 0, 0, 0,
        1573, 27, 42471, 1573, 27, 2916,
        416, 12, 84942, 3146, 54, 5832,
        416, 12, 89934, 3562, 66, 6408,
        416, 12, 94926, 3978, 78, 6984,
        416, 12, 99918, 4394, 90, 7560,
        406, 12, 104910, 4810, 102, 8136,
        406, 12, 109782, 5216, 114, 8712,
        23039, 12, 114654, 5622, 126, 9288,
        23039, 12, 391122, 28661, 138, 9864,
        23071, 12, 667590, 51700, 150, 10440,
        23071, 12, 944442, 74771, 162, 11016
    };

    std::vector<rtype> h_matrix(1221294);
    std::vector<int> h_map(97842);
    std::vector<double> h_scal(174);
    std::vector<rtype> h_psi(53.248 * (1 << ldpsi));

    std::vector<rtype> h_projection(174 * (1 << ldprojection));

    std::vector<rtype> h_projection_comparison(174 * (1 << ldprojection));

    std::vector<rtype> h_phases(2641734);
    constexpr int phase_offset = 0;

    // Initialize host data with random values
    for (auto &v : h_matrix)
        v = {dist(rng), dist(rng)};
    for (auto &s : h_scal)
        s = dist(rng);
    for (auto &p : h_psi)
        p = {dist(rng), dist(rng)};
    for (auto &ph : h_phases)
        ph = {dist(rng), dist(rng)};
    
    std::uniform_int_distribution<int> dist_h(1, ldpsi);
    for (auto &h : h_map)
        h = dist_h(rng);

    // Device allocations
    int *d_offsets, *d_map;
    rtype *d_matrix, *d_psi, *d_projection, *d_phases;
    double *d_scal;

    // Device allocations
    CUDA_CHECK(cuMalloc(&d_offsets, h_offsets.size() * sizeof(int)));
    CUDA_CHECK(cuMalloc(&d_map, h_map.size() * sizeof(int)));
    CUDA_CHECK(cuMalloc(&d_matrix, h_matrix.size() * sizeof(rtype)));
    CUDA_CHECK(cuMalloc(&d_psi, h_psi.size() * sizeof(rtype)));
    CUDA_CHECK(cuMalloc(&d_projection, h_projection.size() * sizeof(rtype)));
    CUDA_CHECK(cuMalloc(&d_phases, h_phases.size() * sizeof(rtype)));
    CUDA_CHECK(cuMalloc(&d_scal, h_scal.size() * sizeof(double)));

    {
        // Launch kernel nexec times and measure execution time
        cuEvent_t start, stop;

        // Create CUDA/HIP events
        CUDA_CHECK(cuEventCreate(&start));
        CUDA_CHECK(cuEventCreate(&stop));

        std::array<float, nexec> execution_times;
        rtype last_result;

        for (int i = 0; i < nexec; ++i) {

            // Copy host to device
            CUDA_CHECK(cuMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_map, h_map.data(), h_map.size() * sizeof(int), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_matrix, h_matrix.data(), h_matrix.size() * sizeof(rtype), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_psi, h_psi.data(), h_psi.size() * sizeof(rtype), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_phases, h_phases.data(), h_phases.size() * sizeof(rtype), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_scal, h_scal.data(), h_scal.size() * sizeof(double), cuMemcpyHostToDevice));
            
            // Reset the projection vector on the device
            CUDA_CHECK(cuMemset(d_projection, 0, h_projection.size() * sizeof(rtype)));
        
            // Record the start event
            CUDA_CHECK(cuEventRecord(start));
        
            // Launch the kernel
            cuLaunchKernel(projector_bra_phase,
                        grid, block,
                        0, 0,
                        nmat,
                        d_offsets,
                        d_matrix,
                        d_map,
                        d_scal,
                        d_psi, ldpsi,
                        d_projection, ldprojection,
                        d_phases, phase_offset);
            
            // Check for kernel launch errors
            CUDA_CHECK(cuGetLastError());
        
            // Record the stop event and synchronize
            CUDA_CHECK(cuEventRecord(stop));
            CUDA_CHECK(cuEventSynchronize(stop));
        
            // Measure the elapsed time
            float milliseconds = 0;
            cuEventElapsedTime(&milliseconds, start, stop);
            execution_times[i] = milliseconds;

            // Copy results back to host
            CUDA_CHECK(cuMemcpy(h_projection.data(), d_projection, h_projection.size() * sizeof(rtype), cuMemcpyDeviceToHost));
            if constexpr (debug) {
                last_result = h_projection[0];  // Store the last result for debugging
                std::cout << "\n[Debug] Last result from projection: " 
                        << last_result.x << ", " 
                        << last_result.y << std::endl;
                std::cout << "[Profiling] Execution time for kernel run " << i + 1 << ": "
                        << milliseconds << " ms" << std::endl;
            }
        }

        // Calculate the mean execution time
        float mean_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0f) / execution_times.size();
        // Calculate the standard deviation
        float sum_squared_diff = 0.0f;
        for (const auto &time : execution_times) {
            sum_squared_diff += (time - mean_time) * (time - mean_time);
        }
        float stddev = std::sqrt(sum_squared_diff / nexec);
        std::cout << "\n[Profiling] Mean projector_bra_phase execution time over " << nexec << " runs: " << mean_time << " ms\n";
        std::cout << "\n[Profiling] Standard deviation of projector_bra_phase execution time: " << stddev << " ms\n";

        std::cout << "Single executions times: " << std::endl;
        for (int i = 0; i < nexec; ++i) {
            std::cout << "Execution " << i + 1 << ": " << execution_times[i] << " ms" << std::endl;
        }
        std::cout << "Minimum execution time: " << *std::min_element(execution_times.begin(), execution_times.end()) << " ms" << std::endl;
        std::cout << "Maximum execution time: " << *std::max_element(execution_times.begin(), execution_times.end()) << " ms" << std::endl;

        // Copy results back to host
        CUDA_CHECK(cuMemcpy(h_projection.data(), d_projection, h_projection.size() * sizeof(rtype), cuMemcpyDeviceToHost));
        CUDA_CHECK(cuMemcpy(h_projection_comparison.data(), d_projection, h_projection_comparison.size() * sizeof(rtype), cuMemcpyDeviceToHost));

        // Print a few values from the result
        if constexpr (debug) {
            std::cout << "\n[Debug] Sample output from projection:" << std::endl;
            for (int i = 0; i < std::min(10, static_cast<int>(h_projection.size())); ++i) {
                std::cout << "h_projection[" << i << "] = (" 
                        << h_projection[i].x << ", " 
                        << h_projection[i].y << ")" << std::endl;
            }
            for (int i = 0; i < std::min(10, static_cast<int>(h_projection_comparison.size())); ++i) {
                std::cout << "h_projection_comparison[" << i << "] = (" 
                        << h_projection_comparison[i].x << ", " 
                        << h_projection_comparison[i].y << ")" << std::endl;
            }
        }

        CUDA_CHECK(cuEventDestroy(start));
        CUDA_CHECK(cuEventDestroy(stop));
    }



    {
        // Launch kernel nexec times and measure execution time
        cuEvent_t start, stop;

        // Create CUDA/HIP events
        CUDA_CHECK(cuEventCreate(&start));
        CUDA_CHECK(cuEventCreate(&stop));

        std::array<float, nexec> execution_times;
        rtype last_result;

        for (int i = 0; i < nexec; ++i) {

            // Copy host to device
            CUDA_CHECK(cuMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_map, h_map.data(), h_map.size() * sizeof(int), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_matrix, h_matrix.data(), h_matrix.size() * sizeof(rtype), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_psi, h_psi.data(), h_psi.size() * sizeof(rtype), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_phases, h_phases.data(), h_phases.size() * sizeof(rtype), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_scal, h_scal.data(), h_scal.size() * sizeof(double), cuMemcpyHostToDevice));

            // Reset the projection vector on the device
            CUDA_CHECK(cuMemset(d_projection, 0, h_projection.size() * sizeof(rtype)));
        
            // Record the start event
            CUDA_CHECK(cuEventRecord(start));
        
            // Launch the kernel
            cuLaunchKernel(projector_bra_phase_opt,
                        grid, block,
                        0, 0,
                        nmat,
                        d_offsets,
                        d_matrix,
                        d_map,
                        d_scal,
                        d_psi, ldpsi,
                        d_projection, ldprojection,
                        d_phases, phase_offset);
            
            // Check for kernel launch errors
            CUDA_CHECK(cuGetLastError());
        
            // Record the stop event and synchronize
            CUDA_CHECK(cuEventRecord(stop));
            CUDA_CHECK(cuEventSynchronize(stop));
        
            // Measure the elapsed time
            float milliseconds = 0;
            cuEventElapsedTime(&milliseconds, start, stop);
            execution_times[i] = milliseconds;

            // Copy results back to host
            CUDA_CHECK(cuMemcpy(h_projection.data(), d_projection, h_projection.size() * sizeof(rtype), cuMemcpyDeviceToHost));
            if constexpr (debug) {
                last_result = h_projection[0];  // Store the last result for debugging
                std::cout << "\n[Debug] Last result from projection: " 
                        << last_result.x << ", " 
                        << last_result.y << std::endl;
                std::cout << "[Profiling] Execution time for kernel run " << i + 1 << ": "
                        << milliseconds << " ms" << std::endl;
            }
        }

        // Calculate the mean execution time
        float mean_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0f) / execution_times.size();
        // Calculate the standard deviation
        float sum_squared_diff = 0.0f;
        for (const auto &time : execution_times) {
            sum_squared_diff += (time - mean_time) * (time - mean_time);
        }
        float stddev = std::sqrt(sum_squared_diff / nexec);
        std::cout << "\n[Profiling] Mean projector_bra_phase_shmem_opt execution time over " << nexec << " runs: " << mean_time << " ms\n";
        std::cout << "\n[Profiling] Standard deviation of projector_bra_phase_shmem_opt execution time: " << stddev << " ms\n";

        std::cout << "Single executions times: " << std::endl;
        for (int i = 0; i < nexec; ++i) {
            std::cout << "Execution " << i + 1 << ": " << execution_times[i] << " ms" << std::endl;
        }
        std::cout << "Minimum execution time: " << *std::min_element(execution_times.begin(), execution_times.end()) << " ms" << std::endl;
        std::cout << "Maximum execution time: " << *std::max_element(execution_times.begin(), execution_times.end()) << " ms" << std::endl;

        // Copy results back to host
        CUDA_CHECK(cuMemcpy(h_projection.data(), d_projection, h_projection.size() * sizeof(rtype), cuMemcpyDeviceToHost));

        // Print a few values from the result
        if constexpr (debug) {
            std::cout << "\n[Debug] Sample output from projection:" << std::endl;
            for (int i = 0; i < std::min(10, static_cast<int>(h_projection.size())); ++i) {
                std::cout << "h_projection[" << i << "] = (" 
                        << h_projection[i].x << ", " 
                        << h_projection[i].y << ")" << std::endl;
            }
        }

        // Compare with the previous results
        check_results(h_projection, h_projection_comparison);

        CUDA_CHECK(cuEventDestroy(start));
        CUDA_CHECK(cuEventDestroy(stop));
    }



    {
        // Launch kernel nexec times and measure execution time
        cuEvent_t start, stop;
        cuEvent_t init, temp_allocation, kernel1, handles_init, blas, handles_destr, kernel2, destr;

        // Create CUDA/HIP events
        CUDA_CHECK(cuEventCreate(&start));
        CUDA_CHECK(cuEventCreate(&stop));
        CUDA_CHECK(cuEventCreate(&init));
        CUDA_CHECK(cuEventCreate(&temp_allocation));
        CUDA_CHECK(cuEventCreate(&kernel1));
        CUDA_CHECK(cuEventCreate(&handles_init));
        CUDA_CHECK(cuEventCreate(&blas));
        CUDA_CHECK(cuEventCreate(&handles_destr));
        CUDA_CHECK(cuEventCreate(&kernel2));
        CUDA_CHECK(cuEventCreate(&destr));

        std::array<float, nexec> execution_times;
        rtype last_result;

        for (int ex = 0; ex < nexec + 1; ++ex) {

            // Copy host to device
            CUDA_CHECK(cuMemcpy(d_offsets, h_offsets.data(), h_offsets.size() * sizeof(int), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_map, h_map.data(), h_map.size() * sizeof(int), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_matrix, h_matrix.data(), h_matrix.size() * sizeof(rtype), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_psi, h_psi.data(), h_psi.size() * sizeof(rtype), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_phases, h_phases.data(), h_phases.size() * sizeof(rtype), cuMemcpyHostToDevice));
            CUDA_CHECK(cuMemcpy(d_scal, h_scal.data(), h_scal.size() * sizeof(double), cuMemcpyHostToDevice));

            // Record the start event
            CUDA_CHECK(cuEventRecord(start));

            // Create an array of streams
            int num_streams = nmat;
            cuStream_t *streams = new cuStream_t[num_streams];
            for (int i = 0; i < num_streams; ++i) {
                CUDA_CHECK(cuStreamCreate(streams[i]));
            }

            CUDA_CHECK(cuEventRecord(init));

            // Temporary vector for phasepsi results and matrix multiplications (only on device)
            double2 **d_phasepsi = new double2*[num_streams];
            double2 **d_projection_int = new double2*[num_streams];
            for (int i = 0; i < num_streams; ++i) {
                int npoints = h_offsets[i * OFFSET_SIZE + 0];
                int nprojs = h_offsets[i * OFFSET_SIZE + 1];
                
                CUDA_CHECK(cuMallocAsync((void**)&d_phasepsi[i], npoints * nst_linear * sizeof(double2), streams[i]));
                CUDA_CHECK(cuMemsetAsync(d_phasepsi[i], 0, npoints * nst_linear * sizeof(double2), streams[i]));

                CUDA_CHECK(cuMallocAsync((void**)&d_projection_int[i], nprojs * nst_linear * sizeof(double2), streams[i]));
                CUDA_CHECK(cuMemsetAsync(d_projection_int[i], 0, nprojs * nst_linear * sizeof(double2), streams[i]));
            }
            
            CUDA_CHECK(cuEventRecord(temp_allocation));
            
            for (int i = 0; i < num_streams; ++i) {
                int npoints = h_offsets[i * OFFSET_SIZE + 0];

                const dim3 block {64, 2};
                const dim3 grid {(npoints + block.x - 1) / block.x, (nst_linear + block.y - 1) / block.y};
                // Launch the gather kernel
                cuLaunchKernel(
                    projector_bra_phase_gather,
                    grid, block,
                    0, streams[i],
                    d_offsets,
                    d_map,
                    d_psi, ldpsi,
                    d_phases, phase_offset,
                    d_phasepsi[i],
                    nst_linear,
                    i
                );
                CUDA_CHECK(cuGetLastError());
            }
            
            CUDA_CHECK(cuEventRecord(kernel1));

            cuBlas_double_complex M_z0 = {0., 0.};
            cuBlas_double_complex M_z1 = {1., 0.};
            
            // Create cuBLAS handles
            cuBlas_handle *handles = new cuBlas_handle[num_streams];
            for (int i = 0; i < num_streams; ++i) {
                if (auto blas_status = cuBlasCreate(handles[i]); blas_status != 0) {
                    std::cerr << "Error creating cuBLAS handle: " << blas_status << std::endl;
                    exit(EXIT_FAILURE);
                }

                if (auto blas_status = cuBlasSetStream(handles[i], streams[i]); blas_status != 0) {
                    std::cerr << "Error setting stream for cuBLAS: " << blas_status << std::endl;
                    exit(EXIT_FAILURE);
                }
            }

            CUDA_CHECK(cuEventRecord(handles_init));

            // Perform matrix multiplication on stream
            for (int i = 0; i < num_streams; ++i) {
                // Fortran code
                //call blas_gemm('C', 'T', nprojs, nst_linear, npoints, &
                //    M_z1, pmat%zprojectors(1, 1), npoints, lpsi(1, 1), nst_linear, M_z0, tmp_proj(1,1), nprojs)
                // Macro
                // cuBlas_gemm (handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
                
                int npoints = h_offsets[i * OFFSET_SIZE + 0];
                int nprojs = h_offsets[i * OFFSET_SIZE + 1];
                int matrix_offset = h_offsets[i * OFFSET_SIZE + 2];
                
                cuBlas_zgemm(handles[i], cuBlas_operation_conjugate_transpose, cuBlas_operation_transpose,
                    nprojs, nst_linear, npoints,
                    &M_z1, reinterpret_cast<cuBlas_double_complex *>(&d_matrix[matrix_offset]), npoints,
                    reinterpret_cast<cuBlas_double_complex *>(d_phasepsi[i]), nst_linear,
                    &M_z0,
                    reinterpret_cast<cuBlas_double_complex *>(d_projection_int[i]), nprojs);
                CUDA_CHECK(cuGetLastError());
            }

            // Synchronize all streams
            for (int i = 0; i < num_streams; ++i) {
                CUDA_CHECK(cuStreamSynchronize(streams[i]));
            }

            CUDA_CHECK(cuEventRecord(blas));
            
            // Destroy cuBLAS handles
            for (int i = 0; i < num_streams; ++i) {
                if (auto blas_status = cuBlasDestroy(handles[i]); blas_status != 0) {
                    std::cerr << "Error destroying cuBLAS handle: " << blas_status << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
            delete[] handles;

            CUDA_CHECK(cuEventRecord(handles_destr));
            
            for (int i = 0; i < num_streams; ++i) {
                int npoints = h_offsets[i * OFFSET_SIZE + 0];
                int nprojs = h_offsets[i * OFFSET_SIZE + 1];
                int matrix_offset = h_offsets[i * OFFSET_SIZE + 2];

                const dim3 block{128, 2};
                const dim3 grid {(nprojs + block.x - 1) / block.x, (nst_linear + block.y - 1) / block.y};
                // Final scaling
                cuLaunchKernel(
                    projector_bra_phase_mult,
                    grid, block,
                    0, streams[i],
                    d_offsets,
                    d_scal,
                    d_projection, ldprojection,
                    nst_linear,
                    d_projection_int[i],
                    i
                );
                CUDA_CHECK(cuGetLastError());
            }

            // Synchronize all streams
            for (int i = 0; i < num_streams; ++i) {
                CUDA_CHECK(cuStreamSynchronize(streams[i]));
            }
            
            CUDA_CHECK(cuEventRecord(kernel2));

            // Cleanup temporary variables
            for (int i = 0; i < num_streams; ++i) {
                CUDA_CHECK(cuFreeAsync(d_phasepsi[i], streams[i])); // Free memory for each stream
                CUDA_CHECK(cuFreeAsync(d_projection_int[i], streams[i])); // Free memory for each stream
            }
            delete[] d_phasepsi;
            delete[] d_projection_int;

            CUDA_CHECK(cuEventRecord(destr));

            // Cleanup
            for (int i = 0; i < num_streams; ++i) {
                CUDA_CHECK(cuStreamDestroy(streams[i]));
            }
            delete[] streams;
        
            // Record the stop event and synchronize
            CUDA_CHECK(cuEventRecord(stop));
            CUDA_CHECK(cuEventSynchronize(stop));
        
            // Measure the elapsed time
            float milliseconds = 0;
            cuEventElapsedTime(&milliseconds, start, stop);
            if(ex != 0){
                execution_times[ex - 1] = milliseconds;
            }

            // Copy results back to host
            CUDA_CHECK(cuMemcpy(h_projection.data(), d_projection, h_projection.size() * sizeof(rtype), cuMemcpyDeviceToHost));
            if constexpr (debug) {
                last_result = h_projection[0];  // Store the last result for debugging
                std::cout << "\n[Debug] Last result from projection: " 
                        << last_result.x << ", " 
                        << last_result.y << std::endl;
                std::cout << "[Profiling] Execution time for kernel run " << ex + 1 << ": "
                        << milliseconds << " ms" << std::endl;
                
                float elapsed0 = 0;
                cuEventElapsedTime(&elapsed0, start, init);
                std::cout << "[Profiling] Execution time for streams initialization " << ex + 1 << ": "
                        << elapsed0 << " ms" << std::endl;
                
                float elapsed1 = 0;
                cuEventElapsedTime(&elapsed1, init, temp_allocation);
                std::cout << "[Profiling] Execution time for data allocation " << ex + 1 << ": "
                        << elapsed1 << " ms" << std::endl;
                
                float elapsed2 = 0;
                cuEventElapsedTime(&elapsed2, temp_allocation, kernel1);
                std::cout << "[Profiling] Execution time for kernel1 " << ex + 1 << ": "
                        << elapsed2 << " ms" << std::endl;
                
                float elapsed3_0 = 0;
                cuEventElapsedTime(&elapsed3_0, kernel1, handles_init);
                std::cout << "[Profiling] Execution time for blas handles initialization " << ex + 1 << ": "
                        << elapsed3_0 << " ms" << std::endl;
                
                float elapsed3 = 0;
                cuEventElapsedTime(&elapsed3, handles_init, blas);
                std::cout << "[Profiling] Execution time for blas " << ex + 1 << ": "
                        << elapsed3 << " ms" << std::endl;
                
                float elapsed3_1 = 0;
                cuEventElapsedTime(&elapsed3_1, blas, handles_destr);
                std::cout << "[Profiling] Execution time for blas handles destruction " << ex + 1 << ": "
                        << elapsed3_1 << " ms" << std::endl;
                
                float elapsed4 = 0;
                cuEventElapsedTime(&elapsed4, blas, kernel2);
                std::cout << "[Profiling] Execution time for kernel2 " << ex + 1 << ": "
                        << elapsed4 << " ms" << std::endl;
                
                        float elapsed5 = 0;
                cuEventElapsedTime(&elapsed5, kernel2, destr);
                std::cout << "[Profiling] Execution time for data freeing " << ex + 1 << ": "
                        << elapsed5 << " ms" << std::endl;
                
                float elapsed6 = 0;
                cuEventElapsedTime(&elapsed6, destr, stop);
                std::cout << "[Profiling] Execution time for streams destruction " << ex + 1 << ": "
                        << elapsed6 << " ms" << std::endl;
                
                std::cout << "[Profiling] Execution time without streams initializations " << ex + 1 << ": "
                        << milliseconds - elapsed0 - elapsed6 << " ms" << std::endl;
                std::cout << "[Profiling] Execution time without memory allocation " << ex + 1 << ": "
                        << milliseconds - elapsed0 - elapsed1 - elapsed5- elapsed6 << " ms" << std::endl;
                std::cout << "[Profiling] Execution time without memory and handles management" << ex + 1 << ": "
                        << milliseconds - elapsed0 - elapsed1 - elapsed3_0 - elapsed3_1 - elapsed5 - elapsed6 << " ms" << std::endl;
                if(ex != 0){
                    execution_times[ex - 1] = milliseconds - elapsed0 - elapsed1 - elapsed3_0 - elapsed3_1 - elapsed5- elapsed6;
                }
            }
        }

        // Calculate the mean execution time
        float mean_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0f) / execution_times.size();
        // Calculate the standard deviation
        float sum_squared_diff = 0.0f;
        for (const auto &time : execution_times) {
            sum_squared_diff += (time - mean_time) * (time - mean_time);
        }
        float stddev = std::sqrt(sum_squared_diff / nexec);
        std::cout << "\n[Profiling] Mean projector_bra_phase_shmem_opt execution time over " << nexec << " runs: " << mean_time << " ms\n";
        std::cout << "\n[Profiling] Standard deviation of projector_bra_phase_shmem_opt execution time: " << stddev << " ms\n";

        std::cout << "Single executions times: " << std::endl;
        for (int i = 0; i < nexec; ++i) {
            std::cout << "Execution " << i + 1 << ": " << execution_times[i] << " ms" << std::endl;
        }
        std::cout << "Minimum execution time: " << *std::min_element(execution_times.begin(), execution_times.end()) << " ms" << std::endl;
        std::cout << "Maximum execution time: " << *std::max_element(execution_times.begin(), execution_times.end()) << " ms" << std::endl;

        // Copy results back to host
        CUDA_CHECK(cuMemcpy(h_projection.data(), d_projection, h_projection.size() * sizeof(rtype), cuMemcpyDeviceToHost));

        // Print a few values from the result
        if constexpr (debug) {
            std::cout << "\n[Debug] Sample output from projection:" << std::endl;
            for (int i = 0; i < std::min(10, static_cast<int>(h_projection.size())); ++i) {
                std::cout << "h_projection[" << i << "] = (" 
                        << h_projection[i].x << ", " 
                        << h_projection[i].y << ")" << std::endl;
            }
        }

        // Compare with the previous results
        check_results(h_projection, h_projection_comparison);

        CUDA_CHECK(cuEventDestroy(start));
        CUDA_CHECK(cuEventDestroy(stop));
        CUDA_CHECK(cuEventDestroy(init));
        CUDA_CHECK(cuEventDestroy(temp_allocation));
        CUDA_CHECK(cuEventDestroy(kernel1));
        CUDA_CHECK(cuEventDestroy(blas));
        CUDA_CHECK(cuEventDestroy(kernel2));
        CUDA_CHECK(cuEventDestroy(destr));
    }



    // Cleanup
    CUDA_CHECK(cuFree(d_offsets));
    CUDA_CHECK(cuFree(d_map));
    CUDA_CHECK(cuFree(d_matrix));
    CUDA_CHECK(cuFree(d_psi));
    CUDA_CHECK(cuFree(d_projection));
    CUDA_CHECK(cuFree(d_phases));
    CUDA_CHECK(cuFree(d_scal));

    return 0;
}


int main() {

    int device_id;
    if (cuGetDevice(&device_id) != 0) {
        std::cerr << "Error: No suitable device found or failed to get the device ID." << std::endl;
        return EXIT_FAILURE;
    }

    cuDeviceProp device_prop;
    if (cuGetDeviceProperties(&device_prop, device_id) != 0) {
        std::cerr << "Error: Failed to get device properties for device ID " << device_id << "." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "\n[Device Info]" << std::endl;
    std::cout << "Using device ID: " << device_id << std::endl;
    std::cout << "Device name: " << device_prop.name << std::endl;
    std::cout << "Compute capability: " << device_prop.major << "." << device_prop.minor << std::endl;
    std::cout << "Global memory: " << (device_prop.totalGlobalMem >> 20) << " MB" << std::endl;
    std::cout << "Shared memory per block: " << (device_prop.sharedMemPerBlock >> 10) << " KB" << std::endl;
    std::cout << "Multiprocessors: " << device_prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per block: " << device_prop.maxThreadsPerBlock << std::endl;
    
    {
        // Define the grid and block dimensions
        dim3 grid(64, 4, 12);  // Example grid dimensions
        dim3 block(32, 8, 1);  // Example block dimensions
        launch_grid(grid, block, 7, 128);
    }

    {
        // Define the grid and block dimensions
        dim3 grid(32, 4, 12);  // Example grid dimensions
        dim3 block(32, 8, 1);  // Example block dimensions
        launch_grid(grid, block, 5, 20);
    }

    return 0;
}
