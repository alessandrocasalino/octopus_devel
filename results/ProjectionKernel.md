## Projection Kernel: projector_bra_phase

| Optimization                       | Viper GPU (AMD MI300A) | Speedup | Raven (Nvidia A100) | Speedup | Notes             |
|------------------------------------|------------------------|---------|---------------------|---------|-------------------|
| Not Optimized                      | **10.426 +/- 1.071**   |         | 11.052 +/- 0.169    |         | Original for HIP  |
| ShMem tiled preload                | 6.949 +/- 0.053        | 1.5x    | Not applicable      |         | Original for CUDA |
| Warp Shuffle optimized             | Not applicable         |         | **2.262 +/- 0.14**  | (4.9x)  | Speed-up wrt Not Optimized (not original CUDA), for reference |                  |
| BLAS with streams (Not Optimized)  | 3.123 +/- 0.052        | 3.3x    | 0.944 +/- 0.005     | 2.4x    |                   |
| BLAS with streams (Optimization 1) | 2.694 +/- 0.011        | 3.9x    | 0.550 +/- 0.001     | 4.1x    |                   |

Bold values are the results of the original kernels, which are not optimized for the respective architectures. If not otherwise specified in the notes, the speedup is calculated with respect to the original kernels.

Results in ms

### Kernel overheads

#### cuBLAS/hipBLAS with streams

| Framework | Total       | w/o streams | w/o streams & mem | w/o streams & mem & BLAS handles |
|-----------|-------------|-------------|-------------------|----------------------------------|
| HIP       | 22.729      | 8.267       | 7.212             | 2.952                            |
| CUDA      | 3.815       | 3.330       | 2.567             | 0.955                            |

Results in ms

## Notes

- HIP version seems not to weak scale, as the BLAS part is not scaling well at least for the data size used.
- The new version of the kernel do not exploit the unified memory of MI300A (but it might not help much anyway in this case).
