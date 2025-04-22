## Projection Kernel

| Optimization                     | Viper GPU (AMD MI300A) | Raven (Nvidia A100) |
|----------------------------------|------------------------|---------------------|
| Original                         | 10.426 +/- 1.071       | 11.052 +/- 0.169    |
| ShMem tiled preload              | 6.949 +/- 0.053        | Not applicable      |
| Warp Shuffle optimized           | Not applicable         | 2.262 +/- 0.14      |
| BLAS with streams (Not Optimized)| 3.123 +/- 0.052        | 0.944 +/- 0.005     |

Results in ms

### Kernel overheads

#### cuBLAS/hipBLAS with streams

| Framework | Total       | w/o streams | w/o streams & mem | w/o streams & mem & BLAS handles |
|-----------|-------------|-------------|-------------------|----------------------------------|
| HIP       | 22.729      | 8.267       | 7.212             | 2.952                            |
| CUDA      | 3.815       | 3.330       | 2.567             | 0.955                            |

Results in ms

