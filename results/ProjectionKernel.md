## Projection Kernel

| Optimization                     | Viper GPU (AMD MI300A) | Raven (Nvidia A100) |
|----------------------------------|------------------------|---------------------|
| Original                         | 10.426 +/- 1.071       | 3.165               |
| ShMem tiled preload              | 6.949 +/- 0.053        | None                |
| BLAS with streams (Not Optimized)| 3.123 +/- 0.052        | TBD                 |

Results in ms

### Kernel overheads

#### cuBLAS/hipBLAS with streams

| Total       | w/o streams | w/o streams & mem | w/o streams & mem & BLAS handles |
|-------------|-------------|-------------------|----------------------------------|
| 22.7292 ms  | 8.26738 ms  | 7.21191 ms        | 2.952 ms                         |

Results in ms

