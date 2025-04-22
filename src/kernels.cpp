#include "kernels.hpp"

__global__ void projector_bra_phase(const int nmat,
    const int * __restrict const offsets,
    const rtype * __restrict const matrix,
    const int * __restrict const map,
    const double * __restrict const scal,
    const double2 * __restrict const psi, const int ldpsi,
    double2 * __restrict const projection, const int ldprojection,
    const double2 * __restrict const phases, const int phases_offset) {
    const int my_warp_size = 1;

    const int ist = get_global_id(0) / my_warp_size;
    const int ipj = get_global_id(1);
    const int imat = get_global_id(2);

    const int npoints = offsets[OFFSET_SIZE * imat + 0];
    const int nprojs = offsets[OFFSET_SIZE * imat + 1];
    const int matrix_offset = offsets[OFFSET_SIZE * imat + 2];
    const int map_offset = offsets[OFFSET_SIZE * imat + 3];
    const int scal_offset = offsets[OFFSET_SIZE * imat + 4];

    if (ipj >= nprojs) return;

    const int nppj = npoints * ipj;

    const int start = 0;
    const int end = npoints;

    double2 aa = 0.0;
    for (int ip = start; ip < end; ip++) {
        double2 phasepsi = complex_mul(phases[phases_offset + map_offset + ip], psi[((map[map_offset + ip] - 1) << ldpsi) + ist]);
        aa += MUL(CONJ(matrix[matrix_offset + ip + nppj]), phasepsi);
    }

    projection[ist + ((scal_offset + ipj) << ldprojection)] = scal[scal_offset + ipj] * aa;
}

#ifdef __HIP_PLATFORM_HCC__
__global__ void projector_bra_phase_opt(const int nmat,
    const int * __restrict const offsets,
    const rtype * __restrict const matrix,
    const int * __restrict const map,
    const double * __restrict const scal,
    const double2 * __restrict const psi, const int ldpsi,
    double2 * __restrict const projection, const int ldprojection,
    const double2 * __restrict const phases, const int phases_offset) {

    const int ist = get_global_id(0);
    const int ipj = get_global_id(1);
    const int imat = get_global_id(2);

    const int npoints = offsets[OFFSET_SIZE * imat + 0];
    const int nprojs = offsets[OFFSET_SIZE * imat + 1];
    const int matrix_offset = offsets[OFFSET_SIZE * imat + 2];
    const int map_offset = offsets[OFFSET_SIZE * imat + 3];
    const int scal_offset = offsets[OFFSET_SIZE * imat + 4];

    if (ipj >= nprojs) return;

    const int nppj = npoints * ipj;

    const int start = 0;
    const int end = npoints;
    
    #ifdef __HIP_PLATFORM_HCC__
    constexpr int smz = 64; // shared memory size
    #else
    constexpr int smz = 32; // shared memory size
    #endif

    __shared__ int shared_map[smz];
    __shared__ double phases_x[smz];
    __shared__ double phases_y[smz];

    double2 aa = 0.0;
    for(int ip = start; ip < end; ip+=smz){
        int lidx = get_local_id(0);
        while(true){
            if(ip + lidx >= end || lidx >= smz) break;
            shared_map[lidx] = map[map_offset + ip + lidx];
            double2 phase = phases[phases_offset + map_offset + ip + lidx];
            phases_x[lidx] = phase.x;
            phases_y[lidx] = phase.y;
            lidx += blockDim.x;
        }
        __syncthreads();

        for(int i = 0; i < smz; i++){
            if(ip+i < end){
                double2 phasepsi = complex_mul({phases_x[i], phases_y[i]}, psi[((shared_map[i] - 1)<<ldpsi) + ist]);
                aa += MUL(CONJ(matrix[matrix_offset + ip+i + nppj]),phasepsi);
            }
        }
    }

    projection[ist + ((scal_offset + ipj) << ldprojection)] = scal[scal_offset + ipj] * aa;
}
#else
__global__ void projector_bra_phase_opt(const int nmat,
    const int * __restrict const offsets,
    const rtype * __restrict const matrix,
    const int * __restrict const map,
    const double * __restrict const scal,
    const double2 * __restrict const psi, const int ldpsi,
    double2 * __restrict const projection, const int ldprojection,
    const double2 * __restrict const phases, const int phases_offset) {

  const int my_warp_size = 32;

  const int ist = get_global_id(0)/my_warp_size;  // the kernel is to be called for (at least) all ist<nst_linear.
  const int ipj = get_global_id(1);
  const int imat = get_global_id(2);

  const int npoints       = offsets[OFFSET_SIZE*imat + 0];
  const int nprojs        = offsets[OFFSET_SIZE*imat + 1];
  const int matrix_offset = offsets[OFFSET_SIZE*imat + 2];
  const int map_offset    = offsets[OFFSET_SIZE*imat + 3];
  const int scal_offset   = offsets[OFFSET_SIZE*imat + 4];

  if(ipj >= nprojs) return;

  const int nppj = npoints*ipj;

  const int slice = npoints%my_warp_size==0 ? npoints/my_warp_size : npoints/my_warp_size+1;
  const int start = slice * (get_local_id(0)%my_warp_size);
  const int end   = min( start + slice, npoints );

  double2 aa = 0.0;
  for(int ip = start; ip < end; ip++){
    double2 phasepsi = complex_mul(phases[phases_offset + map_offset + ip], psi[((map[map_offset + ip] - 1)<<ldpsi) + ist]);
    aa += MUL(CONJ(matrix[matrix_offset + ip + nppj]),phasepsi);
  }

  aa = zwarpReduce(aa);
  if (get_local_id(0)%my_warp_size == 0) 
    projection[ist + ((scal_offset + ipj)<<ldprojection)] = scal[scal_offset + ipj]*aa;
}
#endif


__global__ void projector_bra_phase_gather (
    const int * __restrict const offsets,
    const int * __restrict const map,
    const double2 * __restrict const psi, const int ldpsi,
    const double2 * __restrict const phases, const int phases_offset,
    double2 * __restrict const phasepsi, const int nst_linear,
    int matrix_id) {
    
    const int npoints = offsets[OFFSET_SIZE * matrix_id + 0];
    const int map_offset = offsets[OFFSET_SIZE * matrix_id + 3];

    // C code
    //double2 phasepsi = complex_mul(phases[phases_offset + map_offset + ip], psi[((map[map_offset + ip] - 1) << ldpsi) + ist]);
    
    // If nst_linear is not too large, we can use a 1D grid to parallelize the work
    /*for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < npoints * nst_linear; idx += blockDim.x * gridDim.x) {
        int i = idx % npoints;  // Compute `i` (npoints index)
        int j = idx / npoints;  // Compute `j` (nst_linear index)
    
        if (i < npoints && j < nst_linear) {
            const double2 phases_in = phases[phases_offset + map_offset + i];
            const double2 psi_in = psi[((map[map_offset + i] - 1) << ldpsi) + j];
            phasepsi[npoints * j + i] = complex_mul(phases_in, psi_in);
        }
    }*/

    // If nst_linear is large, we can use a 2D grid to parallelize the work
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < npoints; i += blockDim.x * gridDim.x) {
        for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < nst_linear; j += blockDim.y * gridDim.y) {
            const double2 phases_in = phases[phases_offset + map_offset + i];
            const double2 psi_in = psi[((map[map_offset + i] - 1) << ldpsi) + j];
            //phasepsi[npoints * j + i] = complex_mul(phases_in, psi_in);
            phasepsi[j + i * nst_linear] = complex_mul(phases_in, psi_in);
        }
    }
}


__global__ void projector_bra_phase_mult (
    const int * __restrict const offsets,
    const double * __restrict const scal,
    double2 * __restrict const projection, const int ldprojection,
    const int nst_linear,
    const double2 * __restrict const projection_temp, int matrix_id) {

    const int nprojs = offsets[OFFSET_SIZE * matrix_id + 1];
    const int scal_offset = offsets[OFFSET_SIZE * matrix_id + 4];

    // Fortran code
    // projection%X(projection)(ist, iprojection + iproj) = tmp_proj(iproj, ist)
    // projection%X(projection)(ist, iprojection + iproj) = projection%X(projection)(ist, iprojection + iproj)*pmat%scal(iproj)
    //
    // C code
    // projection[ist + ((scal_offset + ipj) << ldprojection)] = scal[scal_offset + ipj] * aa;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nprojs; i += blockDim.x * gridDim.x) { // Fortran code: iproj
        for (int j = threadIdx.y + blockIdx.y * blockDim.y; j < nst_linear; j += blockDim.y * gridDim.y) { // Fortran code: ist
            projection[j + ((scal_offset + i) << ldprojection)] = scal[scal_offset + i] * projection_temp[i + (j * nprojs)];
        }
    }
}

