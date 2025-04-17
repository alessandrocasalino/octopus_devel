#include "definitions.hpp"


__global__ void projector_bra_phase(const int nmat,
    int const * __restrict offsets,
    rtype const * __restrict matrix,
    int const * __restrict map,
    double const * __restrict scal,
    double2 const * __restrict psi, const int ldpsi,
    double2 * __restrict projection, const int ldprojection,
    double2 const * __restrict phases, const int phases_offset);


__global__ void projector_bra_phase_shmem_opt(const int nmat,
    const int * __restrict const offsets,
    const rtype * __restrict const matrix,
    const int * __restrict const map,
    const double * __restrict const scal,
    const double2 * __restrict const psi, const int ldpsi,
    double2 * __restrict const projection, const int ldprojection,
    const double2 * __restrict const phases, const int phases_offset);

__global__ void projector_bra_phase_gather (
    const int * __restrict const offsets,
    const int * __restrict const map,
    const double2 * __restrict const psi, const int ldpsi,
    const double2 * __restrict const phases, const int phases_offset,
    double2 * __restrict const phasepsi, const int nst_linear,
    int matrix_id);

__global__ void projector_bra_phase_gather_1d_shmem_opt(
    const int * __restrict const offsets,
    const int * __restrict const map,
    const double2 * __restrict const psi, const int ldpsi,
    const double2 * __restrict const phases, const int phases_offset,
    double2 * __restrict const phasepsi, const int nst_linear,
    int matrix_id);

__global__ void projector_bra_phase_mult (
    const int * __restrict const offsets,
    const double * __restrict const scal,
    double2 * __restrict const projection, const int ldprojection,
    const int nst_linear,
    const double2 * __restrict const projection_temp, int matrix_id);
