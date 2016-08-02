#include "include/csrmatrixgpu.hpp"
#include "include/global.cuh"

__global__ void csrmatrix_multvec_kernel(const size_t* const rowptr, const size_t* const colind, const float* const values, const float* const vec, float* const res, size_t numrows)
{
    size_t row{blockDim.x*blockIdx.x + threadIdx.x};
    if (row < numrows)
    {
        res[row] = 0.0;
        for (size_t col{rowptr[row]}; col < rowptr[row+1]; ++col)
            res[row] += values[col] * vec[colind[col]];
    }
}

void CsrMatrixGpu::multvec(const VectorGpu& vec, VectorGpu& res) const
{
    assert(_numcols == vec._size && _numrows == res._size); //TODISCUSS or reallocate when res has a different size?
    dim3 numblocks, numthreads;
    get_kernel_config(&numblocks, &numthreads, _numrows);
    csrmatrix_multvec_kernel<<<numblocks, numthreads>>>(_rowptr, _colind, _values, vec._values, res._values, _numrows);
    cudaDeviceSynchronize();
}
