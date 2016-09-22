#include "include/csrmatrixgpu.hpp"
#include "include/global.cuh"

//***** multvec *****//

__global__ void csrmatrix_multvec_kernel(const size_t* const _rowptr, const size_t* const _colind, const float* const _values, const float* const vec, float* const res, size_t _numrows)
{
    size_t row(blockDim.x*blockIdx.x + threadIdx.x);
    float tmpres(0.0);
    if (row < _numrows)
        for (size_t col(_rowptr[row]); col < _rowptr[row+1]; ++col)
            tmpres += _values[col] * vec[_colind[col]];
    res[row] = tmpres;
}

void CsrMatrixGpu::multvec(const VectorGpu& vec, VectorGpu& res) const
{
    assert(_numcols == vec._size && _numrows == res._size); //TODISCUSS or reallocate when res has a different size?
    dim3 numblocks, numthreads;
    get_kernel_config(&numblocks, &numthreads, _numrows);
    csrmatrix_multvec_kernel<<<numblocks, numthreads>>>(_rowptr, _colind, _values, vec._values, res._values, _numrows);
    cudaDeviceSynchronize();
}

//***** create structure *****//

__global__ void set_zeros(float* values, size_t num_values)
{
    size_t id(blockDim.x*blockIdx.x + threadIdx.x);
    if (id < num_values)
        values[id] = 0.0;
}

//***** get *****//

__global__ void get_global_kernel(const size_t* const _rowptr, const size_t* const _colind, const float* const _values, const size_t row, const size_t col, float* val)
{
    size_t pos_to_get(_rowptr[row]);
    while (_colind[pos_to_get] < col && pos_to_get < _rowptr[row+1])
        ++pos_to_get;
    if (_colind[pos_to_get] == col)
        *val = _values[pos_to_get];
}

float CsrMatrixGpu::get_global(const size_t row, const size_t col) const
{
    assert(row < _numrows && col < _numcols);
    float h_res(0.0);
    float* d_res; malloc_cuda(&d_res, sizeof(float));
    memcpy_cuda(d_res, &h_res, sizeof(float), h2d);
    get_global_kernel<<<1,1>>>(_rowptr, _colind, _values, row, col, d_res);
    memcpy_cuda(&h_res, d_res, sizeof(float), d2h);
    return h_res;
}
