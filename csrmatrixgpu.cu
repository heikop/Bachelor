#include "include/csrmatrixgpu.hpp"
#include "include/global.cuh"

//***** multvec *****//

__global__ void csrmatrix_multvec_kernel(const size_t* const rowptr, const size_t* const colind, const float* const values, const float* const vec, float* const res, size_t numrows)
{
    size_t row{blockDim.x*blockIdx.x + threadIdx.x};
    float tmpres{0.0};
    if (row < numrows)
        for (size_t col{rowptr[row]}; col < rowptr[row+1]; ++col)
            tmpres += values[col] * vec[colind[col]];
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
    size_t id{blockDim.x*blockIdx.x + threadIdx.x};
    if (id < num_values)
        values[id] = 0.0;
}

void CsrMatrixGpu::createStructure(const Triangle* const elements, const size_t num_elem)
{
    std::vector<std::vector<size_t>> lol(_numcols, std::vector<size_t>(0));
    for (size_t i(0); i < num_elem; ++i)
    {
        size_t nodes[]{elements[i].nodeA, elements[i].nodeB, elements[i].nodeC};
        for (size_t node1(0); node1 < 3; ++node1)
        {
            for (size_t node2(0); node2 < 3; ++node2)
            {
                size_t a{nodes[node1]};
                size_t b{nodes[node2]};
                size_t j{0};
                while (j < lol[a].size() && lol[a][j] < b)
                    ++j;
                if (j == lol[a].size())
                    lol[a].push_back(b);
                else if (lol[a][j] != b)
                    lol[a].insert(lol[a].begin()+j, b);
            }
        }
    }

    size_t* h_rowptr = new size_t[_numrows+1];
    size_t num_values(0);
    for (size_t i(0); i < _numrows; ++i)
    {
        h_rowptr[i] = num_values;
        num_values += lol[i].size();
    }
    h_rowptr[_numrows] = num_values;

    free_cuda(_colind);
    malloc_cuda(&_colind, num_values*sizeof(size_t));
    size_t current_pos{0};
    size_t* h_colind = new size_t[num_values];
    for (const auto& row : lol)
        for (const auto col : row)
            h_colind[current_pos++] = col;
    memcpy_cuda(_colind, h_colind, num_values*sizeof(size_t), h2d);

    free_cuda(_values);
    malloc_cuda(&_values, num_values*sizeof(float));
    float* h_values = new float[num_values];
    for (size_t i(0); i < num_values; ++i)
        h_values[i] = 0.0;
    memcpy_cuda(_values, h_values, num_values*sizeof(float), h2d);
// if then not essentially faster
//    dim3 numblocks, numthreads;
//    get_kernel_config(&numblocks, &numthreads, num_values);
//    set_zeros<<<numblocks, numthreads>>>(_values, num_values);
//    cudaDeviceSynchronize();

    memcpy_cuda(_rowptr, h_rowptr, (_numrows+1)*sizeof(size_t), h2d);
    cudaDeviceSynchronize(); // needed?

    delete[] h_rowptr;
//    delete[] h_colind;
//    delete[] h_values;
}
