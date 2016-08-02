#include "include/vectorgpu.hpp"
#include "include/global.cuh"

//***** per block result to global result *****//

__global__ void block_res_to_glob_res(float* res, float* perblockres, size_t numblocks)
{
    float tmpres{0.0};
    for (size_t i{0}; i < numblocks; ++i)
        tmpres += perblockres[i];
    *res = tmpres;
}

//***** scalar multiplication *****//

__global__ void vector_scal_kernel(const float scalar, float* const values, const size_t size)
{
    size_t id{blockDim.x*blockIdx.x + threadIdx.x};
    if (id < size)
        values[id] *= scalar;
}

__global__ void vector_set_scalar_kernel(const float scalar, float* const values, const size_t size)
{
    size_t id{blockDim.x*blockIdx.x + threadIdx.x};
    if (id < size)
        values[id] = scalar;
}

__global__ void vector_set_zero_kernel(float* const values, const size_t size)
{
    size_t id{blockDim.x*blockIdx.x + threadIdx.x};
    if (id < size)
        values[id] = 0.0;
}

void VectorGpu::scal(const float scalar)
{
    if (scalar != 1.0)
    {
        dim3 numblocks, numthreads;
        get_kernel_config(&numblocks, &numthreads, _size);
        if (scalar == 0.0)
            vector_set_zero_kernel<<<numblocks, numthreads>>>(_values, _size);
        else
            vector_scal_kernel<<<numblocks, numthreads>>>(scalar, _values, _size);
    }
}

//***** axpy *****//

__global__ void vector_add_kernel(const float* const x, float* const y, const size_t size)
{
    size_t id{blockDim.x*blockIdx.x + threadIdx.x};
    if (id < size)
        y[id] += x[id];
}

__global__ void vector_axpy_kernel(const float a, const float* const x, float* const y, const size_t size)
{
    size_t id{blockDim.x*blockIdx.x + threadIdx.x};
    if (id < size)
        y[id] += a*x[id];
}

void VectorGpu::axpy(const float a, const VectorGpu& x)
{
    assert(_size == x._size);
    dim3 numblocks, numthreads;
    get_kernel_config(&numblocks, &numthreads, _size);
    if (a == 1.0)
        vector_add_kernel<<<numblocks, numthreads>>>(x._values, _values, _size);
    else if (a != 0.0)
        vector_axpy_kernel<<<numblocks, numthreads>>>(a, x._values, _values, _size);
}

//***** axpby *****//

__global__ void vector_xpby_kernel(const float* const x, const float b, float* const y, const size_t size)
{
    size_t id{blockDim.x*blockIdx.x + threadIdx.x};
    if (id < size)
        y[id] = b*y[id] + x[id];
}

__global__ void vector_axpby_kernel(const float a, const float* const x, const float b, float* const y, const size_t size)
{
    size_t id{blockDim.x*blockIdx.x + threadIdx.x};
    if (id < size)
        y[id] = b*y[id] + a*x[id];
}

void VectorGpu::axpby(const float a, const VectorGpu& x, const float b)
{
    assert(_size == x._size);
    if (b == 1.0)
        axpy(a, x);
    else if (b == 0.0)
        copyscal(a, x);
    else if (a == 0.0)
        scal(b);
    else
    {
        dim3 numblocks, numthreads;
        get_kernel_config(&numblocks, &numthreads, _size);
        if (a == 1.0)
            vector_xpby_kernel<<<numblocks, numthreads>>>(x._values, b, _values, _size);
        else
            vector_axpby_kernel<<<numblocks, numthreads>>>(a, x._values, b, _values, _size);
    }
}

//***** copy with scalar *****//

void VectorGpu::copyscal(const float scalar, const VectorGpu& other)
{
    assert(_size == other._size);
    if (scalar == 1.0)
        memcpy_cuda(_values, other._values, _size*sizeof(float), d2d);
    else
    {
        dim3 numblocks, numthreads;
        get_kernel_config(&numblocks, &numthreads, _size);
        if (scalar == 0.0)
            vector_set_zero_kernel<<<numblocks, numthreads>>>(_values, _size);
        else
        {} //TODO TODISCUSS memcpy and then scal kernel or copyscal kernel?
            //vector_copyscal_kernel<<<numblocks, numthreads>>>(scalar, _values, _size);
    }
}

//***** dot product *****//

__global__ void vector_dot_vec_kernel(float* const perblockres, const float* const veca, const float* const vecb, const size_t size)
{
    size_t globid{blockDim.x*blockIdx.x + threadIdx.x};
    size_t locid{threadIdx.x};
    extern __shared__ float localresults[];
    if (globid < size)
        localresults[locid] = veca[locid] * vecb[locid];
    else
        localresults[locid] = 0.0;
    __syncthreads();

    for (size_t compressed{blockDim.x/2}; compressed > 0; compressed >>= 1)
    {
        if (locid < compressed)
            localresults[locid] += localresults[locid + compressed];
        __syncthreads();
    }

    if (locid == 0)
        perblockres[blockIdx.x] = localresults[0];
}

float VectorGpu::dot_vec(const VectorGpu& other) const
{
    assert(_size == other._size);
    dim3 numblocks, numthreads;
    float* d_res;
    malloc_cuda(&d_res, sizeof(float));
    get_kernel_config(&numblocks, &numthreads, _size);
    float* perblockres;
    malloc_cuda(&perblockres, numblocks.x*sizeof(float));

    vector_dot_vec_kernel<<<numblocks, numthreads, numthreads.x*sizeof(float)>>>(perblockres, _values, other._values, _size);
    block_res_to_glob_res<<<1, 1>>>(d_res, perblockres, numblocks.x);

    free_cuda(perblockres);

    float h_res{0.0};
    memcpy_cuda(&h_res, d_res, sizeof(float), d2h);
    free_cuda(d_res);
    return h_res;
}

//***** l2norm2 *****//

__global__ void vector_l2norm2_kernel(float* const perblockres, const float* const values, const size_t size)
{
    size_t globid{blockDim.x*blockIdx.x + threadIdx.x};
    size_t locid{threadIdx.x};
    extern __shared__ float localresults[];
    if (globid < size)
        localresults[locid] = values[locid] * values[locid];
    else
        localresults[locid] = 0.0;
    __syncthreads();

    for (size_t compressed{blockDim.x/2}; compressed > 0; compressed >>= 1)
    {
        if (locid < compressed)
            localresults[locid] += localresults[locid + compressed];
        __syncthreads();
    }

    if (locid == 0)
        perblockres[blockIdx.x] = localresults[0];
}

float VectorGpu::l2norm2() const
{
    dim3 numblocks, numthreads;
    float* d_res;
    malloc_cuda(&d_res, sizeof(float));
    get_kernel_config(&numblocks, &numthreads, _size);
    float* perblockres;
    malloc_cuda(&perblockres, numblocks.x*sizeof(float));

    vector_l2norm2_kernel<<<numblocks, numthreads, numthreads.x*sizeof(float)>>>(perblockres, _values, _size);
    block_res_to_glob_res<<<1, 1>>>(d_res, perblockres, numblocks.x);

    float h_res{0.0};
    memcpy_cuda(&h_res, d_res, sizeof(float), d2h);
    free_cuda(d_res);
    return h_res;
}

//***** constructors *****//

VectorGpu::VectorGpu(const size_t size, const float value):
    _size(size)
{
    malloc_cuda(&_values, _size*sizeof(float));
    dim3 numblocks, numthreads;
    get_kernel_config(&numblocks, &numthreads, _size);
    if (value == 0.0)
        vector_set_zero_kernel<<<numblocks, numthreads>>>(_values, _size);
    else
        vector_set_scalar_kernel<<<numblocks, numthreads>>>(value, _values, _size);
}

//TODO
//void set_local(const size_t, const float);
//void add_local(const size_t, const float);
//void set_global(const size_t, const float);
//void add_global(const size_t, const float);
