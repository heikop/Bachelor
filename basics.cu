#include <cassert>

#include "include/global.hpp"

template<typename scalar>
void malloc_cuda(scalar** devPtr, size_t size)
{
    cudaMalloc(devPtr, size);
}
template void malloc_cuda<size_t>(size_t**, size_t);
template void malloc_cuda<float>(float**, size_t);
template void malloc_cuda<double>(double**, size_t);
template void malloc_cuda<FullTriangle>(FullTriangle**, size_t);

template<typename scalar>
void free_cuda(scalar* devPtr)
{
    cudaFree(devPtr);
}
template void free_cuda<size_t>(size_t*);
template void free_cuda<float>(float*);
template void free_cuda<double>(double*);
template void free_cuda<FullTriangle>(FullTriangle*);

void memcpy_cuda(void* dst, const void* src, size_t count, memcpyKind kind)
{
    switch(kind)
    {
    case h2d: cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice); break;
    case d2h: cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost); break;
    //default: //error
    }
}

void get_kernel_config(dim3* const numblocks, dim3* const numthreads, size_t totalthreads) // pointers declared as void pointers in global.hpp
{
    int devCount;
    cudaGetDeviceCount(&devCount);
    assert(devCount > 0);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    *numthreads = dim3(props.maxThreadsDim[0], 1, 1);
    *numblocks = dim3(totalthreads / (*numthreads).x + (totalthreads % (*numthreads).x == 0 ? 0 : 1), 1, 1);
}
