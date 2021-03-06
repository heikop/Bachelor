#include <cassert>

#include "include/global.hpp"
#include <iostream>

void initCuda()
{
    int* dummy;
    cudaMalloc(&dummy, 100*sizeof(int));
    cudaFree(dummy);
    cudaDeviceSynchronize();
}

template<typename scalar> void malloc_cuda(scalar** devPtr, size_t size)
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
    case d2d: cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice); break;
    case h2h: cudaMemcpy(dst, src, count, cudaMemcpyHostToHost); break;
    default: assert(false); //error
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
    //*numthreads = dim3(64, 1, 1);
    //*numthreads = dim3(128, 1, 1);
    //*numthreads = dim3(256, 1, 1);
    *numblocks = dim3(totalthreads / (*numthreads).x + (totalthreads % (*numthreads).x == 0 ? 0 : 1), 1, 1);
    //std::cout << "shared mem per block: " << props.sharedMemPerBlock << std::endl;
    while (numblocks->x > 65535)
    {
        numblocks->x = numblocks->x / 2 + numblocks->x % 2;
        numblocks->y *= 2;
    }

    //std::cout << "max grid size: " << props.maxGridSize[0] << std::endl;
    assert(numblocks->x < props.maxGridSize[0]);
    assert((*numblocks).y < props.maxGridSize[1]);
}
