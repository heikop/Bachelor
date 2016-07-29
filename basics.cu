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
