#ifndef __GLOBAL_HPP_
#define __GLOBAL_HPP_

enum memcpyKind {h2d, d2h, d2d, h2h};

struct Node { size_t ID; float x; float y; };
struct Triangle1 { size_t ID; size_t nodeA; size_t nodeB; size_t nodeC; };
struct FullTriangle { size_t ID; Node nodeA; Node nodeB; Node nodeC; };

void initCuda();
template<typename scalar> void malloc_cuda(scalar** devPtr, size_t size);
template<typename scalar> void free_cuda(scalar* devPtr);
void memcpy_cuda(void* dst, const void* src, size_t count, memcpyKind kind);
//void get_kernel_config(void* const numblocks, void* const numthreads, size_t totalthreads); // both void pointers have to point to a dim3 variable

#endif
