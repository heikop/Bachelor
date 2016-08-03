#include "include/global.hpp"
#include "include/global.cuh"
#include <cassert>

__device__ void add_local_atomic(const size_t* const rowptr, const size_t* const colind, float* const values, const size_t row, const size_t col, const float val)
{
    size_t pos_to_insert(rowptr[row]);
    while (colind[pos_to_insert] < col && pos_to_insert < rowptr[row+1])
        ++pos_to_insert;
    atomicAdd(values + pos_to_insert, val);
}

__global__ void atomic(const size_t* const rowptr, const size_t* const colind, float* const values, const size_t numrows, const FullTriangle* const elements, const size_t numelem)
{
    size_t pos_of_elem(blockDim.x*blockIdx.x + threadIdx.x);
    if (pos_of_elem < numelem)
    {
        FullTriangle elem(elements[pos_of_elem]);
        float B[2][2];
        B[0][0] = elem.nodeB.x - elem.nodeA.x;
        B[1][0] = elem.nodeB.y - elem.nodeA.y;
        B[0][1] = elem.nodeC.x - elem.nodeA.x;
        B[1][1] = elem.nodeC.y - elem.nodeA.y;
        //float d[2];
        //d[0] = elem.nodeA.x;
        //d[1] = elem.nodeA.y;
        float detB(std::abs(B[0][0]*B[1][1] - B[0][1]*B[1][0]));
        float gradA[2], gradB[2], gradC[2]; // in fact these are the gradient multplied by det(B)
        gradA[0] = B[1][0] - B[1][1];
        gradA[1] = B[0][1] - B[0][0];
        gradB[0] = B[1][1];
        gradB[1] = -B[0][1];
        gradC[0] = -B[1][0];
        gradC[1] = B[0][0];

        add_local_atomic(rowptr, colind, values, elem.nodeA.ID, elem.nodeA.ID, (gradA[0]*gradA[0] + gradA[1]*gradA[1]) / 2.0 / detB);
        add_local_atomic(rowptr, colind, values, elem.nodeA.ID, elem.nodeB.ID, (gradA[0]*gradB[0] + gradA[1]*gradB[1]) / 2.0 / detB);
        add_local_atomic(rowptr, colind, values, elem.nodeA.ID, elem.nodeC.ID, (gradA[0]*gradC[0] + gradA[1]*gradC[1]) / 2.0 / detB);
        add_local_atomic(rowptr, colind, values, elem.nodeB.ID, elem.nodeA.ID, (gradB[0]*gradA[0] + gradB[1]*gradA[1]) / 2.0 / detB);
        add_local_atomic(rowptr, colind, values, elem.nodeB.ID, elem.nodeB.ID, (gradB[0]*gradB[0] + gradB[1]*gradB[1]) / 2.0 / detB);
        add_local_atomic(rowptr, colind, values, elem.nodeB.ID, elem.nodeC.ID, (gradB[0]*gradC[0] + gradB[1]*gradC[1]) / 2.0 / detB);
        add_local_atomic(rowptr, colind, values, elem.nodeC.ID, elem.nodeA.ID, (gradC[0]*gradA[0] + gradC[1]*gradA[1]) / 2.0 / detB);
        add_local_atomic(rowptr, colind, values, elem.nodeC.ID, elem.nodeB.ID, (gradC[0]*gradB[0] + gradC[1]*gradB[1]) / 2.0 / detB);
        add_local_atomic(rowptr, colind, values, elem.nodeC.ID, elem.nodeC.ID, (gradC[0]*gradC[0] + gradC[1]*gradC[1]) / 2.0 / detB);
    }
}

__global__ void boundary(const size_t* const rowptr, const size_t* const colind, float* const values, const size_t numrows, const size_t* const boundaryNodes, const size_t numboundaryNodes)
{
    size_t id(blockDim.x*blockIdx.x + threadIdx.x);
    if (id < numboundaryNodes)
    {
        size_t b_id{boundaryNodes[id]};
        for (size_t i{rowptr[b_id]}; i < rowptr[b_id+1]; ++i)
            values[i] = (colind[i] == b_id ? 1.0 : 0.0);
    }
}

void assemble_atomic(size_t* d_rowptr, size_t* d_colind, float* d_values, size_t numrows, FullTriangle* h_elements, size_t numelem, size_t* h_boundaryNodes, size_t numboundaryNodes)
{
    FullTriangle* d_elements;
    malloc_cuda(&d_elements, numelem*sizeof(FullTriangle));
    memcpy_cuda(d_elements, h_elements, numelem*sizeof(FullTriangle), h2d);

    dim3 numthreads;
    dim3 numblocks;
    get_kernel_config(&numblocks, &numthreads, numelem);
    atomic<<<numblocks, numthreads>>>(d_rowptr, d_colind, d_values, numrows, d_elements, numelem);
    get_kernel_config(&numblocks, &numthreads, numboundaryNodes);
    cudaDeviceSynchronize();
    free_cuda(d_elements);

    size_t* d_boundaryNodes;
    malloc_cuda(&d_boundaryNodes, numboundaryNodes*sizeof(size_t));
    memcpy_cuda(d_boundaryNodes, h_boundaryNodes, numboundaryNodes*sizeof(size_t), h2d);
    boundary<<<numblocks, numthreads>>>(d_rowptr, d_colind, d_values, numrows, d_boundaryNodes, numboundaryNodes);
    cudaDeviceSynchronize();

    free_cuda(d_elements);
}
