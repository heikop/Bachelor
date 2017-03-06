#include <cuda.h>
#include <cuda_runtime.h>

#include "include/csrmatrixgpu.hpp"
#include "include/vectorgpu.hpp"
#include "include/global.cuh"
#include <iostream>

// csr, q1, loop, nopre???, f
__device__ void add_atomic(const size_t* const rowptr, const size_t* const colind, float* const values, const size_t row, const size_t col, const float val)
{
    size_t pos_to_insert(rowptr[row]);
    size_t endline = rowptr[row+1];
    while (colind[pos_to_insert] < col && pos_to_insert < endline)
        ++pos_to_insert;
    atomicAdd(values + pos_to_insert, val);
}

// needs 27 * sizeo(float) Bytes shared memory per block
// only utilizes x-Dimension
__global__ void assemble_element_Q1_O5(const float* node_coords, const size_t* dof_ids, const float* quadpoints, const float* weights, const size_t num_elements,
                                       const size_t* const rowptr, const size_t* const colind, float* values, const size_t numrows)
{
size_t globalID = blockDim.x * blockIdx.x + threadIdx.x;
if (globalID < num_elements)
{
    const float* coords = node_coords + 8*globalID;
    const size_t* dofs = dof_ids + 4*globalID;
    // load quadrature points and there corresponding weights into shared memory
    __shared__ float sharedmemory[27];
    float* s_quadpoints = sharedmemory;
    float* s_weights    = sharedmemory+18;
    if (threadIdx.x < 18)
        s_quadpoints[threadIdx.x] = quadpoints[threadIdx.x];
    if (threadIdx.x < 9)
        s_weights[threadIdx.x] = weights[threadIdx.x];
    __syncthreads();
    // following is by far most useful for optimizing
    float coords0(coords[0]);
    float coords1(coords[1]);
    float coords2(coords[2]);
    float coords3(coords[3]);
    float coords4(coords[4]);
    float coords5(coords[5]);
    float coords6(coords[6]);
    float coords7(coords[7]);
    size_t dof[4] = {dofs[0], dofs[1], dofs[2], dofs[3]}; // this one not that much


    for (short i(0); i < 4; ++i)
    {
        for (short j(0); j < 4; ++j)
        {
            float val(0.0f);

            for (short p(0); p < 9; ++p)
            {
                float xi  = s_quadpoints[2*p  ];
                float eta = s_quadpoints[2*p+1];
                float B[2][2] =
                    { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                        ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
                      { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                        ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

                // help vars
                float grad1[2];
                float grad2[2];
                grad1[0] = (i == 0 ? -(1.0f - eta) :
                           (i == 1 ?  (1.0f - eta) :
                           (i == 2 ?  (1.0f + eta) :
                                     -(1.0f + eta) ) ) );
                grad1[1] = (i == 0 ? -(1.0f - xi ) :
                           (i == 1 ? -(1.0f + xi ) :
                           (i == 2 ?  (1.0f + xi ) :
                                      (1.0f - xi ) ) ) );
                grad2[0] = (j == 0 ? -(1.0f - eta) :
                           (j == 1 ?  (1.0f - eta) :
                           (j == 2 ?  (1.0f + eta) :
                                     -(1.0f + eta) ) ) );
                grad2[1] = (j == 0 ? -(1.0f - xi ) :
                           (j == 1 ? -(1.0f + xi ) :
                           (j == 2 ?  (1.0f + xi ) :
                                      (1.0f - xi ) ) ) );

                val  += s_weights[p]
                        * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
                            + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
                        / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
            } // end for p (quadrature point)
            add_atomic(rowptr, colind, values, dof[i], dof[j], val);
        } // end for j
    } // end for i
}// end if globalID < num_elements
}

// quadpoints have to be factor 0.0625!
void __K_072(const float* d_coords, const size_t* d_dof, const float* d_quadpoints, const float* d_weights, const size_t num_elements,
                                const size_t* const d_rowptr, const size_t* const d_colind, float* d_values, const size_t numrows)
{
    dim3 numthreads;
    dim3 numblocks;
    get_kernel_config(&numblocks, &numthreads, num_elements);
    assemble_element_Q1_O5<<<numblocks, numthreads>>>(d_coords, d_dof, d_quadpoints, d_weights, num_elements,
                                                      d_rowptr, d_colind, d_values, numrows);
    cudaDeviceSynchronize();
}
