#include <cuda.h>
#include <cuda_runtime.h>

#include "include/csrmatrixgpu.hpp"
#include "include/vectorgpu.hpp"
#include "include/global.cuh"
#include <iostream>

// standard assembly

__device__ void add_atomic(const size_t* const rowptr, const size_t* const colind, float* const values, const size_t row, const size_t col, const float val)
{
    size_t pos_to_insert(rowptr[row]);
    size_t endline = rowptr[row+1];
    while (colind[pos_to_insert] < col && pos_to_insert < endline)
        ++pos_to_insert;
    atomicAdd(values + pos_to_insert, val);
    //*(values + pos_to_insert) += val; // this is not faster !?
}

// SHARED IS NOT USED!
// needs 18 * sizeo(double) Bytes shared memory per block
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
            //add_atomic(rowptr, colind, values, dofs[i], dofs[j], val);
            add_atomic(rowptr, colind, values, dof[i], dof[j], val);
        } // end for j
    } // end for i
}// end if globalID < num_elements
}

// quadpoints have to be factor 0.0625!
void __K_assemble_element_Q1_O5(const float* d_coords, const size_t* d_dof, const float* d_quadpoints, const float* d_weights, const size_t num_elements,
                                const size_t* const d_rowptr, const size_t* const d_colind, float* d_values, const size_t numrows)
{
    dim3 numthreads;
    dim3 numblocks;
    get_kernel_config(&numblocks, &numthreads, num_elements);
    assemble_element_Q1_O5<<<numblocks, numthreads>>>(d_coords, d_dof, d_quadpoints, d_weights, num_elements,
                                                      d_rowptr, d_colind, d_values, numrows);
    cudaDeviceSynchronize();
}


__global__ void assemble_element_Q1_O5_unrolled(const float* node_coords, const size_t* dof_ids, const float* quadpoints, const float* weights, const size_t num_elements,
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

    // i = 0, j = 0
    float val(0.0f);
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f - eta);
        float grad11 = -(1.0f - xi );
        float grad20 = -(1.0f - eta);
        float grad21 = -(1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[0], dof[0], val);

    // i = 0, j = 1
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f - eta);
        float grad11 = -(1.0f - xi );
        float grad20 =  (1.0f - eta);
        float grad21 = -(1.0f + xi );


        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[0], dof[1], val);

    // i = 0, j = 2
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f - eta);
        float grad11 = -(1.0f - xi );
        float grad20 =  (1.0f + eta);
        float grad21 =  (1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[0], dof[2], val);

    // i = 0, j = 3
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f - eta);
        float grad11 = -(1.0f - xi );
        float grad20 = -(1.0f + eta);
        float grad21 =  (1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[0], dof[3], val);

    // i = 1, j = 0
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f - eta);
        float grad11 = -(1.0f + xi );
        float grad20 = -(1.0f - eta);
        float grad21 = -(1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[1], dof[0], val);

    // i = 1, j = 1
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f - eta);
        float grad11 = -(1.0f + xi );
        float grad20 =  (1.0f - eta);
        float grad21 = -(1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[1], dof[1], val);

    // i = 1, j = 2
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f - eta);
        float grad11 = -(1.0f + xi );
        float grad20 =  (1.0f + eta);
        float grad21 =  (1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[1], dof[2], val);

    // i = 1, j = 3
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f - eta);
        float grad11 = -(1.0f + xi );
        float grad20 = -(1.0f + eta);
        float grad21 =  (1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[1], dof[3], val);

    // i = 2, j = 0
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f + eta);
        float grad11 =  (1.0f + xi );
        float grad20 = -(1.0f - eta);
        float grad21 = -(1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[2], dof[0], val);

    // i = 2, j = 1
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f + eta);
        float grad11 =  (1.0f + xi );
        float grad20 =  (1.0f - eta);
        float grad21 = -(1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[2], dof[1], val);

    // i = 2, j = 2
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f + eta);
        float grad11 =  (1.0f + xi );
        float grad20 =  (1.0f + eta);
        float grad21 =  (1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[2], dof[2], val);

    // i = 2, j = 3
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f + eta);
        float grad11 =  (1.0f + xi );
        float grad20 = -(1.0f + eta);
        float grad21 =  (1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[2], dof[3], val);

    // i = 3, j = 0
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = quadpoints[2*p  ];
        float eta = quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f + eta);
        float grad11 =  (1.0f - xi );
        float grad20 = -(1.0f - eta);
        float grad21 = -(1.0f - xi );

        val  += weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[3], dof[0], val);

    // i = 3, j = 1
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = quadpoints[2*p  ];
        float eta = quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f + eta);
        float grad11 =  (1.0f - xi );
        float grad20 =  (1.0f - eta);
        float grad21 = -(1.0f + xi );

        val  += weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[3], dof[1], val);

    // i = 3, j = 2
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = quadpoints[2*p  ];
        float eta = quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f + eta);
        float grad11 =  (1.0f - xi );
        float grad20 =  (1.0f + eta);
        float grad21 =  (1.0f + xi );

        val  += weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[3], dof[2], val);

    // i = 3, j = 3
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = quadpoints[2*p  ];
        float eta = quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f + eta);
        float grad11 =  (1.0f - xi );
        float grad20 = -(1.0f + eta);
        float grad21 =  (1.0f - xi );

        val  += weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    add_atomic(rowptr, colind, values, dof[3], dof[3], val);

}// end if globalID < num_elements
}

// quadpoints have to be factor 0.0625!
void __K_assemble_element_Q1_O5_unrolled(const float* d_coords, const size_t* d_dof, const float* d_quadpoints, const float* d_weights, const size_t num_elements,
                                         const size_t* const d_rowptr, const size_t* const d_colind, float* d_values, const size_t numrows)
{
    dim3 numthreads;
    dim3 numblocks;
    get_kernel_config(&numblocks, &numthreads, num_elements);
//    numthreads.y = 4; numthreads.z = 4;
//    std::cout << numblocks.x << ", " << numblocks.y << ", " << numblocks.z << " | " << numthreads.x << ", " << numthreads.y << ", " << numthreads.z << std::endl;
    assemble_element_Q1_O5_unrolled<<<numblocks, numthreads>>>(d_coords, d_dof, d_quadpoints, d_weights, num_elements,
                                                               d_rowptr, d_colind, d_values, numrows);
    cudaDeviceSynchronize();
}



// ***** ***** ***** LOCAL MATRIX APPROACH ***** ***** *****

__global__ void assemble_element_Q1_O5_LM(const float* node_coords, const size_t* dof_ids, const float* quadpoints, const float* weights, const size_t num_elements,
                                          float* localmatrices)
{
size_t globalID = blockDim.x * blockIdx.x + threadIdx.x;
if (globalID < num_elements)
{
    const float* coords = node_coords + 8*globalID;
    //const size_t* dofs = dof_ids + 4*globalID;
    float* LM = localmatrices + 16*globalID;
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

    for (short i(0); i < 4; ++i)
    {
        for (short j(0); j < 4; ++j)
        {
            float val(0.0);

            for (size_t p(0); p < 9; ++p)
            {
                float xi  = s_quadpoints[2*p  ];
                float eta = s_quadpoints[2*p+1];
                float B[2][2] =
                    { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) * 0.25f ,
                        ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) * 0.25f },
                      { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) * 0.25f ,
                        ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) * 0.25f } };

                // help vars
                float grad1[2];
                float grad2[2];
                grad1[0] = (i == 0 ? (1.0f - eta) * (-0.25f) :
                           (i == 1 ? (1.0f - eta) *   0.25f  :
                           (i == 2 ? (1.0f + eta) *   0.25f  :
                                     (1.0f + eta) * (-0.25f) ) ) );
                grad1[1] = (i == 0 ? (1.0f - xi ) * (-0.25f) :
                           (i == 1 ? (1.0f + xi ) * (-0.25f) :
                           (i == 2 ? (1.0f + xi ) *   0.25f  :
                                     (1.0f - xi ) *   0.25f  ) ) );
                grad2[0] = (j == 0 ? (1.0f - eta) * (-0.25f) :
                           (j == 1 ? (1.0f - eta) *   0.25f  :
                           (j == 2 ? (1.0f + eta) *   0.25f  :
                                     (1.0f + eta) * (-0.25f) ) ) );
                grad2[1] = (j == 0 ? (1.0f - xi ) * (-0.25f) :
                           (j == 1 ? (1.0f + xi ) * (-0.25f) :
                           (j == 2 ? (1.0f + xi ) *   0.25f  :
                                     (1.0f - xi ) *   0.25f  ) ) );

                val  += s_weights[p]
                        * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
                            + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
                        / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
            } // end for p (quadrature point)
            LM[4*i + j] = val;
        } // end for j
    } // end for i
}// end if globalID < num_elements
}

void gather_res(const size_t* d_dof, const float* const d_localmatrices, const size_t numelem)//, float* res)
{
    //size_t h_dof[4*numelem];
    size_t h_dof[4];
    //memcpy_cuda(h_dof, d_dof, 4*numelem*sizeof(size_t), d2h);
    for (size_t e(0); e < numelem; ++e)
    {
        memcpy_cuda(h_dof, d_dof+4*e, 4*sizeof(size_t), d2h);
        for (size_t k(0); k < 4; ++k)
            if (h_dof[k] == 0)
            {
                float tmp[4];
                memcpy_cuda(tmp, d_localmatrices + 16*e + 4*k, 4*sizeof(float), d2h);
                std::cout << h_dof[0] << ": " << tmp[0] << std::endl;
                std::cout << h_dof[1] << ": " << tmp[1] << std::endl;
                std::cout << h_dof[2] << ": " << tmp[2] << std::endl;
                std::cout << h_dof[3] << ": " << tmp[3] << std::endl;
            }
    }
}


void __K_assemble_element_Q1_O5_LM(const float* d_coords, const size_t* d_dof, const float* d_quadpoints, const float* d_weights, const size_t num_elements,
                                   float* d_localmatrices)
{
    dim3 numthreads;
    dim3 numblocks;
    get_kernel_config(&numblocks, &numthreads, num_elements);
    std::cout << numblocks.x << " blocks a " << numthreads.x << " threads" << std::endl;
    assemble_element_Q1_O5_LM<<<numblocks, numthreads>>>(d_coords, d_dof, d_quadpoints, d_weights, num_elements,
                                                         d_localmatrices);
    cudaDeviceSynchronize();
    //gather_res(d_dof, d_localmatrices, num_elements);
}


__global__ void assemble_element_Q1_O5_LM_unrolled(const float* node_coords, const size_t* dof_ids, const float* quadpoints, const float* weights, const size_t num_elements,
                                                   float* localmatrices)
{
size_t globalID = blockDim.x * blockIdx.x + threadIdx.x;
if (globalID < num_elements)
{
    const float* coords = node_coords + 8*globalID;
    //const size_t* dofs = dof_ids + 4*globalID;
    float* LM = localmatrices + 16*globalID;
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
    //size_t dof[4] = {dofs[0], dofs[1], dofs[2], dofs[3]}; // this one not that much

    // i = 0, j = 0
    float val(0.0f);
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f - eta);
        float grad11 = -(1.0f - xi );
        float grad20 = -(1.0f - eta);
        float grad21 = -(1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[0] = val;

    // i = 0, j = 1
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f - eta);
        float grad11 = -(1.0f - xi );
        float grad20 =  (1.0f - eta);
        float grad21 = -(1.0f + xi );


        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[1] = val;

    // i = 0, j = 2
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f - eta);
        float grad11 = -(1.0f - xi );
        float grad20 =  (1.0f + eta);
        float grad21 =  (1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[2] = val;

    // i = 0, j = 3
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f - eta);
        float grad11 = -(1.0f - xi );
        float grad20 = -(1.0f + eta);
        float grad21 =  (1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[3] = val;

    // i = 1, j = 0
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f - eta);
        float grad11 = -(1.0f + xi );
        float grad20 = -(1.0f - eta);
        float grad21 = -(1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[4] = val;

    // i = 1, j = 1
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f - eta);
        float grad11 = -(1.0f + xi );
        float grad20 =  (1.0f - eta);
        float grad21 = -(1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[5] = val;

    // i = 1, j = 2
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f - eta);
        float grad11 = -(1.0f + xi );
        float grad20 =  (1.0f + eta);
        float grad21 =  (1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[6] = val;

    // i = 1, j = 3
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f - eta);
        float grad11 = -(1.0f + xi );
        float grad20 = -(1.0f + eta);
        float grad21 =  (1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[7] = val;

    // i = 2, j = 0
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f + eta);
        float grad11 =  (1.0f + xi );
        float grad20 = -(1.0f - eta);
        float grad21 = -(1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[8] = val;

    // i = 2, j = 1
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f + eta);
        float grad11 =  (1.0f + xi );
        float grad20 =  (1.0f - eta);
        float grad21 = -(1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[9] = val;

    // i = 2, j = 2
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f + eta);
        float grad11 =  (1.0f + xi );
        float grad20 =  (1.0f + eta);
        float grad21 =  (1.0f + xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[10] = val;

    // i = 2, j = 3
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = s_quadpoints[2*p  ];
        float eta = s_quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 =  (1.0f + eta);
        float grad11 =  (1.0f + xi );
        float grad20 = -(1.0f + eta);
        float grad21 =  (1.0f - xi );

        val  += s_weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[11] = val;

    // i = 3, j = 0
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = quadpoints[2*p  ];
        float eta = quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f + eta);
        float grad11 =  (1.0f - xi );
        float grad20 = -(1.0f - eta);
        float grad21 = -(1.0f - xi );

        val  += weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[12] = val;

    // i = 3, j = 1
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = quadpoints[2*p  ];
        float eta = quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f + eta);
        float grad11 =  (1.0f - xi );
        float grad20 =  (1.0f - eta);
        float grad21 = -(1.0f + xi );

        val  += weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[13] = val;

    // i = 3, j = 2
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = quadpoints[2*p  ];
        float eta = quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f + eta);
        float grad11 =  (1.0f - xi );
        float grad20 =  (1.0f + eta);
        float grad21 =  (1.0f + xi );

        val  += weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[14] = val;

    // i = 3, j = 3
    val = 0.0f;
    for (size_t p(0); p < 9; ++p)
    {
        float xi  = quadpoints[2*p  ];
        float eta = quadpoints[2*p+1];
        float B[2][2] =
            { { ( -(1.0f-eta)*coords0 + (1.0f-eta)*coords1 + (1.0f+eta)*coords2 - (1.0f+eta)*coords3 ) ,
                ( -(1.0f-xi )*coords0 - (1.0f+xi )*coords1 + (1.0f+xi )*coords2 + (1.0f-xi )*coords3 ) },
              { ( -(1.0f-eta)*coords4 + (1.0f-eta)*coords5 + (1.0f+eta)*coords6 - (1.0f+eta)*coords7 ) ,
                ( -(1.0f-xi )*coords4 - (1.0f+xi )*coords5 + (1.0f+xi )*coords6 + (1.0f-xi )*coords7 ) } };

        // help vars
        float grad10 = -(1.0f + eta);
        float grad11 =  (1.0f - xi );
        float grad20 = -(1.0f + eta);
        float grad21 =  (1.0f - xi );

        val  += weights[p]
                * (   ( B[1][1] * grad10 - B[1][0] * grad11) * ( B[1][1] * grad20 - B[1][0] * grad21)
                    + (-B[0][1] * grad10 + B[0][0] * grad11) * (-B[0][1] * grad20 + B[0][0] * grad21) )
                / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
    } // end for p (quadrature point)
    LM[15] = val;

}// end if globalID < num_elements
}

void __K_assemble_element_Q1_O5_LM_unrolled(const float* d_coords, const size_t* d_dof, const float* d_quadpoints, const float* d_weights, const size_t num_elements,
                                            float* d_localmatrices)
{
    dim3 numthreads;
    dim3 numblocks;
    get_kernel_config(&numblocks, &numthreads, num_elements);
    std::cout << numblocks.x << " blocks a " << numthreads.x << " threads" << std::endl;
    assemble_element_Q1_O5_LM_unrolled<<<numblocks, numthreads>>>(d_coords, d_dof, d_quadpoints, d_weights, num_elements,
                                                                  d_localmatrices);
    cudaDeviceSynchronize();
    //gather_res(d_dof, d_localmatrices, num_elements);
}

// on element per block

//__global__ void assemble_element_Q1_O5_LM_1epb(const float* node_coords, const float* quadpoints,
//                                               const float* weights, float* localmatrices)
__global__ void assemble_element_Q1_O5_LM_1epb(const float* node_coords, const float* quadpoints_weights, float* localmatrices)
{
    const float* coords = node_coords + 8*blockIdx.x;
    //const size_t* dofs = dof_ids + 4*globalID;
    float* LM = localmatrices + 16*blockIdx.x;
    // load quadrature points and there corresponding weights and the coordinagtes into shared memory
    __shared__ float sharedmemory[35 + 144]; // 2*9qp + 9w + 2*4co + 4*4*9val
    float* s_quadpoints = sharedmemory;
    float* s_weights    = sharedmemory + 18;
    float* s_coords     = sharedmemory + 27;

    int i = threadIdx.x;
    int j = threadIdx.y;
    s_quadpoints[4*i + j] = quadpoints_weights[4*i + j];
    if (4*i + j < 11)
        s_quadpoints[16 + 4*i + j] = quadpoints_weights[16 + 4*i + j];
    if (i < 2)
        s_coords[2*i + j] = coords[2*i + j];
            //float val(0.0);

            //for (size_t p(0); p < 9; ++p)
            //{
            int p = threadIdx.z;
                float xi  = s_quadpoints[2*p  ];
                float eta = s_quadpoints[2*p+1];
                float B[2][2] =
                    { { ( -(1.0f-eta)*s_coords[0] + (1.0f-eta)*s_coords[1] + (1.0f+eta)*s_coords[2] - (1.0f+eta)*s_coords[3] ) * 0.25f ,
                        ( -(1.0f-xi )*s_coords[0] - (1.0f+xi )*s_coords[1] + (1.0f+xi )*s_coords[2] + (1.0f-xi )*s_coords[3] ) * 0.25f },
                      { ( -(1.0f-eta)*s_coords[4] + (1.0f-eta)*s_coords[5] + (1.0f+eta)*s_coords[6] - (1.0f+eta)*s_coords[7] ) * 0.25f ,
                        ( -(1.0f-xi )*s_coords[4] - (1.0f+xi )*s_coords[5] + (1.0f+xi )*s_coords[6] + (1.0f-xi )*s_coords[7] ) * 0.25f } };
                //    { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) * 0.25f ,
                //        ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) * 0.25f },
                //      { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) * 0.25f ,
                //        ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) * 0.25f } };
                //    { { ( -(1.0f-eta)*node_coords[8*globalID + 0] + (1.0f-eta)*node_coords[8*globalID + 1] + (1.0f+eta)*node_coords[8*globalID + 2] - (1.0f+eta)*node_coords[8*globalID + 3] ) * 0.25f ,
                //        ( -(1.0f-xi )*node_coords[8*globalID + 0] - (1.0f+xi )*node_coords[8*globalID + 1] + (1.0f+xi )*node_coords[8*globalID + 2] + (1.0f-xi )*node_coords[8*globalID + 3] ) * 0.25f },
                //      { ( -(1.0f-eta)*node_coords[8*globalID + 4] + (1.0f-eta)*node_coords[8*globalID + 5] + (1.0f+eta)*node_coords[8*globalID + 6] - (1.0f+eta)*node_coords[8*globalID + 7] ) * 0.25f ,
                //        ( -(1.0f-xi )*node_coords[8*globalID + 4] - (1.0f+xi )*node_coords[8*globalID + 5] + (1.0f+xi )*node_coords[8*globalID + 6] + (1.0f-xi )*node_coords[8*globalID + 7] ) * 0.25f } };

                // help vars
                float grad1[2];
                float grad2[2];

                grad1[0] = (i < 2 ? (1.0f - eta) : (1.0f + eta) );
                grad1[0] *= (i == 0 || i == 3 ? -0.25f : 0.25f );
                grad1[1] = (i < 2 ? -0.25f : 0.25f);
                grad1[1] *= (i == 0 || i == 3 ? (1.0f - xi ) : (1.0f + xi ) );
                grad2[0] = (j < 2 ? (1.0f - eta) : (1.0f + eta) );
                grad2[0] *= (j == 0 || j == 3 ? -0.25f : 0.25f );
                grad2[1] = (j < 2 ? -0.25f : 0.25f);
                grad2[1] *= (j == 0 || j == 3 ? (1.0f - xi ) : (1.0f + xi ) );

                //grad1[0] = (i == 0 ? (1.0f - eta) * (-0.25f) :
                //           (i == 1 ? (1.0f - eta) *   0.25f  :
                //           (i == 2 ? (1.0f + eta) *   0.25f  :
                //                     (1.0f + eta) * (-0.25f) ) ) );
                //grad1[1] = (i == 0 ? (1.0f - xi ) * (-0.25f) :
                //           (i == 1 ? (1.0f + xi ) * (-0.25f) :
                //           (i == 2 ? (1.0f + xi ) *   0.25f  :
                //                     (1.0f - xi ) *   0.25f  ) ) );
                //grad2[0] = (j == 0 ? (1.0f - eta) * (-0.25f) :
                //           (j == 1 ? (1.0f - eta) *   0.25f  :
                //           (j == 2 ? (1.0f + eta) *   0.25f  :
                //                     (1.0f + eta) * (-0.25f) ) ) );
                //grad2[1] = (j == 0 ? (1.0f - xi ) * (-0.25f) :
                //           (j == 1 ? (1.0f + xi ) * (-0.25f) :
                //           (j == 2 ? (1.0f + xi ) *   0.25f  :
                //                     (1.0f - xi ) *   0.25f  ) ) );

                //val  += s_weights[p]
                //        * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
                //            + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
                //        / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
                sharedmemory[35 + 9*(4*i + j) + p] = s_weights[p]
                                                 * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
                                                     + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
                                                 / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
            //} // end for p (quadrature point)
            __syncthreads();
            if (p == 0)
            {
                float val(0.0f);
                for (int k(0); k < 9; ++k)
                    val += sharedmemory[35 + 4*i + j + p];
                LM[4*i + j] = val;
            }
            //LM[4*i + j] = val;
    //    } // end for j
    //} // end for i
}

void __K_assemble_element_Q1_O5_LM_1epb(const float* d_coords, const float* d_quadpoints, const float* d_weights,
                                        const size_t num_elements, float* d_localmatrices)
{
    //assemble_element_Q1_O5_LM_1epb<<<dim3(num_elements, 1, 1), dim3(4, 4, 1)>>>(d_coords, d_quadpoints, d_weights, d_localmatrices);
    assemble_element_Q1_O5_LM_1epb<<<dim3(num_elements, 1, 1), dim3(4, 4, 9)>>>(d_coords, d_quadpoints, d_localmatrices);
    cudaDeviceSynchronize();
}
