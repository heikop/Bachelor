#include "include/csrmatrixgpu.hpp"

__device__ void add_atomic(const size_t* const rowptr, const size_t* const colind, float* const values, const size_t row, const size_t col, const float val)
{
    size_t pos_to_insert(rowptr[row]);
    while (colind[pos_to_insert] < col && pos_to_insert < rowptr[row+1])
        ++pos_to_insert;
    atomicAdd(values + pos_to_insert, val);
    //*(values + pos_to_insert) += val; // this is not faster !?
}

// needs 18 * sizeo(double) Bytes shared memory per block
// only utilizes x-Dimension
__global__ void assemble_element_Q1_O5(const double* node_coords, const size_t* dof_ids, const double* quadpoints, const double* weights, const size_t num_elements,
                                       const size_t* const rowptr, const size_t* const colind, float* const values, const size_t numrows)
{
size_t globalID = blockDim.x * blockIdx.x + threadIdx.x;
if (globalID < num_elements)
{
    // load quadrature points and there corresponding weights into shared memory
    __shared__ double sharedmemory[27];
    double* s_quadpoints = sharedmemory;
    double* s_weights    = sharedmemory+18;
    if (threadIdx.x < 27)
    {
        if (threadIdx.x < 18)
            s_quadpoints[threadIdx.x] = quadpoints[threadIdx.x];
        else
            s_weights[threadIdx.x-18] = weights[threadIdx.x-18];
    }

    for (short i(0); i < 4; ++i)
    {
        for (short j(0); j < 4; ++j)
        {
            double val(0.0);

            for (size_t p(0); p < 9; ++p)
            {
                double xi  = s_quadpoints[2*p  ];
                double eta = s_quadpoints[2*p+1];
                double B[2][2] =
                    { { ( -(1.0-eta)*node_coords[0] + (1.0-eta)*node_coords[1] + (1.0+eta)*node_coords[2] - (1.0+eta)*node_coords[3] ) * 0.25 ,
                        ( -(1.0-xi )*node_coords[0] - (1.0+xi )*node_coords[1] + (1.0+xi )*node_coords[2] + (1.0-xi )*node_coords[3] ) * 0.25 },
                      { ( -(1.0-eta)*node_coords[4] + (1.0-eta)*node_coords[5] + (1.0+eta)*node_coords[6] - (1.0+eta)*node_coords[7] ) * 0.25 ,
                        ( -(1.0-xi )*node_coords[4] - (1.0+xi )*node_coords[5] + (1.0+xi )*node_coords[6] + (1.0-xi )*node_coords[7] ) * 0.25 } };

                // help vars
                double grad1[2];
                double grad2[2];
                if (i == 0)
                {
                    grad1[0] = (1.0 - eta) * (-0.25);
                    grad1[1] = (1.0 - xi ) * (-0.25);
                }
                else if (i == 1)
                {
                    grad1[0] = (1.0 - eta) *   0.25 ;
                    grad1[1] = (1.0 + xi ) * (-0.25);
                }
                else if (i == 2)
                {
                    grad1[0] = (1.0 + eta) *   0.25 ;
                    grad1[1] = (1.0 + xi ) *   0.25 ;
                }
                else //if (i == 3)
                {
                    grad1[0] = (1.0 + eta) * (-0.25);
                    grad1[1] = (1.0 - xi ) *   0.25 ;
                }
                if (j == 0)
                {
                    grad2[0] = (1.0 - eta) * (-0.25);
                    grad2[1] = (1.0 - xi ) * (-0.25);
                }
                else if (j == 1)
                {
                    grad2[0] = (1.0 - eta) *   0.25 ;
                    grad2[1] = (1.0 + xi ) * (-0.25);
                }
                else if (j == 2)
                {
                    grad2[0] = (1.0 + eta) *   0.25 ;
                    grad2[1] = (1.0 + xi ) *   0.25 ;
                }
                else //if (j == 3)
                {
                    grad2[0] = (1.0 + eta) * (-0.25);
                    grad2[1] = (1.0 - xi ) *   0.25 ;
                }

                val  += s_weights[p]
                        * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
                            + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
                        / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
            } // end for p (quadrature point)
            val /= 81.0; // all weights are .../81
            //val *= 0.0123456790123456790123;
            add_atomic(rowptr, colind, values, dof_ids[i], dof_ids[j], val);
            //matrix.add(dof_ids[i], dof_ids[j], val);
        } // end for j
    } // end for i
}// end if globalID < num_elements
}

void assemble_matrix_Q1_O5(const std::vector<Element<double>*>& elements, CsrMatrixGpu& matrix)
{
    const size_t numelem(elements.size());

    // copy all data onto the GPU memory
    std::vector<size_t> all_dof_ids(4 * numelem);
    std::vector<double> all_coords(8 * numelem);
    for (size_t e(0); e < numelem; ++e)
    {
        const std::vector<size_t> elem_dofs{elements[e]->vertexids()};
        std::array<double, 8> elem_coords = static_cast<QuadrilateralQ1<double>*>(elements[e])->get_pointcoords();
        all_dof_ids[4 * e + 0] = elem_dofs[0];
        all_dof_ids[4 * e + 0] = elem_dofs[1];
        all_dof_ids[4 * e + 0] = elem_dofs[2];
        all_dof_ids[4 * e + 0] = elem_dofs[3];
        all_coords[8 * e + 0] = elem_coords[0];
        all_coords[8 * e + 0] = elem_coords[1];
        all_coords[8 * e + 0] = elem_coords[2];
        all_coords[8 * e + 0] = elem_coords[3];
        all_coords[8 * e + 0] = elem_coords[4];
        all_coords[8 * e + 0] = elem_coords[5];
        all_coords[8 * e + 0] = elem_coords[6];
        all_coords[8 * e + 0] = elem_coords[7];
    }
    size_t* d_dofs;
    double* d_coords;
    malloc_cuda(&d_dofs, numelem*4*sizeof(size_t));
    malloc_cuda(&d_coords, numelem*8*sizeof(double));
    memcpy_cuda(d_dofs, all_dofs_ids.data(), numelem*4*sizeof(size_t), h2d);
    memcpy_cuda(d_coords, all_coords.data(), numelem*8*sizeof(double), h2d);
    all_dof_ids.clear();
    all_coords.clear();
    // also copy the 'shared data'
    double qp(std::sqrt(0.6));
    double weights[9]; weights[0] = 25.0; weights[1] = 40.0; weights[2] = 25.0;
                       weights[3] = 40.0; weights[4] = 64.0; weights[5] = 40.0;
                       weights[6] = 25.0; weights[7] = 40.0; weights[8] = 25.0;
    double quadpoints[9][2]; quadpoints[0][0] = -qp; quadpoints[0][1] = -qp;
                             quadpoints[1][0] = -qp; quadpoints[1][1] = 0.0;
                             quadpoints[2][0] = -qp; quadpoints[2][1] =  qp;
                             quadpoints[3][0] = 0.0; quadpoints[3][1] = -qp;
                             quadpoints[4][0] = 0.0; quadpoints[4][1] = 0.0;
                             quadpoints[5][0] = 0.0; quadpoints[5][1] =  qp;
                             quadpoints[6][0] =  qp; quadpoints[6][1] = -qp;
                             quadpoints[7][0] =  qp; quadpoints[7][1] = 0.0;
                             quadpoints[8][0] =  qp; quadpoints[8][1] =  qp;
    double* d_quadpoints;
    double* d_weights;
    malloc_cuda(&d_quadpoints, 18*sizeof(double));
    malloc_cuda(&d_weights, 9*sizeof(double));
    memcpy_cuda(&d_quadpoints, quadpoints, 18*sizeof(double), h2d);
    memcpy_cuda(&d_weights, weights, 9*sizeof(double), h2d);

    // call kernel
    dim3 numthreads;
    dim3 numblocks;
    get_kernel_config(&numblocks, &numthreads, elements.size());
    assemble_element_Q1_O5<<<numblocks, numthreads>>>(d_coords, d_dof_ids, d_quadpoints, d_weights, numelem
                                                      matrix._rowptr, matrix._colind, matrix._values, matrix._numrows);
    cudaDeviceSynchronize();
}
