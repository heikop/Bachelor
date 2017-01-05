#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <typeinfo>
#include <functional>

#include "include/elements.hpp"
#include "include/readfile.hpp"
#include "include/structure.hpp"
#include "include/quadrature.hpp"
#include "include/cgsolver.hpp"

#include "include/csrmatrixcpu.hpp"
#include "include/vectorcpu.hpp"
#include "include/csrmatrixgpu.hpp"
#include "include/vectorgpu.hpp"

#include "include/global.hpp"
#include "include/global.cuh"

#include <omp.h>
#include <stdio.h>

using namespace std;

// pre declaration
void assemble_matrix_Q1_O5(const std::vector<Element<float>*>& elements, CsrMatrixGpu& matrix, float* d_localmatrices);
void __K_assemble_element_Q1_O5(const float* node_coords, const size_t* dof_ids, const float* quadpoints, const float* weights, const size_t num_elements,
                                const size_t* const rowptr, const size_t* const colind, float* values, const size_t numrows);
void __K_assemble_element_Q1_O5_LM(const float* d_coords, const size_t* d_dof, const float* d_quadpoints, const float* d_weights, const size_t num_elements,
                                   float* d_localmatrices);
void __K_assemble_element_Q1_O5_LM_1epb(const float* d_coords, const float* d_quadpoints, const float* d_weights,
                                        const size_t num_elements, float* d_localmatrices);

int main(int argc, char* argv[])
{
    initCuda();

    double walltime[2];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Vertex<float>> nodes;
    std::vector<Element<float>*> elements;

    file_to_mesh_all("../data/square_quadrilateral_q1_c.msh", nodes, elements);
    std::cout << "num nodes: " << nodes.size() << std::endl;
    std::cout << "num elements: " << elements.size() << std::endl;

    CsrMatrixCpu<float> mat_cpu(nodes.size());
    std::cout << "structure" << std::flush;
    walltime[0] = omp_get_wtime();
    structure(mat_cpu, elements);
    walltime[0] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[0] * 1000.0 << ")" << std::endl;
    std::cout << "assemble ..." << std::endl;//std::flush;

    // copy matrix to gpu
    CsrMatrixGpu mat_gpu(nodes.size());
    malloc_cuda(&mat_gpu._colind, mat_cpu._rowptr[mat_cpu._numrows] * sizeof(size_t));
    malloc_cuda(&mat_gpu._values, mat_cpu._rowptr[mat_cpu._numrows] * sizeof(float));
    memcpy_cuda(mat_gpu._rowptr, mat_cpu._rowptr, (mat_cpu._numrows + 1) * sizeof(size_t), h2d);
    memcpy_cuda(mat_gpu._colind, mat_cpu._colind, mat_cpu._rowptr[mat_cpu._numrows] * sizeof(size_t), h2d);
    memcpy_cuda(mat_gpu._values, mat_cpu._values, mat_cpu._rowptr[mat_cpu._numrows] * sizeof(float), h2d);

    // local matrices
    float* d_localmatrices;
    malloc_cuda(&d_localmatrices, elements.size()*4*4*sizeof(float)); // 4*4 = dof*dof dense matrices

    cudaDeviceSynchronize();

    // function begin: assemble
    walltime[1] = omp_get_wtime();
    assemble_matrix_Q1_O5(elements, mat_gpu, d_localmatrices);

    // function end: assemble
    walltime[1] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[1] * 1000.0 << ")" << std::endl;

    free_cuda(d_localmatrices);

    return 0;
}

void assemble_matrix_Q1_O5(const std::vector<Element<float>*>& elements, CsrMatrixGpu& matrix, float* d_localmatrices)
{
    cudaDeviceSynchronize();
    double copytime = omp_get_wtime();

    const size_t numelem(elements.size());

    // copy all data onto the GPU memory
    std::vector<size_t> all_dof_ids(4 * numelem);
    std::vector<float> all_coords(8 * numelem);
    for (size_t e(0); e < numelem; ++e)
    {
        const std::vector<size_t> elem_dof(elements[e]->vertexids());
        std::array<float, 8> elem_coords = static_cast<QuadrilateralQ1<float>*>(elements[e])->get_pointcoords();
        all_dof_ids[4 * e + 0] = elem_dof[0];
        all_dof_ids[4 * e + 1] = elem_dof[1];
        all_dof_ids[4 * e + 2] = elem_dof[2];
        all_dof_ids[4 * e + 3] = elem_dof[3];
        all_coords[8 * e + 0] = elem_coords[0];
        all_coords[8 * e + 1] = elem_coords[1];
        all_coords[8 * e + 2] = elem_coords[2];
        all_coords[8 * e + 3] = elem_coords[3];
        all_coords[8 * e + 4] = elem_coords[4];
        all_coords[8 * e + 5] = elem_coords[5];
        all_coords[8 * e + 6] = elem_coords[6];
        all_coords[8 * e + 7] = elem_coords[7];
    }
    size_t* d_dof;
    float* d_coords;
    malloc_cuda(&d_dof, numelem*4*sizeof(size_t));
    malloc_cuda(&d_coords, numelem*8*sizeof(float));
    memcpy_cuda(d_dof, all_dof_ids.data(), numelem*4*sizeof(size_t), h2d);
    memcpy_cuda(d_coords, all_coords.data(), numelem*8*sizeof(float), h2d);
    all_dof_ids.clear();
    all_coords.clear();
    // also copy the 'shared data'
    float qp(std::sqrt(0.6));
    float weights[9]; weights[0] = 25.0/81.0; weights[1] = 40.0/81.0; weights[2] = 25.0/81.0;
                      weights[3] = 40.0/81.0; weights[4] = 64.0/81.0; weights[5] = 40.0/81.0;
                      weights[6] = 25.0/81.0; weights[7] = 40.0/81.0; weights[8] = 25.0/81.0;
    float quadpoints[18]; quadpoints[0]  = -qp; quadpoints[1]  = -qp;
                          quadpoints[2]  = -qp; quadpoints[3]  = 0.0;
                          quadpoints[4]  = -qp; quadpoints[5]  =  qp;
                          quadpoints[6]  = 0.0; quadpoints[7]  = -qp;
                          quadpoints[8]  = 0.0; quadpoints[9]  = 0.0;
                          quadpoints[10] = 0.0; quadpoints[11] =  qp;
                          quadpoints[12] =  qp; quadpoints[13] = -qp;
                          quadpoints[14] =  qp; quadpoints[15] = 0.0;
                          quadpoints[16] =  qp; quadpoints[17] =  qp;
    float* d_quadpoints;
    float* d_weights;
    malloc_cuda(&d_quadpoints, 27*sizeof(float));
    //malloc_cuda(&d_quadpoints, 18*sizeof(float));
    //malloc_cuda(&d_weights, 9*sizeof(float));
    d_weights = d_quadpoints + 18;
    memcpy_cuda(d_quadpoints, quadpoints, 18*sizeof(float), h2d);
    memcpy_cuda(d_weights, weights, 9*sizeof(float), h2d);

    copytime -= omp_get_wtime();
    std::cout << " - copytime: " << -copytime * 1000.0 << std::endl;

    // call kernel
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    double kerneltime = omp_get_wtime();
    __K_assemble_element_Q1_O5(d_coords, d_dof, d_quadpoints, d_weights, numelem,
                               matrix._rowptr, matrix._colind, matrix._values, matrix._numrows);
    //__K_assemble_element_Q1_O5_LM(d_coords, d_dof, d_quadpoints, d_weights, numelem,
    //                              d_localmatrices);
    //__K_assemble_element_Q1_O5_LM_1epb(d_coords, d_quadpoints, d_weights, numelem, d_localmatrices);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    kerneltime -= omp_get_wtime();
    std::cout << " - kerneltime: " << -kerneltime * 1000.0 << std::endl;
    std::cout << " - total: " << (-copytime -kerneltime) * 1000.0 << std::endl;
    float values[10];
    size_t colind[10];
    memcpy_cuda(values, matrix._values, 10*sizeof(float), d2h);
    memcpy_cuda(colind, matrix._colind, 10*sizeof(size_t), d2h);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    for (int k(0); k < 10; ++k)
        std::cout << "(0, " << colind[k] << ") = " << values[k] << std::endl;

    free_cuda(d_dof);
    free_cuda(d_coords);
    free_cuda(d_quadpoints);
    //free_cuda(d_weights);
}
