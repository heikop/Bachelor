#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <typeinfo>
#include <functional>

#include "../../include/elements.hpp"
#include "../../include/readfile.hpp"
#include "../../include/structure.hpp"
#include "../../include/quadrature.hpp"
#include "../../include/cgsolver.hpp"

#include "../../include/csrmatrixcpu.hpp"
#include "../../include/vectorcpu.hpp"
#include "../../include/csrmatrixgpu.hpp"
#include "../../include/vectorgpu.hpp"

#include "../../include/global.hpp"
#include "../../include/global.cuh"

#include <cuda_profiler_api.h>

#include <omp.h>
#include <stdio.h>

using namespace std;

// pre declaration
//void __K_assemble_element_Q1_O5(const float* node_coords, const size_t* dof_ids, const float* quadpoints, const float* weights, const size_t num_elements,
//                                const size_t* const rowptr, const size_t* const colind, float* values, const size_t numrows);
void __K_072(const float* node_coords, const size_t* dof_ids, const float* quadpoints, const float* weights, const size_t num_elements,
             const size_t* const rowptr, const size_t* const colind, float* values, const size_t numrows);

int main(int argc, char* argv[])
{
    const size_t OUTPUTSIZE = 10;
    if (argc > 1)
        std::cout << "incl solving and output" << std::endl;

    initCuda();

    double walltime[2];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Vertex<float>> nodes;
    std::vector<Element<float>*> elements;

    file_to_mesh_all("../data/square_quadrilateral_q1_d.msh", nodes, elements); // Q1
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

    cudaDeviceSynchronize();

    // function begin: assemble
    walltime[1] = omp_get_wtime();

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
    float weights[9]; weights[0] = 25.0/81.0*0.0625; weights[1] = 40.0/81.0*0.0625; weights[2] = 25.0/81.0*0.0625;
                      weights[3] = 40.0/81.0*0.0625; weights[4] = 64.0/81.0*0.0625; weights[5] = 40.0/81.0*0.0625;
                      weights[6] = 25.0/81.0*0.0625; weights[7] = 40.0/81.0*0.0625; weights[8] = 25.0/81.0*0.0625;
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
    cudaProfilerStart();
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    double kerneltime = omp_get_wtime();
    //__K_assemble_element_Q1_O5(d_coords, d_dof, d_quadpoints, d_weights, numelem,
    //                           mat_gpu._rowptr, mat_gpu._colind, mat_gpu._values, mat_cpu._numrows);
    __K_072(d_coords, d_dof, d_quadpoints, d_weights, numelem,
            mat_gpu._rowptr, mat_gpu._colind, mat_gpu._values, mat_cpu._numrows);
    cudaDeviceSynchronize();
    cudaThreadSynchronize();
    kerneltime -= omp_get_wtime();
    std::cout << " - kerneltime: " << -kerneltime * 1000.0 << std::endl;
    std::cout << " - total: " << (-copytime -kerneltime) * 1000.0 << std::endl;
    cudaProfilerStop();
    //float values[10];
    //size_t colind[50];
    //memcpy_cuda(values, mat_gpu._values, 10*sizeof(float), d2h);
    //memcpy_cuda(colind, mat_gpu._colind, 50*sizeof(size_t), d2h);
    //cudaDeviceSynchronize();
    //for (int k(0); k < 10; ++k)
    //    std::cout << "(0, " << colind[k] << ") = " << values[k] << std::endl;

    free_cuda(d_dof);
    free_cuda(d_coords);
    free_cuda(d_quadpoints);
    //free_cuda(d_weights);

    // function end: assemble
    walltime[1] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[1] * 1000.0 << ")" << std::endl;

    memcpy_cuda(mat_cpu._values, mat_gpu._values, mat_cpu._rowptr[mat_cpu._numrows] * sizeof(float), d2h);
    //cudaDeviceSynchronize();
    //cudaThreadSynchronize();
    std::cout << "num values: " << mat_cpu._rowptr[mat_cpu._numrows] << std::endl;
    for (int k(0); k < OUTPUTSIZE; ++k)
        std::cout << "(0, " << mat_cpu._colind[mat_cpu._rowptr[mat_cpu._numrows] - k] << ") = " << mat_cpu._values[mat_cpu._rowptr[mat_cpu._numrows] - k] << std::endl;
        //std::cout << "(0, " << mat_cpu._colind[k] << ") = " << mat_cpu._values[k] << std::endl;
    std::cout << "extra: " << mat_cpu._colind[mat_cpu._rowptr[mat_cpu._numrows] - 50] << ") = " << mat_cpu._values[mat_cpu._rowptr[mat_cpu._numrows] - 50] << std::endl;

if (argc > 1)
{

    // assemble rhs
    std::function<float(float, float)> f = [](float x, float y)
                    { return static_cast<float>(2.0) * (x - x*x + y - y*y); };
    size_t numvertices{nodes.size()};
    VectorCpu rhs(numvertices, 0.0);
    for (const auto& e : elements)
    {
        const std::vector<size_t> nodeids = e->vertexids();
        Quadrature<Element, float> quad(e);
        for (size_t i{0}; i < nodeids.size(); ++i)
//            rhs.add(nodeids[i], f(nodes[nodeids[i]].x, nodes[nodeids[i]].y) * quad.integrate_basisfunction(2, i));
            rhs.add(nodeids[i], f(nodes[nodeids[i]].x, nodes[nodeids[i]].y) * quad.integrate_basisfunction(3, i));
    }
    // dirichlet boundary
    for (const auto& n : nodes)
    {
        if (n.x == 0.0 || n.y == 0.0 || n.x == 1.0 || n.y == 1.0)
        {
            for (size_t i{mat_cpu._rowptr[n.id]}; i < mat_cpu._rowptr[n.id + 1]; ++i)
                mat_cpu._values[i] = 0.0f;
            mat_cpu.set(n.id, n.id, 1.0f);
            rhs.set(n.id, 0.0f);
        }
    }

    // solve LGS
    std::cout << "solve" << std::flush;
    CgSolver<CsrMatrixCpu<float>, VectorCpu> solver(mat_cpu, rhs);
    VectorCpu res(numvertices, 0.1);
    solver.solve(res);
    std::cout << " - done" << std::endl;

    // write vtk-file
    ofstream output("../data/gpu.vtk");
    output << "# vtk DataFile Version 3.0" << std::endl;
    output << "square q2" << std::endl;
    output << "ASCII" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID" << std::endl;
    output << std::endl;
    output << "POINTS " << numvertices << " float" << std::endl;
    for (const auto& n : nodes)
        output << n.x << " " << n.y << " 0" << std::endl;
    output << std::endl;
    output << "CELLS " << elements.size() << " " << 5*elements.size() << std::endl;
    for (const auto& e : elements)
        output << "4 " << e->vertexids()[0] << " " << e->vertexids()[1] << " " << e->vertexids()[2] << " " << e->vertexids()[3] << std::endl;
    output << std::endl;
    output << "CELL_TYPES " << elements.size() << std::endl;
    for (size_t i{0}; i < elements.size(); ++i)
        output << "9" << std::endl; // QuadrilateralQ1
    output << std::endl;
    output << "POINT_DATA " << numvertices << std::endl;
    output << "SCALARS u " << (typeid(float) == typeid(float) ? "float" : "float") << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    for (size_t i{0}; i < numvertices; ++i)
        output << (std::abs(res._values[i]) < 0.0001 ? 0 : res._values[i]) << std::endl;
    output.close();
}

    free_cuda(d_localmatrices);





/*
    CsrMatrixCpu<float> mat(nodes.size());
    std::cout << "structure" << std::flush;
    walltime[0] = omp_get_wtime();
    structure(mat, elements);
    walltime[0] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[0] * 1000.0 << ")" << std::endl;
    std::cout << "assemble" << std::flush;
    walltime[1] = omp_get_wtime();

    //float qp{std::sqrt(0.6)};
    float weight[9] = {25.0, 40.0, 25.0,
                        40.0, 64.0, 40.0,
                        25.0, 40.0, 25.0};
    float quadpoint[9][2] = {{-qp, -qp},
                              {-qp, 0.0},
                              {-qp,  qp},
                              {0.0, -qp},
                              {0.0, 0.0},
                              {0.0,  qp},
                              { qp, -qp},
                              { qp, 0.0},
                              { qp,  qp}};
    // function begin: assemble
    const size_t num_elem{elements.size()};
    #pragma omp parallel for num_threads(16)
    for (size_t e=0; e < num_elem; e++)
    {
        const std::vector<size_t> vertexids{elements[e]->vertexids()}; // only needed for vertexids.size() -> always known (=8)
        std::array<float, 8> coords = static_cast<QuadrilateralQ1<float>*>(elements[e])->get_pointcoords();
        for (size_t i{0}; i < 4; ++i)
        {
            for (size_t j{0}; j < 4; ++j)
            {
                float val{0.0};

                for (size_t p{0}; p < 9; ++p)
                {
                    float xi  = quadpoint[p][0];
                    float eta = quadpoint[p][1];
                    float B[2][2] =
                        { { ( -(1.0f-eta)*coords[0] + (1.0f-eta)*coords[1] + (1.0f+eta)*coords[2] - (1.0f+eta)*coords[3] ) * 0.25f ,
                            ( -(1.0f-xi )*coords[0] - (1.0f+xi )*coords[1] + (1.0f+xi )*coords[2] + (1.0f-xi )*coords[3] ) * 0.25f },
                          { ( -(1.0f-eta)*coords[4] + (1.0f-eta)*coords[5] + (1.0f+eta)*coords[6] - (1.0f+eta)*coords[7] ) * 0.25f ,
                            ( -(1.0f-xi )*coords[4] - (1.0f+xi )*coords[5] + (1.0f+xi )*coords[6] + (1.0f-xi )*coords[7] ) * 0.25f } };

                    // help vars
                    std::array<float, 2> grad1;
                    std::array<float, 2> grad2;
                    if (i == 0)
                        grad1 = {(1.0f - eta) * (-0.25f) ,
                                 (1.0f - xi ) * (-0.25f) };
                    else if (i == 1)
                        grad1 = {(1.0f - eta) *   0.25f ,
                                 (1.0f + xi ) * (-0.25f) };
                    else if (i == 2)
                        grad1 = {(1.0f + eta) *   0.25f ,
                                 (1.0f + xi ) *   0.25f };
                    else //if (i == 3)
                        grad1 = {(1.0f + eta) * (-0.25f) ,
                                 (1.0f - xi ) *   0.25f };
                    if (j == 0)
                        grad2 = {(1.0f - eta) * (-0.25f) ,
                                 (1.0f - xi ) * (-0.25f) };
                    else if (j == 1)
                        grad2 = {(1.0f - eta) *   0.25f ,
                                 (1.0f + xi ) * (-0.25f) };
                    else if (j == 2)
                        grad2 = {(1.0f + eta) *   0.25f ,
                                 (1.0f + xi ) *   0.25f };
                    else //if (j == 3)
                        grad2 = {(1.0f + eta) * (-0.25f) ,
                                 (1.0f - xi ) *   0.25f };

                    val  += weight[p]
                            * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
                                + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
                            / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
                } // end for p (quadrature point)
                val /= 81.0f; // all weights are .../81
                //val *= 0.0123456790123456790123;
                mat.add(vertexids[i], vertexids[j], val);
            } // end for j
        } // end for i
    } // end for elements

    // function end: assemble
    walltime[1] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[1] * 1000.0 << ")" << std::endl;
    std::cout << "num values: " << mat._rowptr[mat._numrows] << std::endl;
    for (int k(0); k < OUTPUTSIZE; ++k)
        std::cout << "(0, " << mat._colind[mat._rowptr[mat._numrows] - k] << ") = " << mat._values[mat._rowptr[mat._numrows] - k] << std::endl;
        //std::cout << "(0, " << mat._colind[k] << ") = " << mat._values[k] << std::endl;
    std::cout << "extra: " << mat._colind[mat._rowptr[mat._numrows] - 50] << ") = " << mat._values[mat._rowptr[mat._numrows] - 50] << std::endl;
*/

    return 0;
}
