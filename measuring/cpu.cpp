#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <typeinfo>
#include <functional>

#include "../include/elements.hpp"
#include "../include/readfile.hpp"
#include "../include/structure.hpp"
#include "../include/quadrature.hpp"
#include "../include/cgsolver.hpp"

#include "../include/csrmatrixcpu.hpp"
#include "../include/vectorcpu.hpp"

#include <omp.h>
#include <stdio.h>

using namespace std;

int main(int argc, char* argv[])
{
    if (argc > 1)
        std::cout << "incl solving and output" << std::endl;

    float walltime[2];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Vertex<float>> nodes;
    std::vector<Element<float>*> elements;

    file_to_mesh_all("../data/square_quadrilateral_q1_b.msh", nodes, elements);
    std::cout << "num nodes: " << nodes.size() << std::endl;
    std::cout << "num elements: " << elements.size() << std::endl;

    CsrMatrixCpu<float> mat(nodes.size());
    std::cout << "structure" << std::flush;
    walltime[0] = omp_get_wtime();
    structure(mat, elements);
    walltime[0] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[0] * 1000.0 << ")" << std::endl;
    std::cout << "assemble" << std::flush;
    walltime[1] = omp_get_wtime();

    float qp{std::sqrt(0.6)};
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
    #pragma omp parallel for// num_threads(16)
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
    for (int k(0); k < 10; ++k)
        std::cout << "(0, " << mat._colind[k] << ") = " << mat._values[k] << std::endl;

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
            for (size_t i{mat._rowptr[n.id]}; i < mat._rowptr[n.id + 1]; ++i)
                mat._values[i] = 0.0f;
            mat.set(n.id, n.id, 1.0f);
            rhs.set(n.id, 0.0f);
        }
    }

    // solve LGS
    std::cout << "solve" << std::flush;
    CgSolver<CsrMatrixCpu<float>, VectorCpu> solver(mat, rhs);
    VectorCpu res(numvertices, 0.1);
    solver.solve(res);
    std::cout << " - done" << std::endl;

    // write vtk-file
    ofstream output("../data/cpp.vtk");
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

    return 0;
}
