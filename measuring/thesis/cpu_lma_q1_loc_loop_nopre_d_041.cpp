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

#include <omp.h>
#include <stdio.h>

using namespace std;

int main(int argc, char* argv[])
{
    const size_t OUTPUTSIZE = 10;
    if (argc > 1)
        std::cout << "incl solving and output" << std::endl;

    double walltime[2];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Vertex<double>> nodes;
    std::vector<Element<double>*> elements;

    file_to_mesh_all("../data/square_quadrilateral_q1_d.msh", nodes, elements); // Q1
    std::cout << "num nodes: " << nodes.size() << std::endl;
    std::cout << "num elements: " << elements.size() << std::endl;

    CsrMatrixCpu<double> mat(nodes.size());
    double* localmatrices = new double[16*elements.size()];
    std::cout << "structure" << std::flush;
    walltime[0] = omp_get_wtime();
    structure(mat, elements);
    walltime[0] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[0] * 1000.0 << ")" << std::endl;
    std::cout << "assemble" << std::flush;
    walltime[1] = omp_get_wtime();

    double qp{std::sqrt(0.6)};
    double weight[9] = {25.0/81.0*0.0625, 40.0/81.0*0.0625, 25.0/81.0*0.0625,
                       40.0/81.0*0.0625, 64.0/81.0*0.0625, 40.0/81.0*0.0625,
                       25.0/81.0*0.0625, 40.0/81.0*0.0625, 25.0/81.0*0.0625};
    double quadpoint[9][2] = {{ -qp,  -qp},
                             { -qp, 0.0 },
                             { -qp,   qp},
                             {0.0 ,  -qp},
                             {0.0 , 0.0 },
                             {0.0 ,   qp},
                             {  qp,  -qp},
                             {  qp, 0.0 },
                             {  qp,   qp}};

    // function begin: assemble
    const size_t num_elem{elements.size()};

    #pragma omp parallel for num_threads(16)
    for (size_t e=0; e < num_elem; e++) // ROLLED
    {
        const std::vector<size_t> vertexids{elements[e]->vertexids()};      // LOCAL
        std::array<double, 8> coords = static_cast<QuadrilateralQ1<double>*>(elements[e])->get_pointcoords();
        for (size_t i{0}; i < 4; ++i)
        {
            for (size_t j{0}; j < 4; ++j)
            {
                double val{0.0};

                for (size_t p{0}; p < 9; ++p)
                {
                    double xi  = quadpoint[p][0];    // LOCAL
                    double eta = quadpoint[p][1];
                    double B[2][2] =
                        { { ( -(1.0-eta)*coords[0] + (1.0-eta)*coords[1] + (1.0+eta)*coords[2] - (1.0+eta)*coords[3] ) ,
                            ( -(1.0-xi )*coords[0] - (1.0+xi )*coords[1] + (1.0+xi )*coords[2] + (1.0-xi )*coords[3] ) },
                          { ( -(1.0-eta)*coords[4] + (1.0-eta)*coords[5] + (1.0+eta)*coords[6] - (1.0+eta)*coords[7] ) ,
                            ( -(1.0-xi )*coords[4] - (1.0+xi )*coords[5] + (1.0+xi )*coords[6] + (1.0-xi )*coords[7] ) } };

                    // help vars
                    std::array<double, 2> grad1 = (i == 0 ? std::array<double, 2>{-(1.0 - eta),          // LOCAL
                                                                                -(1.0 - xi )} :
                                                 (i == 1 ? std::array<double, 2>{ (1.0 - eta),
                                                                                -(1.0 + xi )} :
                                                 (i == 2 ? std::array<double, 2>{ (1.0 + eta),
                                                                                 (1.0 + xi )} :
                                                           std::array<double, 2>{-(1.0 + eta),
                                                                                 (1.0 - xi )} ) ) );
                    std::array<double, 2> grad2 = (j == 0 ? std::array<double, 2>{-(1.0 - eta),
                                                                                -(1.0 - xi )} :
                                                 (j == 1 ? std::array<double, 2>{ (1.0 - eta),
                                                                                -(1.0 + xi )} :
                                                 (j == 2 ? std::array<double, 2>{ (1.0 + eta),
                                                                                 (1.0 + xi )} :
                                                           std::array<double, 2>{-(1.0 + eta),
                                                                                 (1.0 - xi )} ) ) );

                    val  += weight[p]
                            * (   ( B[1][1] * grad1[0] - B[1][0] * grad1[1]) * ( B[1][1] * grad2[0] - B[1][0] * grad2[1])
                                + (-B[0][1] * grad1[0] + B[0][0] * grad1[1]) * (-B[0][1] * grad2[0] + B[0][0] * grad2[1]) )
                            / std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0]);
                } // end for p (quadrature point)
                localmatrices[16*e + 4*i + j] = val;        // LMA
            } // end for j
        } // end for i
    } // end for elements

    // function end: assemble
    walltime[1] -= omp_get_wtime();
    std::cout << " - done (" << -walltime[1] * 1000.0 << ")" << std::endl;
    std::cout << "num values: " << mat._rowptr[mat._numrows] << std::endl;
    for (int k(0); k < 4; ++k)
    {
        for (int l(0); l < 4; ++l)
            std::cout << localmatrices[k*4 + l] << "    ";
        std::cout << std::endl;
    }
    delete localmatrices;

if (argc > 1)
{
    // assemble rhs
    std::function<double(double, double)> f = [](double x, double y)
                    { return static_cast<double>(2.0) * (x - x*x + y - y*y); };
    size_t numvertices{nodes.size()};
    VectorCpu rhs(numvertices, 0.0);
    for (const auto& e : elements)
    {
        const std::vector<size_t> nodeids = e->vertexids();
        Quadrature<Element, double> quad(e);
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
                mat._values[i] = 0.0;
            mat.set(n.id, n.id, 1.0);
            rhs.set(n.id, 0.0f);
        }
    }

    // solve LGS
    std::cout << "solve" << std::flush;
    CgSolver<CsrMatrixCpu<double>, VectorCpu> solver(mat, rhs);
    VectorCpu res(numvertices, 0.1);
    solver.solve(res);
    std::cout << " - done" << std::endl;

    // write vtk-file
    ofstream output("../data/cpu.vtk");
    output << "# vtk DataFile Version 3.0" << std::endl;
    output << "square q2" << std::endl;
    output << "ASCII" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID" << std::endl;
    output << std::endl;
    output << "POINTS " << numvertices << " double" << std::endl;
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
    output << "SCALARS u " << (typeid(double) == typeid(double) ? "double" : "double") << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    for (size_t i{0}; i < numvertices; ++i)
        output << (std::abs(res._values[i]) < 0.0001 ? 0 : res._values[i]) << std::endl;
    output.close();
}

    return 0;
}
