#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <typeinfo>
#include <functional>

#include "include/elements.hpp"
#include "include/readfile.hpp"
#include "include/structure.hpp"
#include "include/assemble.hpp"
#include "include/quadrature.hpp"
#include "include/cgsolver.hpp"

#include "include/csrmatrixcpu.hpp"
#include "include/vectorcpu.hpp"

#include <omp.h>
#include <stdio.h>

#define NUMOUTERRUNS 3
#define NUMINNERRUNS 30

using namespace std;

template<typename datatype>
void read_assemble_calc_post(std::string filename, bool solve);

int main(int argc, char* argv[])
{
    //#pragma omp parallel for
    //for (int i=0; i < 10; i++)
    //    //std::cout << i << ": " << omp_get_thread_num() << std::endl;
    //    printf("threadID: %d - %d\n", omp_get_thread_num(), i);
    if (argc < 2)
    {
        std::cout << "no input file was given" << std::endl;
        return 1;
    }

    if (argc > 2 && argv[2][0] == 'f')
    {
        std::cout << "datatype is float" << std::endl;
        if (argc > 3)
            read_assemble_calc_post<float>(std::string{argv[1]}, true);
        else
            read_assemble_calc_post<float>(std::string{argv[1]}, false);
    }
    else
    {
        std::cout << "datatype is double" << std::endl;
        if (argc > 3)
            read_assemble_calc_post<double>(std::string{argv[1]}, true);
        else
            read_assemble_calc_post<double>(std::string{argv[1]}, false);
    }
    return 0;
}

template<typename datatype>
void read_assemble_calc_post(std::string filename, bool solve)
{
    // TODO make something like this work to use user input
    //using datatype = ((argc > 2 && (std::string{argv[2]} == "float")) ? float : double);
    //using typelist[2] = {float, double};

    //double walltime[2];
    double walltime[NUMOUTERRUNS][NUMINNERRUNS+10];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Vertex<datatype>> nodes;
    std::vector<Element<datatype>*> elements;

    //file_to_mesh(filename, nodes, elements);
    file_to_mesh_all(filename, nodes, elements);
//    quad_to_tri(nodes, elements);
    std::cout << "num nodes: " << nodes.size() << std::endl;
    std::cout << "num elements: " << elements.size() << std::endl;

    CsrMatrixCpu<datatype> mat(nodes.size());
    for (size_t outer{0}; outer < NUMOUTERRUNS; ++outer)
    {
        std::cout << "===== ===== RUN " << outer + 1 << " ===== =====" << std::endl;
        std::cout << "structure" << std::flush;
        walltime[outer][0] = omp_get_wtime();
        structure(mat, elements);
        walltime[outer][0] -= omp_get_wtime();
        std::cout << " - done (" << -walltime[outer][0] * 1000.0 << ")" << std::endl;
        std::cout << "assemble" << std::flush;
        for (size_t inner{0}; inner < NUMINNERRUNS+10; ++inner) // puffer of 5 at the beginning and end
        {
            walltime[outer][inner] = omp_get_wtime();
            assemble(mat, elements);
            walltime[outer][inner] -= omp_get_wtime();
            std::cout << " - done (" << -walltime[outer][inner] * 1000.0 << ")" << std::endl;
        }
        double avg{0.0};
        for (size_t inner{5}; inner < NUMINNERRUNS+5; ++inner) // dont add puffer runs
            avg -= walltime[outer][inner];
        std::cout << ">>>>> average of run " << outer + 1 << ": " << avg * 1000.0 / static_cast<double>(NUMINNERRUNS) << "ms" << std::endl;
    }

    // assemble rhs
    std::function<datatype(datatype, datatype)> f = [](datatype x, datatype y)
                    { return static_cast<datatype>(2.0) * (x - x*x + y - y*y); };
    size_t numvertices{nodes.size()};
    VectorCpu rhs(numvertices, 0.0);
    for (const auto& e : elements)
    {
        const std::vector<size_t> nodeids = e->vertexids();
        Quadrature<Element, datatype> quad(e);
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


    if (solve)
    {
    // solve LGS
    std::cout << "solve" << std::flush;
    CgSolver<CsrMatrixCpu<datatype>, VectorCpu> solver(mat, rhs);
    VectorCpu res(numvertices, 0.1);
    solver.solve(res);
    std::cout << " - done" << std::endl;

    bool triangles_used = true; // false -> quadrilaterals
    // write vtk-file
    ofstream output("../data/square_q2.vtk");
    output << "# vtk DataFile Version 3.0" << std::endl;
    output << "square q2" << std::endl;
    output << "ASCII" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID" << std::endl;
    output << std::endl;
    output << "POINTS " << numvertices << (typeid(datatype) == typeid(float) ? " float" : " double") << std::endl;
    for (const auto& n : nodes)
        output << n.x << " " << n.y << " 0" << std::endl;
    output << std::endl;
    if (triangles_used) // bad style
        output << "CELLS " << elements.size() << " " << 4*elements.size() << std::endl;
    else
        output << "CELLS " << elements.size() << " " << 5*elements.size() << std::endl;
    for (const auto& e : elements)
    {
        //for (const auto id : e->vertexids())
            //TODO
        if (typeid(*e) == typeid(TriangleQ1<datatype>)
         || typeid(*e) == typeid(TriangleQ2<datatype>) )
            output << "3 " << e->vertexids()[0] << " " << e->vertexids()[1] << " " << e->vertexids()[2] << std::endl;
        else if (typeid(*e) == typeid(QuadrilateralQ1<datatype>)
              || typeid(*e) == typeid(QuadrilateralQ2<datatype>) )
            output << "4 " << e->vertexids()[0] << " " << e->vertexids()[1] << " " << e->vertexids()[2] << " " << e->vertexids()[3] << std::endl;
        else
            assert(false);
    }
    output << std::endl;
    output << "CELL_TYPES " << elements.size() << std::endl;
    for (size_t i{0}; i < elements.size(); ++i)
    {
        if (typeid(*(elements[i])) == typeid(TriangleQ1<datatype>)
         || typeid(*(elements[i])) == typeid(TriangleQ2<datatype>) )
            output << "5" << std::endl; // TriangleQ1
        //output << "22" << std::endl; // TriangleQ2
        else if (typeid(*(elements[i])) == typeid(QuadrilateralQ1<datatype>)
              || typeid(*(elements[i])) == typeid(QuadrilateralQ2<datatype>) )
            output << "9" << std::endl; // QuadrilateralQ1
        //output << "23" << std::endl; // QuadrilateralQ2
        else
            assert(false);
    }
    output << std::endl;
    output << "POINT_DATA " << numvertices << std::endl;
    output << "SCALARS u " << (typeid(datatype) == typeid(float) ? "float" : "double") << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    for (size_t i{0}; i < numvertices; ++i)
        output << (std::abs(res._values[i]) < 0.0001 ? 0 : res._values[i]) << std::endl;
    output.close();
    }
}
