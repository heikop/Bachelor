#include <iostream>
#include <string>
#include <vector>
#include <ctime>
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

using namespace std;

template<typename datatype>
void read_assemble_calc_post(std::string filename);

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "no input file was given" << std::endl;
        return 1;
    }

    if (argc > 2 && argv[2][0] == 'f')
    {
        std::cout << "datatype is float" << std::endl;
        read_assemble_calc_post<float>(std::string{argv[1]});
    }
    else
    {
        std::cout << "datatype is double" << std::endl;
        read_assemble_calc_post<double>(std::string{argv[1]});
    }
    return 0;
}

template<typename datatype>
void read_assemble_calc_post(std::string filename)
{
    // TODO make something like this work to use user input
    //using datatype = ((argc > 2 && (std::string{argv[2]} == "float")) ? float : double);
    //using typelist[2] = {float, double};
    clock_t time[2];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Vertex<datatype>> nodes;
    std::vector<Element<datatype>*> elements;

    //file_to_mesh(filename, nodes, elements);
    file_to_mesh_all(filename, nodes, elements);
    std::cout << "num nodes: " << nodes.size() << std::endl;
    std::cout << "num elements: " << elements.size() << std::endl;

    std::cout << "structure" << std::flush;
    time[0] = clock();
    CsrMatrixCpu<datatype> mat(nodes.size());
    structure(mat, elements);
    time[0] -= clock();
    std::cout << " - done (" << float(-time[0]) / CLOCKS_PER_SEC * 1000.0f << ")" << std::endl;
    std::cout << "assemble" << std::flush;
    time[1] = clock();
    assemble(mat, elements);
    time[1] -= clock();
    std::cout << " - done (" << float(-time[1]) / CLOCKS_PER_SEC * 1000.0f << ")" << std::endl;

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

/*
    // solve LGS
    CgSolver<CsrMatrixCpu<datatype>, VectorCpu> solver(mat, rhs);
    VectorCpu res(numvertices, 0.1);
    solver.solve(res);

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
    //output << "CELLS " << elements.size() << " " << 4*elements.size() << std::endl;
    output << "CELLS " << elements.size() << " " << 5*elements.size() << std::endl;
    for (const auto& e : elements)
    {
        //for (const auto id : e->vertexids())
            //TODO
    //    output << "3 " << e->vertexids()[0] << " " << e->vertexids()[1] << " " << e->vertexids()[2] << std::endl;
        output << "4 " << e->vertexids()[0] << " " << e->vertexids()[1] << " " << e->vertexids()[2] << " " << e->vertexids()[3] << std::endl;
    }
    output << std::endl;
    output << "CELL_TYPES " << elements.size() << std::endl;
    for (size_t i{0}; i < elements.size(); ++i)
    //if (typeid(*_element) == typeid(TriangleQ1<datatype>)
    //    output << "5" << std::endl; // TriangleQ1
        //output << "22" << std::endl; // TriangleQ2
        output << "9" << std::endl; // QuadrilateralQ1
        //output << "23" << std::endl; // QuadrilateralQ2
    output << std::endl;
    output << "POINT_DATA " << numvertices << std::endl;
    output << "SCALARS u " << (typeid(datatype) == typeid(float) ? "float" : "double") << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    for (size_t i{0}; i < numvertices; ++i)
        output << (std::abs(res._values[i]) < 0.0001 ? 0 : res._values[i]) << std::endl;
    output.close();
*/
}
