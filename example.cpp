#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <cmath>

#include "include/elements.hpp"
#include "include/readfile.hpp"
#include "include/structure.hpp"
#include "include/assemble.hpp"
#include "include/quadrature.hpp"
#include "include/cgsolver.hpp"

#include "include/csrmatrixcpu.hpp"
#include "include/vectorcpu.hpp"

using namespace std;

int main()
{
    clock_t time[2];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Vertex<double>> vertices;
    std::vector<TriangleQ2<double>> elements;

    std::string filename{"../data/square_fine_q2.msh"};
    //std::string filename{"../data/square_veryrough_q2.msh"};

    //mesh_q2_id(filename, nodes, elements, num_edgenodes);
    file_to_mesh(filename, vertices, elements);
    std::cout << vertices.size() << std::endl;
    std::cout << elements.size() << std::endl;

    std::cout << "assemble" << std::flush;
    time[0] = clock();
    CsrMatrixCpu<double> mat(vertices.size());
    structure(mat, elements);
    time[0] -= clock();
    time[1] = clock();
    assemble(mat, elements);
    time[1] -= clock();
    std::cout << " - done (" << float(-time[0]) / CLOCKS_PER_SEC * 1000.0f << ", " << float(-time[1]) / CLOCKS_PER_SEC * 1000.0f << ")" << std::endl;

    // assemble rhs
    size_t numvertices{vertices.size()};
    VectorCpu rhs(numvertices, 0.0);
    for (const auto& e : elements)
    {
        const double a{e._p1.x - e._p0.x};
        const double c{e._p1.y - e._p0.y};
        const double b{e._p2.x - e._p0.x};
        const double d{e._p2.y - e._p0.y};
        const double detB{std::abs(a*d - b*c)};
        rhs.add(e._p0.id, (-27.0/96.0 * (1.0 - 1.0/3.0 - 1.0/3.0)*(1.0 - 2.0/3.0 - 2.0/3.0)
                          + 25.0/96.0 * (1.0 - 1.0/5.0 - 3.0/5.0)*(1.0 - 2.0/5.0 - 6.0/5.0)
                          + 25.0/96.0 * (1.0 - 1.0/5.0 - 1.0/5.0)*(1.0 - 2.0/5.0 - 2.0/5.0)
                          + 25.0/96.0 * (1.0 - 3.0/5.0 - 1.0/5.0)*(1.0 - 6.0/5.0 - 2.0/5.0) )
                          * detB * (2.0) * (e._p0.x * (1.0 - e._p0.x) + e._p0.y * (1.0 - e._p0.y) ) );
        rhs.add(e._p1.id, (-27.0/96.0 * 1.0/3.0 * (2.0*1.0/3.0 - 1.0)
                          + 25.0/96.0 * 1.0/5.0 * (2.0*1.0/5.0 - 1.0)
                          + 25.0/96.0 * 1.0/5.0 * (2.0*1.0/5.0 - 1.0)
                          + 25.0/96.0 * 3.0/5.0 * (2.0*3.0/5.0 - 1.0) )
                          * detB * (2.0) * (e._p1.x * (1.0 - e._p1.x) + e._p1.y * (1.0 - e._p1.y) ) );
        rhs.add(e._p2.id, (-27.0/96.0 * 1.0/3.0 * (2.0*1.0/3.0 - 1.0)
                          + 25.0/96.0 * 3.0/5.0 * (2.0*3.0/5.0 - 1.0)
                          + 25.0/96.0 * 1.0/5.0 * (2.0*1.0/5.0 - 1.0)
                          + 25.0/96.0 * 1.0/5.0 * (2.0*1.0/5.0 - 1.0) )
                          * detB * (2.0) * (e._p2.x * (1.0 - e._p2.x) + e._p2.y * (1.0 - e._p2.y) ) );
        rhs.add(e._p3_id, (-27.0/96.0 * 4.0 * 1.0/3.0 *(1.0 - 1.0/3.0 - 1.0/3.0)
                          + 25.0/96.0 * 4.0 * 1.0/5.0 *(1.0 - 1.0/5.0 - 3.0/5.0)
                          + 25.0/96.0 * 4.0 * 1.0/5.0 *(1.0 - 1.0/5.0 - 1.0/5.0)
                          + 25.0/96.0 * 4.0 * 3.0/5.0 *(1.0 - 3.0/5.0 - 1.0/5.0) )
                          * detB * (2.0) * (vertices[e._p3_id].x * (1.0 - vertices[e._p3_id].x) + vertices[e._p3_id].y * (1.0 - vertices[e._p3_id].y) ) );
        rhs.add(e._p4_id, (-27.0/96.0 * 4.0 * 1.0/3.0 * 1.0/3.0
                          + 25.0/96.0 * 4.0 * 1.0/5.0 * 3.0/5.0
                          + 25.0/96.0 * 4.0 * 1.0/5.0 * 1.0/5.0
                          + 25.0/96.0 * 4.0 * 3.0/5.0 * 1.0/5.0 )
                          * detB * (2.0) * (vertices[e._p4_id].x * (1.0 - vertices[e._p4_id].x) + vertices[e._p4_id].y * (1.0 - vertices[e._p4_id].y) ) );
        rhs.add(e._p5_id, (-27.0/96.0 * 4.0 * 1.0/3.0 *(1.0 - 1.0/3.0 - 1.0/3.0)
                          + 25.0/96.0 * 4.0 * 3.0/5.0 *(1.0 - 1.0/5.0 - 3.0/5.0)
                          + 25.0/96.0 * 4.0 * 1.0/5.0 *(1.0 - 1.0/5.0 - 1.0/5.0)
                          + 25.0/96.0 * 4.0 * 1.0/5.0 *(1.0 - 3.0/5.0 - 1.0/5.0) )
                          * detB * (2.0) * (vertices[e._p5_id].x * (1.0 - vertices[e._p5_id].x) + vertices[e._p5_id].y * (1.0 - vertices[e._p5_id].y) ) );
    }
    // dirichlet boundary
    for (const auto& n : vertices)
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
    CgSolver<CsrMatrixCpu<double>, VectorCpu> solver(mat, rhs);
    VectorCpu res(numvertices, 0.1);
    solver.solve(res);

    // write vtk-file
    ofstream output("../data/square_q2.vtk");
    output << "# vtk DataFile Version 3.0" << std::endl;
    output << "square q2" << std::endl;
    output << "ASCII" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID" << std::endl;
    output << std::endl;
    output << "POINTS " << numvertices << " double" << std::endl;
    for (const auto& n : vertices)
        output << n.x << " " << n.y << " 0" << std::endl;
    output << std::endl;
    output << "CELLS " << elements.size() << " " << 4*elements.size() << std::endl;
    for (const auto& e : elements)
        output << "3 " << e._p0.id << " " << e._p1.id << " " << e._p2.id << std::endl;
    output << std::endl;
    output << "CELL_TYPES " << elements.size() << std::endl;
    for (size_t i{0}; i < elements.size(); ++i)
        output << "5" << std::endl;
    output << std::endl;
    output << "POINT_DATA " << numvertices << std::endl;
    output << "SCALARS u double" << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    for (size_t i{0}; i < numvertices; ++i)
        output << (std::abs(res._values[i]) < 0.0001 ? 0 : res._values[i]) << std::endl;
    output.close();

    return 0;
}
