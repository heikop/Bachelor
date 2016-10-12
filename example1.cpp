#include <iostream>
#include <ctime>
#include <cmath>

#include "include/global.hpp"

#include "include/readmesh.hpp"
#include "include/assemble.hpp"

#include "include/csrmatrixcpu.hpp"
#include "include/vectorcpu.hpp"

#include "include/cgsolver.hpp"

int main()
{
    clock_t time[2];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Node>       nodes;
    std::vector<TriangleQ2> elements;
    size_t numnodes;
    size_t num_edgenodes{0};

    std::string filename{"../data/square_fine_q2.msh"};

    mesh_q2_id(filename, nodes, elements, num_edgenodes);
    std::cout << nodes.size() << std::endl;
    std::cout << elements.size() << std::endl;
    numnodes = nodes.size();

    std::cout << "assemble" << std::flush;
    time[0] = clock();
    CsrMatrixCpu mat(numnodes);
    structure_id(mat, elements);
    time[0] -= clock();
    time[1] = clock();
    assemble_id(mat, nodes, elements);
    time[1] -= clock();
    std::cout << " - done (" << float(-time[0]) / CLOCKS_PER_SEC * 1000.0f << ", " << float(-time[1]) / CLOCKS_PER_SEC * 1000.0f << ")" << std::endl;

    // assemble rhs
    VectorCpu rhs(numnodes, 0.0);
    for (const auto& e : elements)
    {
        const double a{nodes[e.nodeB].x - nodes[e.nodeA].x};
        const double c{nodes[e.nodeB].y - nodes[e.nodeA].y};
        const double b{nodes[e.nodeC].x - nodes[e.nodeA].x};
        const double d{nodes[e.nodeC].y - nodes[e.nodeA].y};
        const double detB{std::abs(a*d - b*c)};
        rhs.add_local(e.nodeA, (-27.0/96.0 * (1.0 - 1.0/3.0 - 1.0/3.0)*(1.0 - 2.0/3.0 - 2.0/3.0)
                               + 25.0/96.0 * (1.0 - 1.0/5.0 - 3.0/5.0)*(1.0 - 2.0/5.0 - 6.0/5.0)
                               + 25.0/96.0 * (1.0 - 1.0/5.0 - 1.0/5.0)*(1.0 - 2.0/5.0 - 2.0/5.0)
                               + 25.0/96.0 * (1.0 - 3.0/5.0 - 1.0/5.0)*(1.0 - 6.0/5.0 - 2.0/5.0) )
                               * detB * (2.0) * (nodes[e.nodeA].x * (1.0 - nodes[e.nodeA].x) + nodes[e.nodeA].y * (1.0 - nodes[e.nodeA].y) ) );
        rhs.add_local(e.nodeB, (-27.0/96.0 * 1.0/3.0 * (2.0*1.0/3.0 - 1.0)
                               + 25.0/96.0 * 1.0/5.0 * (2.0*1.0/5.0 - 1.0)
                               + 25.0/96.0 * 1.0/5.0 * (2.0*1.0/5.0 - 1.0)
                               + 25.0/96.0 * 3.0/5.0 * (2.0*3.0/5.0 - 1.0) )
                               * detB * (2.0) * (nodes[e.nodeB].x * (1.0 - nodes[e.nodeB].x) + nodes[e.nodeB].y * (1.0 - nodes[e.nodeB].y) ) );
        rhs.add_local(e.nodeC, (-27.0/96.0 * 1.0/3.0 * (2.0*1.0/3.0 - 1.0)
                               + 25.0/96.0 * 3.0/5.0 * (2.0*3.0/5.0 - 1.0)
                               + 25.0/96.0 * 1.0/5.0 * (2.0*1.0/5.0 - 1.0)
                               + 25.0/96.0 * 1.0/5.0 * (2.0*1.0/5.0 - 1.0) )
                               * detB * (2.0) * (nodes[e.nodeC].x * (1.0 - nodes[e.nodeC].x) + nodes[e.nodeC].y * (1.0 - nodes[e.nodeC].y) ) );
        rhs.add_local(e.nodeD, (-27.0/96.0 * 4.0 * 1.0/3.0 *(1.0 - 1.0/3.0 - 1.0/3.0)
                               + 25.0/96.0 * 4.0 * 1.0/5.0 *(1.0 - 1.0/5.0 - 3.0/5.0)
                               + 25.0/96.0 * 4.0 * 1.0/5.0 *(1.0 - 1.0/5.0 - 1.0/5.0)
                               + 25.0/96.0 * 4.0 * 3.0/5.0 *(1.0 - 3.0/5.0 - 1.0/5.0) )
                               * detB * (2.0) * (nodes[e.nodeD].x * (1.0 - nodes[e.nodeD].x) + nodes[e.nodeD].y * (1.0 - nodes[e.nodeD].y) ) );
        rhs.add_local(e.nodeE, (-27.0/96.0 * 4.0 * 1.0/3.0 * 1.0/3.0
                               + 25.0/96.0 * 4.0 * 1.0/5.0 * 3.0/5.0
                               + 25.0/96.0 * 4.0 * 1.0/5.0 * 1.0/5.0
                               + 25.0/96.0 * 4.0 * 3.0/5.0 * 1.0/5.0 )
                               * detB * (2.0) * (nodes[e.nodeE].x * (1.0 - nodes[e.nodeE].x) + nodes[e.nodeE].y * (1.0 - nodes[e.nodeE].y) ) );
        rhs.add_local(e.nodeF, (-27.0/96.0 * 4.0 * 1.0/3.0 *(1.0 - 1.0/3.0 - 1.0/3.0)
                               + 25.0/96.0 * 4.0 * 3.0/5.0 *(1.0 - 1.0/5.0 - 3.0/5.0)
                               + 25.0/96.0 * 4.0 * 1.0/5.0 *(1.0 - 1.0/5.0 - 1.0/5.0)
                               + 25.0/96.0 * 4.0 * 1.0/5.0 *(1.0 - 3.0/5.0 - 1.0/5.0) )
                               * detB * (2.0) * (nodes[e.nodeF].x * (1.0 - nodes[e.nodeF].x) + nodes[e.nodeF].y * (1.0 - nodes[e.nodeF].y) ) );
    }
    // dirichlet boundary
    for (const auto& n : nodes)
    {
        if (n.x == 0.0 || n.y == 0.0 || n.x == 1.0 || n.y == 1.0)
        {
            for (size_t i{mat._rowptr[n.ID]}; i < mat._rowptr[n.ID + 1]; ++i)
                mat._values[i] = 0.0f;
            mat.set_local(n.ID, n.ID, 1.0f);
            rhs.set_local(n.ID, 0.0f);
        }
    }

    // solve LGS
    CgSolver<CsrMatrixCpu, VectorCpu> solver(mat, rhs);
    VectorCpu res(numnodes, 0.1);
    solver.solve(res);

    // write vtk-file
    ofstream output("../data/square_q2.vtk");
    output << "# vtk DataFile Version 3.0" << std::endl;
    output << "square q2" << std::endl;
    output << "ASCII" << std::endl;
    output << "DATASET UNSTRUCTURED_GRID" << std::endl;
    output << std::endl;
    output << "POINTS " << numnodes << " double" << std::endl;
    for (const auto& n : nodes)
        output << n.x << " " << n.y << " 0" << std::endl;
    output << std::endl;
    output << "CELLS " << elements.size() << " " << 4*elements.size() << std::endl;
    for (const auto& e : elements)
        output << "3 " << e.nodeA << " " << e.nodeB << " " << e.nodeC << std::endl;
    output << std::endl;
    output << "CELL_TYPES " << elements.size() << std::endl;
    for (size_t i{0}; i < elements.size(); ++i)
        output << "5" << std::endl;
    output << std::endl;
    output << "POINT_DATA " << numnodes << std::endl;
    output << "SCALARS u double" << std::endl;
    output << "LOOKUP_TABLE default" << std::endl;
    for (size_t i{0}; i < numnodes; ++i)
        output << (std::abs(res._values[i]) < 0.0001 ? 0 : res._values[i]) << std::endl;
    output.close();

    return 0;
}
