#include <iostream>
#include <ctime>
#include "include/global.hpp"
#include "include/readmesh.hpp"
#include "include/csrmatrixcpu.hpp"
#include "include/vectorcpu.hpp"
#include "include/assemble.hpp"

int main()
{
    clock_t time[4];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    vector<Node>       nodes_id,       nodes_full;
    vector<TriangleQ2> elements_id,    elements_full;
    size_t             numnodes_id,    numnodes_full;
    size_t             numelements_id, numelements_full;
    size_t num_edgenodes{0};

    mesh_q2("../data/square_q2.msh", nodes_id, elements_id, num_edgenodes);
    numnodes_id    = nodes_id.size();
    numelements_id = elements_id.size();
    nodes_id.resize(num_edgenodes);
//    mesh_q2("../data/square_q2.msh", nodes_full, elements_full, num_edgenodes);
//    numnodes_id    = nodes_full.size();
//    numelements_id = elements_full.size();
//    nodes_full.resize(num_edgenodes);

    std::cout << std::endl << "IDs" << std::flush;
    time[0] = clock();
    CsrMatrixCpu mat_id(numnodes_id);
    structure_id(mat_id, elements_id);
    time[0] -= clock();
    time[1] = clock();
    assemble_id(mat_id, nodes_id, elements_id);
    time[1] -= clock();
    std::cout << " - done" << std::endl;

//    std::cout << std::endl << "no IDs" << std::flush;
//    time[2] = clock();
//    CsrMatrixCpu mat_full(numnodes_full);
//    structure_full(mat_full, elements_full);
//    time[2] -= clock();
//    time[3] = clock();
//    assemble_full(mat_full, nodes_full, elements_full);
//    time[3] -= clock();
//    std::cout << " - done" << std::endl;

    // nice output
    float duration[4];
    for (size_t i{0}; i < 4; ++i)
        duration[i] = float(-time[i]) / CLOCKS_PER_SEC * 1000.0f;
    std::cout << std::endl;
    std::cout << duration[0] << ", " << duration[1] << std::endl;
    std::cout << duration[2] << ", " << duration[3] << std::endl;

//    std::cout << std::endl;
//    matrix_id.print_local_data();
//    std::cout << std::endl;
//    matrix_full.print_local_data();
//    std::cout << std::endl;

    // matrix check
//    for (size_t i(0); i < matrix_id._numrows_global; ++i)
//    {
//        std::cout << "check row " << i << ": ";
//        for (size_t j(0); j < matrix_id._numcols_global; ++j)
//            assert(matrix_id.get_global(i, j) == matrix_full.get_global(i, j));
//        std::cout << "checked " << std::endl;
//    }

    return 0;
}
