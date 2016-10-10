#include <iostream>
#include <ctime>
#include "include/global.hpp"
#include "include/readmesh.hpp"
#include "include/csrmatrixcpu.hpp"
#include "include/vectorcpu.hpp"
#include "include/assemble.hpp"

int main()
{
    clock_t time[4][11];
    std::cout << ">- CPU: MESH FORMAT, Q2 -<" << std::endl;
    std::vector<Node>       nodes_id;
    std::vector<TriangleQ2> elements_id;
    std::vector<FullTriangleQ2> elements_full;
    size_t             numnodes_id,    numnodes_full;
    size_t num_edgenodes{0};

    std::string filename{"../data/square_veryrough_q2.msh"};

    nodes_id.resize(num_edgenodes);
    mesh_q2_full(filename, elements_full, numnodes_full);
    mesh_q2_id(filename, nodes_id, elements_id, num_edgenodes);
    numnodes_id    = nodes_id.size();

    for (size_t i{0}; i < 11; ++i)
    {
        std::cout << std::endl << "Full nodes" << std::flush;
        time[2][i] = clock();
        CsrMatrixCpu mat_full(numnodes_full);
        structure_full(mat_full, elements_full);
        time[2][i] -= clock();
        time[3][i] = clock();
        assemble_full(mat_full, elements_full);
        time[3][i] -= clock();
        std::cout << " - done (" << float(-time[2][i]) / CLOCKS_PER_SEC * 1000.0f << ", " << float(-time[3][i]) / CLOCKS_PER_SEC * 1000.0f << ")" << std::endl;
    }
    for (size_t i{0}; i < 11; ++i)
    {
        std::cout << std::endl << "IDs" << std::flush;
        time[0][i] = clock();
        CsrMatrixCpu mat_id(numnodes_id);
        structure_id(mat_id, elements_id);
        time[0][i] -= clock();
        time[1][i] = clock();
        assemble_id(mat_id, nodes_id, elements_id);
        time[1][i] -= clock();
        std::cout << " - done (" << float(-time[0][i]) / CLOCKS_PER_SEC * 1000.0f << ", " << float(-time[1][i]) / CLOCKS_PER_SEC * 1000.0f << ")" << std::endl;
    }


    // nice output
    float duration[4]{0.0, 0.0, 0.0, 0.0};
    for (size_t i{0}; i < 4; ++i)
        for (size_t j{1}; j < 11; ++j)
            duration[i] += float(-time[i][j]) / CLOCKS_PER_SEC * 1000.0f;
    std::cout << std::endl;
    std::cout << duration[0]/10.0 << ", " << duration[1]/10.0 << std::endl;
    std::cout << duration[2]/10.0 << ", " << duration[3]/10.0 << std::endl;

//    std::cout << std::endl;
//    matrix_id.print_local_data();
//    std::cout << std::endl;
//    matrix_full.print_local_data();
//    std::cout << std::endl;

    // matrix check
//    for (size_t i(0); i < mat_id._numrows_global; ++i)
//    {
//        std::cout << "check row " << i;
//        for (size_t j(0); j < mat_id._numcols_global; ++j)
//            assert(mat_id.get_global(i, j) == mat_full.get_global(i, j));
//        std::cout << " - checked" << std::endl;
//    }

    return 0;
}
