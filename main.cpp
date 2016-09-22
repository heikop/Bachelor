#include <iostream>
#include <ctime>
#include "include/global.hpp"
#include "include/readmesh.hpp"
#include "include/csrmatrixcpu.hpp"
#include "include/csrmatrixgpu.hpp"
#include "include/assemblecpu.hpp"
#include "include/assemblegpu.hpp"
#include "include/vectorcpu.hpp"
#include "include/vectorgpu.hpp"
#include "include/cgsolver.hpp"

void fillFullElements(vector<Node>& nodes, vector<Triangle>& elements, vector<FullTriangle>& fullElements);

int main()
{
    initCuda();

    clock_t time[7];
    std::cout << "start demo" << std::endl;
    time[0] = clock();
    vector<Node> nodes;
    vector<Triangle> elements;
    vector<FullTriangle> fullElements;
    vector<size_t> boundaryNodes;
    readmesh("../data/square_fine.msh", nodes, elements, fullElements, boundaryNodes);
    std::cout << "read mesh" << std::endl;
    fillFullElements(nodes, elements, fullElements);
    std::cout << nodes.size() << ", " << elements.size() << std::endl;
    time[0] -= clock();

    std::cout << std::endl << "start assembling cpu" << std::endl;
    std::cout << "create structure: ";
    time[1] = clock();
    CsrMatrixCpu matrix_cpu(nodes.size());
    time[1] -= clock();
    time[2] = clock();
    matrix_cpu.createStructure(elements.data(), elements.size());
    time[2] -= clock();
    std::cout << "created" << std::endl;

    std::cout << "assemble matrix: ";
    time[3] = clock();
    assemble_cpu_elem(matrix_cpu, fullElements, boundaryNodes);
    time[3] -= clock();
    std::cout << "assembled" << std::endl;

    std::cout << std::endl << "start assembling gpu" << std::endl;
    std::cout << "create structure: ";
    time[4] = clock();
    CsrMatrixGpu matrix_gpu(nodes.size());
    time[4] -= clock();
    time[5] = clock();
    matrix_gpu.createStructure(elements.data(), elements.size());
    time[5] -= clock();
    std::cout << "created" << std::endl;

    std::cout << "assemble matrix: ";
    time[6] = clock();
    assemble_gpu_atomic(matrix_gpu, fullElements, boundaryNodes);
    time[6] -= clock();
    std::cout << "assembled" << std::endl;

    // nice output
    float duration[5];
    duration[0] = float(-time[0]) / CLOCKS_PER_SEC * 1000.0f;
    duration[1] = float(-time[1]) / CLOCKS_PER_SEC * 1000.0f;
    duration[2] = float(-time[2]) / CLOCKS_PER_SEC * 1000.0f;
    duration[3] = float(-time[3]) / CLOCKS_PER_SEC * 1000.0f;
    duration[4] = float(-time[4]) / CLOCKS_PER_SEC * 1000.0f;
    duration[5] = float(-time[5]) / CLOCKS_PER_SEC * 1000.0f;
    duration[6] = float(-time[6]) / CLOCKS_PER_SEC * 1000.0f;
    std::cout << std::endl;
    std::cout.width(7); std::cout << "part" << "   "; std::cout.width(6); std::cout << "total" << " | ";                 std::cout << "splitted"; std::cout << std::endl;
    std::cout.width(7); std::cout << "mesh" << " : "; std::cout.width(6); std::cout << duration[0] << " | ";             std::cout.width(7); std::cout << ""; std::cout << std::endl;
    std::cout.width(7); std::cout <<  "CPU" << " : "; std::cout.width(6); std::cout << duration[1]+duration[2]+duration[3] << " | "; std::cout.width(7); std::cout << duration[1] << ", "; std::cout.width(7); std::cout << duration[2] << ", "; std::cout.width(7); std::cout << duration[3]; std::cout << std::endl;
    std::cout.width(7); std::cout <<  "GPU" << " : "; std::cout.width(6); std::cout << duration[4]+duration[5]+duration[6] << " | "; std::cout.width(7); std::cout << duration[4] << ", "; std::cout.width(7); std::cout << duration[5] << ", "; std::cout.width(7); std::cout << duration[6]; std::cout << std::endl;

    // matrix check
    /*
    size_t* rowptr_gpu_check = new size_t[matrix_gpu._numrows+1];
    memcpy_cuda(rowptr_gpu_check, matrix_gpu._rowptr, (matrix_gpu._numrows+1)*sizeof(size_t), d2h);
    for (size_t i(0); i <= matrix_gpu._numrows; ++i)
        assert(rowptr_gpu_check[i] == matrix_cpu._rowptr[i]);
    size_t* colind_gpu_check = new size_t[rowptr_gpu_check[matrix_gpu._numrows]];
    memcpy_cuda(colind_gpu_check, matrix_gpu._colind, rowptr_gpu_check[matrix_gpu._numrows]*sizeof(size_t), d2h);
    for (size_t i(0); i < rowptr_gpu_check[matrix_gpu._numrows]; ++i)
        assert(colind_gpu_check[i] == matrix_cpu._colind[i]);
    float* values_gpu_check = new float[rowptr_gpu_check[matrix_gpu._numrows]];
    memcpy_cuda(values_gpu_check, matrix_gpu._values, rowptr_gpu_check[matrix_gpu._numrows]*sizeof(float), d2h);
    for (size_t i(0); i < rowptr_gpu_check[matrix_gpu._numrows]; ++i)
        assert(values_gpu_check[i] == matrix_cpu._values[i]);
    delete[] rowptr_gpu_check;
    delete[] colind_gpu_check;
    delete[] values_gpu_check;
    */

    //if (__mpi_instance__.get_global_rank() == 0)
    //for (size_t i(0); i < matrix_cpu._numrows_global; ++i)
    //{
    //    std::cout << "check row " << i << ": ";
    //    for (size_t j(0); j < matrix_cpu._numcols_global; ++j)
    //        assert(matrix_cpu.get_global(i, j) == matrix_gpu.get_global(i, j));
    //    std::cout << "checked " << std::endl;
    //}

/*
    // calculation - solving LGS
std::cout << "LGS" << std::endl;
    nodes.clear();
    elements.clear();
    fullElements.clear();
    VectorCpu rhs_cpu(matrix_cpu._numrows, 1.0);
    VectorCpu res_cpu(matrix_cpu._numrows, 1.0);
    CgSolver<CsrMatrixCpu, VectorCpu> solver_cpu(matrix_cpu, rhs_cpu);
    solver_cpu.solve(res_cpu);
    VectorGpu rhs_gpu(matrix_gpu._numrows, 1.0);
    VectorGpu res_gpu(matrix_gpu._numrows, 1.0);
    CgSolver<CsrMatrixGpu, VectorGpu> solver_gpu(matrix_gpu, rhs_gpu);
    solver_gpu.solve(res_gpu);
//res_cpu.print_local_data(1);
//res_gpu.print_local_data(1);

    // solution check
    float* res_gpu_check = new float[res_gpu._size];
    memcpy_cuda(res_gpu_check, res_gpu._values, res_gpu._size*sizeof(float), d2h);
    for (size_t i(0); i < res_gpu._size; ++i)
{
        if(std::abs(res_gpu_check[i] - res_cpu._values[i]) > 1.0e-3)
std::cout << i << ": " << std::abs(res_gpu_check[i] - res_cpu._values[i]) << std::endl;
}
    delete[] res_gpu_check;
*/

    return 0;
}


void fillFullElements(vector<Node>& nodes, vector<Triangle>& elements, vector<FullTriangle>& fullElements)
{
    fullElements.clear();
    FullTriangle newelement;
    for (size_t i(0); i < elements.size(); ++i)
    {
        newelement.ID = elements[i].ID;
        newelement.nodeA = nodes[elements[i].nodeA];
        newelement.nodeB = nodes[elements[i].nodeB];
        newelement.nodeC = nodes[elements[i].nodeC];
        fullElements.push_back(newelement);
    }
    fullElements.shrink_to_fit();
}
