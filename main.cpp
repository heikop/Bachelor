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
    clock_t time[5];
    std::cout << "start demo" << std::endl;
    time[0] = clock();
    vector<Node> nodes;
    vector<Triangle> elements;
    vector<FullTriangle> fullElements;
    //readmesh("../data/square_evenfiner.msh", nodes, elements, fullElements);
    readmesh("../data/square_rough.msh", nodes, elements, fullElements);
    std::cout << "read mesh" << std::endl;
    fillFullElements(nodes, elements, fullElements);
    time[0] -= clock();
//    std::cout << "startup: " << float(-time[0]) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

    std::cout << "start assembling cpu" << std::endl;
    time[1] = clock();
    CsrMatrixCpu matrix_cpu(nodes.size());
    matrix_cpu.createStructure(elements.data(), elements.size());
    time[1] -= clock();
//    std::cout << "createStructure: " << float(-time[1]) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

    time[2] = clock();
    assemble_cpu_elem(matrix_cpu, fullElements);
    time[2] -= clock();
//    std::cout << "assembly: " << float(-time[2]) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
//    std::cout << "cpu total: " << float(-time[1]-time[2]) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
//    matrix_cpu.print_local_data(1);

    std::cout << "start assembling gpu" << std::endl;
    time[3] = clock();
    CsrMatrixGpu matrix_gpu(nodes.size());
    matrix_gpu.createStructure(elements.data(), elements.size());
    time[3] -= clock();
//    std::cout << "createStructure: " << float(-time[3]) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

    time[4] = clock();
    assemble_gpu_atomic(matrix_gpu, fullElements);
    time[4] -= clock();
//    std::cout << "assembly: " << float(-time[4]) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
//    std::cout << "gpu total: " << float(-time[3]-time[4]) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
//    matrix_gpu.print_local_data(1);

    // check
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

    // nice output
    float duration[5];
    duration[0] = float(-time[0]) / CLOCKS_PER_SEC * 1000.0f;
    duration[1] = float(-time[1]) / CLOCKS_PER_SEC * 1000.0f;
    duration[2] = float(-time[2]) / CLOCKS_PER_SEC * 1000.0f;
    duration[3] = float(-time[3]) / CLOCKS_PER_SEC * 1000.0f;
    duration[4] = float(-time[4]) / CLOCKS_PER_SEC * 1000.0f;
    std::cout << std::endl;
    std::cout.width(7); std::cout << "part" << "   "; std::cout.width(6); std::cout << "total" << " | ";                 std::cout << "splitted"; std::cout << std::endl;
    std::cout.width(7); std::cout << "mesh" << " : "; std::cout.width(6); std::cout << duration[0] << " | ";             std::cout.width(7); std::cout << ""; std::cout << std::endl;
    std::cout.width(7); std::cout <<  "CPU" << " : "; std::cout.width(6); std::cout << duration[1]+duration[2] << " | "; std::cout.width(7); std::cout << duration[1] << ", "; std::cout.width(7); std::cout << duration[2]; std::cout << std::endl;
    std::cout.width(7); std::cout <<  "GPU" << " : "; std::cout.width(6); std::cout << duration[3]+duration[4] << " | "; std::cout.width(7); std::cout << duration[3] << ", "; std::cout.width(7); std::cout << duration[4]; std::cout << std::endl;


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
