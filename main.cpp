#include <iostream>
#include <ctime>
#include "include/global.hpp"
#include "include/readmesh.hpp"
#include "include/csrmatrixcpu.hpp"
#include "include/csrmatrixgpu.hpp"
#include "include/assemblecpu.hpp"
#include "include/assemblegpu.hpp"

void fillFullElements(vector<Node>& nodes, vector<Triangle>& elements, vector<FullTriangle>& fullElements);

int main()
{
    std::cout << "start demo" << std::endl;
    vector<Node> nodes;
    vector<Triangle> elements;
    vector<FullTriangle> fullElements;
    readmesh("../data/square.msh", nodes, elements, fullElements);
    std::cout << "read mesh" << std::endl;
    fillFullElements(nodes, elements, fullElements);
    nodes.clear();

    std::cout << "start assembling cpu" << std::endl;
    clock_t time(clock());
    CsrMatrixCpu matrix_cpu(nodes.size());
    matrix_cpu.createStructure(elements.data(), elements.size());
    time -= clock();
    std::cout << "createStructure: " << float(-time) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

    time = clock();
    assemble_cpu_elem(matrix_cpu, fullElements);
    time -= clock();
    std::cout << "assembly: " << float(-time) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
//    matrix.print_local_data(1);

    std::cout << "start assembling gpu" << std::endl;
    time = clock();
    CsrMatrixGpu matrix_gpu(nodes.size());
    matrix_gpu.createStructure(elements.data(), elements.size());
    time -= clock();
    std::cout << "createStructure: " << float(-time) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;

    time = clock();
    assemble_gpu_atomic(matrix_gpu, fullElements);
    time -= clock();
    std::cout << "assembly: " << float(-time) / CLOCKS_PER_SEC * 1000 << "ms" << std::endl;
//    matrix.print_local_data(1);

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
