#include <iostream>
#include <ctime>
#include "include/csrmatrix.hpp"
#include "include/global.hpp"
#include "include/readmesh.hpp"
#include "include/assemblecpu.hpp"
#include "include/assemblegpu.hpp"

void fillFullElements(vector<Node>& nodes, vector<Triangle>& elements, vector<FullTriangle>& fullElements);
//void assemble_gpu_atomic(CsrMatrix& matrix, std::vector<FullTriangle>& elements);

int main()
{
    std::cout << "start demo" << std::endl;
    vector<Node> nodes;
    vector<Triangle> elements;
    vector<FullTriangle> fullElements;
    readmesh("../data/square_small.msh", nodes, elements, fullElements);
    std::cout << "read mesh" << std::endl;
    fillFullElements(nodes, elements, fullElements);

    std::cout << "start assembling" << std::endl;
    clock_t time(clock());
    CsrMatrix matrix(nodes.size());
    matrix.createStructure(elements.data(), elements.size());
    time -= clock();
    std::cout << float(-time) / CLOCKS_PER_SEC << std::endl;

    time = clock();
    assemble_cpu_elem(matrix, fullElements);
    time -= clock();
    std::cout << float(-time) / CLOCKS_PER_SEC << std::endl;
//    matrix.print_local_data(1);

    time = clock();
    assemble_gpu_atomic(matrix, fullElements);
    //assemble_gpu_atomic(matrix._rowptr, matrix._colind, matrix._values, matrix._numrows, fullElements.data(), fullElements.size());
    time -= clock();
    std::cout << float(-time) / CLOCKS_PER_SEC << std::endl;
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
}
