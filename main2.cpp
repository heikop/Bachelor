#include <iostream>
#include <ctime>
#include "include/global.hpp"
#include "include/readmesh.hpp"
#include "include/csrmatrixcpu.hpp"
#include "include/assemblecpu.hpp"
#include "include/vectorcpu.hpp"
#include "include/cgsolver.hpp"

void fillFullElements(vector<Node>& nodes, vector<Triangle>& elements, vector<FullTriangle>& fullElements);

int main()
{
//    clock_t time[2];
//    time[0] = clock();

    vector<Node> nodes;
    vector<Triangle> elements;
    vector<FullTriangle> fullElements;
    vector<size_t> boundaryNodes;
    readmesh("../data/square_evenfiner.msh", nodes, elements, fullElements, boundaryNodes);
    fillFullElements(nodes, elements, fullElements);
//    time[0] -= clock();

//    time[1] = clock();
    CsrMatrixCpu matrix_cpu(nodes.size());
    matrix_cpu.createStructure_fast(elements.data(), elements.size());
//    time[1] -= clock();

//    time[2] = clock();
//    assemble_cpu_elem(matrix_cpu, fullElements, boundaryNodes);
//    time[2] -= clock();

//    float duration[3];
//    duration[0] = float(-time[0]) / CLOCKS_PER_SEC * 1000.0f;
//    duration[1] = float(-time[1]) / CLOCKS_PER_SEC * 1000.0f;
//    duration[2] = float(-time[2]) / CLOCKS_PER_SEC * 1000.0f;
//    std::cout << duration[0] << std::endl;
//    std::cout << duration[1] << std::endl;
//    std::cout << duration[2] << std::endl;

    return 0;
}

void fillFullElements(vector<Node>& nodes, vector<Triangle>& elements, vector<FullTriangle>& fullElements)
{
    fullElements.clear();
    fullElements.resize(elements.size());
    //FullTriangle newelement;
    for (size_t i(0); i < elements.size(); ++i)
    {
        //newelement.ID = elements[i].ID;
        //newelement.nodeA = nodes[elements[i].nodeA];
        //newelement.nodeB = nodes[elements[i].nodeB];
        //newelement.nodeC = nodes[elements[i].nodeC];
        //fullElements.push_back(newelement);
        fullElements[i].ID = elements[i].ID;
        fullElements[i].nodeA = nodes[elements[i].nodeA];
        fullElements[i].nodeB = nodes[elements[i].nodeB];
        fullElements[i].nodeC = nodes[elements[i].nodeC];
    }
    //fullElements.shrink_to_fit();
}
