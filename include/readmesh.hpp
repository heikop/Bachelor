#ifndef __READMESH_HPP_
#define __READMESH_HPP_

#include <iostream>
#include <fstream>
#include <vector>
//#include <sstream>

using namespace std;

void readmesh(string filename, std::vector<Node>& nodes, std::vector<Triangle>& elements, std::vector<FullTriangle>& fullElements, std::vector<size_t>& boundaryNodes)
{
    ifstream fin(filename.c_str());

    string tmp;
    do { fin >> tmp; } while(tmp != "$Nodes");
    size_t num_nodes; fin >> num_nodes;
    nodes.resize(num_nodes);
    for (size_t i(0); i < num_nodes; ++i)
    {
        size_t ID; fin >> ID;
        double x, y, z; fin >> x >> y >> z;
        nodes[i].ID = ID-1;
        nodes[i].x = x;
        nodes[i].y = y;
    }

    do { fin >> tmp; } while(tmp != "$Elements");
    size_t num_elements; fin >> num_elements;
    elements.clear();
    for (size_t i(0); i < num_elements; ++i)
    {
        size_t ID; fin >> ID;
        size_t type; fin >> type;
        if (type == 2)
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j(0); j < number_of_tags; ++j) fin >> tagtrash;
            size_t A, B, C; fin >> A >> B >> C;
            Triangle new_element;
            new_element.ID = ID-1;
            new_element.nodeA = A-1;
            new_element.nodeB = B-1;
            new_element.nodeC = C-1;
            elements.push_back(new_element);
        }
        else if (type == 1)
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j(0); j < number_of_tags; ++j) fin >> tagtrash;
            size_t a, b;
            fin >> a >> b;
            boundaryNodes.push_back(a);
            boundaryNodes.push_back(b);
        }
        else
            fin.ignore(256, '\n');
    }

    fin.close();
    elements.shrink_to_fit();
    boundaryNodes.shrink_to_fit();
}

#endif
