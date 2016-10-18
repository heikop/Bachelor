#ifndef __READMESH_HPP_
#define __READMESH_HPP_

#include <iostream>
#include <fstream>
#include <vector>
//#include <sstream>

using namespace std;

void readmesh(string filename, std::vector<Node>& nodes, std::vector<Triangle1>& elements, std::vector<FullTriangle>& fullElements, std::vector<size_t>& boundaryNodes)
{
    ifstream fin(filename.c_str());

    string tmp;
    do { fin >> tmp; } while(tmp != "$Nodes");
    size_t num_nodes; fin >> num_nodes;
    nodes.resize(num_nodes);
    double z_coord_trash;
    for (size_t i(0); i < num_nodes; ++i)
    {
        fin >> nodes[i].ID >> nodes[i].x >> nodes[i].y >> z_coord_trash;
        --nodes[i].ID;
    }

    do { fin >> tmp; } while(tmp != "$Elements");
    size_t num_elements; fin >> num_elements;
    size_t num_triangles{num_elements};
    elements.resize(num_elements);
    size_t current{0};
    for (size_t i(0); i < num_elements; ++i)
    {
        size_t ID; fin >> ID;
        size_t type; fin >> type;
        if (type == 2)
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j(0); j < number_of_tags; ++j) fin >> tagtrash;
            elements[current].ID = ID-1;
            fin >> elements[current].nodeA >> elements[current].nodeB >> elements[current].nodeC;
            elements[current].nodeA -= 1;
            elements[current].nodeB -= 1;
            elements[current].nodeC -= 1;
            ++current;
        }
        else if (type == 1)
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j(0); j < number_of_tags; ++j) fin >> tagtrash;
            size_t a, b;
            fin >> a >> b;
            boundaryNodes.push_back(a-1);
            boundaryNodes.push_back(b-1);
            //TODO most(or all) boundary nodes are doubled
            --num_triangles;
        }
        else
        {
            --num_triangles;
            fin.ignore(256, '\n');
        }
    }

    fin.close();
    //elements.shrink_to_fit();
    elements.resize(num_triangles);
    boundaryNodes.shrink_to_fit();
}

#endif
