#ifndef __READFILE_HPP_
#define __READFILE_HPP_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "elements.hpp"

using namespace std;

template<typename datatype>
void file_to_mesh(string filename, std::vector<Vertex<datatype>>& vertices, std::vector<TriangleQ2<datatype>>& elements)
{
    ifstream fin(filename);

    string tmp;
    do { fin >> tmp; } while(tmp != "$Nodes");
    size_t num_vertices; fin >> num_vertices;
    vertices.resize(num_vertices);
    double z_coord_trash;
    for (size_t i(0); i < num_vertices; ++i)
    {
        fin >> vertices[i].id >> vertices[i].x >> vertices[i].y >> z_coord_trash;
        --vertices[i].id;
    }

    do { fin >> tmp; } while(tmp != "$Elements");
    size_t num_elements; fin >> num_elements;
    for (size_t i(0); i < num_elements; ++i)
    {
        size_t id; fin >> id;
        size_t type; fin >> type;
        if (type == 9) // TriangleQ2
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j(0); j < number_of_tags; ++j) fin >> tagtrash;
            size_t id0, id1, id2, id3, id4, id5;
            fin >> id0 >> id1 >> id2 >> id3 >> id4 >> id5;
            elements.push_back( TriangleQ2<datatype>(vertices[id0-1], vertices[id1-1], vertices[id2-1], id3-1, id4-1, id5-1) );
        }
        else if (type == 1)
        {
            //size_t number_of_tags; fin >> number_of_tags;
            //size_t tagtrash;
            //for (size_t j(0); j < number_of_tags; ++j) fin >> tagtrash;
            //size_t a, b;
            //fin >> a >> b;
            //boundaryNodes.push_back(a-1);
            //boundaryNodes.push_back(b-1);
            ////TODO most(or all) boundary vertices are doubled
        }
        else
        {
            fin.ignore(256, '\n');
        }
    }

    fin.close();
    elements.shrink_to_fit();
}

#endif
