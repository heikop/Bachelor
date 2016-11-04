#ifndef __READFILE_HPP_
#define __READFILE_HPP_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "elements.hpp"

using namespace std;

template<typename datatype>
//void file_to_mesh(string filename, std::vector<Vertex<datatype>>& vertices, std::vector<TriangleQ2<datatype>>& elements)
void file_to_mesh(string filename, std::vector<Vertex<datatype>>& vertices, std::vector<Element<datatype>*>& elements)
{
    ifstream fin(filename);

    string tmp;
    do { fin >> tmp; } while(tmp != "$Nodes");
    size_t num_vertices; fin >> num_vertices;
    vertices.resize(num_vertices);
    double z_coord_trash;
    for (size_t i{0}; i < num_vertices; ++i)
    {
        fin >> vertices[i].id >> vertices[i].x >> vertices[i].y >> z_coord_trash;
        --vertices[i].id;
    }

    do { fin >> tmp; } while(tmp != "$Elements");
    size_t num_elements; fin >> num_elements;
    for (size_t i{0}; i < num_elements; ++i)
    {
        size_t id; fin >> id;
        size_t type; fin >> type;
        if (type == 9) // TriangleQ2
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j{0}; j < number_of_tags; ++j) fin >> tagtrash;
            size_t id0, id1, id2, id3, id4, id5;
            fin >> id0 >> id1 >> id2 >> id3 >> id4 >> id5;
            elements.push_back( new TriangleQ2<datatype>(vertices[id0-1], vertices[id1-1], vertices[id2-1], id3-1, id4-1, id5-1) );
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

template<typename datatype>
void file_to_mesh_all(string filename, std::vector<Vertex<datatype>>& vertices, std::vector<Element<datatype>*>& elements)
{
    ifstream fin(filename);

    string tmp;
    do { fin >> tmp; } while(tmp != "$Nodes");
    size_t num_vertices; fin >> num_vertices;
    vertices.resize(num_vertices);
    double z_coord_trash;
    for (size_t i{0}; i < num_vertices; ++i)
    {
        fin >> vertices[i].id >> vertices[i].x >> vertices[i].y >> z_coord_trash;
        --vertices[i].id;
    }

    do { fin >> tmp; } while(tmp != "$Elements");
    size_t num_elements; fin >> num_elements;
    for (size_t i{0}; i < num_elements; ++i)
    {
        size_t id; fin >> id;
        size_t type; fin >> type;
        size_t number_of_tags; fin >> number_of_tags;
        size_t tagtrash;
        for (size_t j{0}; j < number_of_tags; ++j) fin >> tagtrash;
        switch (type)
        {
        //case 15: // 1-node-point
        //{
        //    // TODO ?
        //    fin.ignore(256, '\n');
        //    break;
        //}
        //case 1: // LineQ1
        //{
        //    // TODO
        //    fin.ignore(256, '\n');
        //    break;
        //}
        //case 8: // LineQ2
        //{
        //    // TODO
        //    fin.ignore(256, '\n');
        //    break;
        //}
        case 2: // TriangleQ1
        {
            size_t id0, id1, id2;
            fin >> id0 >> id1 >> id2;
            elements.push_back( new TriangleQ1<datatype>(vertices[id0-1], vertices[id1-1], vertices[id2-1]) );
            break;
        }
        case 9: // TriangleQ2
        {
            size_t id0, id1, id2, id3, id4, id5;
            fin >> id0 >> id1 >> id2 >> id3 >> id4 >> id5;
            elements.push_back( new TriangleQ2<datatype>(vertices[id0-1], vertices[id1-1], vertices[id2-1], id3-1, id4-1, id5-1) );
            break;
        }
        case 3: // QuadrilateralQ1
        {
            size_t id0, id1, id2, id3;
            fin >> id0 >> id1 >> id2 >> id3;
            elements.push_back( new QuadrilateralQ1<datatype>(vertices[id0-1], vertices[id1-1], vertices[id2-1], vertices[id3-1]) );
            break;
        }
        case 10: // QuadrilateralQ2 (9 nodes)
        {
            size_t id0, id1, id2, id3, id4, id5, id6, id7, id8;
            fin >> id0 >> id1 >> id2 >> id3 >> id4 >> id5 >> id6 >> id7 >> id8;
            elements.push_back( new QuadrilateralQ2<datatype>(vertices[id0-1], vertices[id1-1], vertices[id2-1], vertices[id3-1], id4-1, id5-1, id6-1, id7-1, id8-1) );
            break;
        }
        //case 16: // QuadrilateralQ2 (8 nodes)
        //{
        //    // TODO
        //    fin.ignore(256, '\n');
        //    break;
        //}
        default: fin.ignore(256, '\n');
        }
    }

    fin.close();
    elements.shrink_to_fit();
}


template<typename datatype>
void quad_to_tri(std::vector<Vertex<datatype>>& vertices, std::vector<Element<datatype>*>& elements)
{
    size_t num_quads{elements.size()};
    for (size_t e{0}; e < num_quads; ++e)
    {
        if (typeid(*(elements[e])) == typeid(QuadrilateralQ1<datatype>))
        {
            const std::vector<size_t> vertexids{elements[e]->vertexids()};
            delete elements[e];
            elements[e] = new TriangleQ1<datatype>(vertices[vertexids[0]], vertices[vertexids[1]], vertices[vertexids[2]]);
            elements.push_back( new TriangleQ1<datatype>(vertices[vertexids[2]], vertices[vertexids[3]], vertices[vertexids[0]]) );
        }
        else if (typeid(*(elements[e])) == typeid(QuadrilateralQ2<datatype>))
        {
            //TODO
        }
    }
}

#endif
