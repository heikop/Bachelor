#ifndef __READNAG_HPP_
#define __READNAG_HPP_

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>


void readnag(std::string filename,
             std::vector<Node>& nodes,
             std::vector<size_t>& num_neighbours,
             std::vector<size_t>& nag,
             std::vector<size_t>& num_midpoints,
             std::vector<size_t>& gaps,
             std::vector<size_t>& num_gaps)
{
    nodes.clear();
    num_neighbours.clear();
    nag.clear();
    num_midpoints.clear();
    gaps.clear();
    num_gaps.clear();

    ifstream fin(filename);

    string tmp;
    do { fin >> tmp; } while(tmp != "nodecount");
    size_t num_nodes; fin >> num_nodes;
    fin >> tmp;
    bool withIDs{tmp == "with"};
    fin >> tmp;
    if (withIDs && tmp == "IDs")
    {
        // everything ok. nothing happens
    }
    else
    {
        withIDs = false;
        std::cout << "without IDs is not implemented (yet)." << std::endl;
        return;
    }

    const size_t max_gaps{1};
    nodes.resize(num_nodes);
    num_neighbours.resize(num_nodes);
    //nag.resize(num_nodes);
    num_midpoints.resize(num_nodes);
    gaps.resize(max_gaps * num_nodes);
    num_gaps.resize(num_nodes);

    do { fin >> tmp; } while(tmp != "nodes");
    for (size_t i{0}; i < num_nodes; ++i)
        fin >> nodes[i].ID >> nodes[i].x >> nodes[i].y;

    do { fin >> tmp; } while(tmp != "adjacent-graph");
    size_t current{0};
    for (size_t i{0}; i < num_nodes; ++i)
    {
        size_t actual_node_id;
        fin >> actual_node_id;
        fin >> num_neighbours[i] >> num_midpoints[i] >> num_gaps[i];
        assert(num_gaps[i] <= max_gaps);
        for (size_t gap{0}; gap < num_gaps[i]; ++gap)
            fin >> gaps[i*max_gaps + gap];
        for (size_t neighbours{0}; neighbours < num_neighbours[i]; ++neighbours)
        {
            fin >> current;
            nag.push_back(current);
//            fin >> nag[current];
//            ++current;
            // in short, pre or post?: fin >> nag[++current++];
        }
    }
    nag.shrink_to_fit();
}

#endif
