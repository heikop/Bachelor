#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstddef>

#include "include/global.hpp"
#include "include/assembleq2.hpp"

int msh_to_nag_q1(std::ifstream& fin, std::ofstream& fout);
int msh_to_nag_q2(std::ifstream& fin, std::ofstream& fout);

// ***** // ***** // ***** // ***** // ***** main ***** // ***** // ***** // ***** // ***** //
int main(int argc, char* argv[])
{
    // some informing output
    if (argc < 1)
    {
        std::cout << "something went wrong." << std::endl;
        return -1;
    }
    if (argc == 1)
    {
        std::cout << "no input file was given." << std::endl;
        return 10;
    }
    if (argc == 2)
    {
        std::cout << "no output format was given." << std::endl;
        return 20;
    }
    if (argc == 3)
    {
        std::cout << "no polynome degree was given." << std::endl;
        return 30;
    }
    if (argc > 4)
        std::cout << "too many inputs given. only the first two will be used." << std::endl;

    std::string input_file{argv[1]};
    std::string input_format{input_file.substr(input_file.find_last_of(".") + 1)};
    std::string output_format{argv[2]};
    std::string output_file{input_file.substr(0, input_file.find_last_of(".") + 1) + output_format};
    if (input_format == output_format)
    {
        std::cout << "output format is the same as the input format. got nothing to do." << std::endl;
        return 0;
    }

    std::ifstream fin(input_file);
    if (! fin.is_open())
    {
        std::cout << "input file '" << input_file << "' not found." << std::endl;
        return 11;
    }
    std::ofstream fout(output_file);
    if (! fout.is_open())
    {
        std::cout << "output file '" << output_file << "' could not be opened." << std::endl;
        return 21;
    }

    std::cout << " input file: " <<  input_file << std::endl;
    std::cout << "output file: " << output_file << std::endl;

    std::string polynom_type{argv[3]};
    if (input_format == "msh" && output_format == "nag")
    {
        if (polynom_type == "q1" || polynom_type == "Q1")
            return msh_to_nag_q1(fin, fout);
        else if (polynom_type == "q2" || polynom_type == "Q2")
            return msh_to_nag_q2(fin, fout);
    }

    std::cout << "no suitable converting function from '" << input_format
              << "' to '" << output_format << "' with polynoms of " << polynom_type << " found." << std::endl;
    return -10;
}

// ***** // ***** // ***** // ***** // ***** msh_to_nag_q2 ***** // ***** // ***** // ***** // ***** //
int msh_to_nag_q2(std::ifstream& fin, std::ofstream& fout)
{
    std::cout << "start converting from 'msh' to 'nag'." << std::endl;

    const size_t max_neighbours{30};
    const size_t max_gaps{1};
    // read input msh file
    std::string tmp;
    do { fin >> tmp; } while(tmp != "$Nodes");
    size_t num_nodes; fin >> num_nodes;
    std::vector<Node> nodes(num_nodes);
    std::vector<size_t> tuples(5 * num_nodes * max_neighbours, 0);
    std::vector<size_t> num_neighbours(num_nodes, 0);
    std::vector<size_t> gaps(num_nodes * max_gaps, 0);
    std::vector<size_t> num_midpoints(num_nodes, 0);
    std::vector<size_t> boundary_midnodes;

    std::cout << " - read input file" << std::flush;
    // read and save nodes
    double z_coord_trash;
    for (size_t i{0}; i < num_nodes; ++i)
    {
        fin >> nodes[i].ID >> nodes[i].x >> nodes[i].y >> z_coord_trash;
        --nodes[i].ID;
    }

    size_t num_edgenodes = 0;
    size_t num_boundary_midnodes = 0;

    // read elements and save the tuples
    do { fin >> tmp; } while(tmp != "$Elements");
    size_t num_elements; fin >> num_elements;
    for (size_t i{0}; i < num_elements; ++i)
    {
        size_t ID; fin >> ID;
        size_t type; fin >> type;
        if (type == 8)
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j{0}; j < number_of_tags; ++j) fin >> tagtrash;
            size_t nodes[3];
            fin >> nodes[0] >> nodes[1] >> nodes[2];
            boundary_midnodes.push_back(nodes[2] - 1);
            ++num_boundary_midnodes;
        }
        if (type == 9)
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j{0}; j < number_of_tags; ++j) fin >> tagtrash;
            size_t nodes[6];
            fin >> nodes[0] >> nodes[1] >> nodes[2] >> nodes[3] >> nodes[4] >> nodes[5];
                 --nodes[0]; --nodes[1]; --nodes[2]; --nodes[3]; --nodes[4]; --nodes[5];

            tuples[5 * (nodes[0] * max_neighbours + num_neighbours[nodes[0]])    ] = nodes[3];
            tuples[5 * (nodes[0] * max_neighbours + num_neighbours[nodes[0]]) + 1] = nodes[1];
            tuples[5 * (nodes[0] * max_neighbours + num_neighbours[nodes[0]]) + 2] = nodes[4];
            tuples[5 * (nodes[0] * max_neighbours + num_neighbours[nodes[0]]) + 3] = nodes[2];
            tuples[5 * (nodes[0] * max_neighbours + num_neighbours[nodes[0]]) + 4] = nodes[5];

            tuples[5 * (nodes[1] * max_neighbours + num_neighbours[nodes[1]])    ] = nodes[4];
            tuples[5 * (nodes[1] * max_neighbours + num_neighbours[nodes[1]]) + 1] = nodes[2];
            tuples[5 * (nodes[1] * max_neighbours + num_neighbours[nodes[1]]) + 2] = nodes[5];
            tuples[5 * (nodes[1] * max_neighbours + num_neighbours[nodes[1]]) + 3] = nodes[0];
            tuples[5 * (nodes[1] * max_neighbours + num_neighbours[nodes[1]]) + 4] = nodes[3];

            tuples[5 * (nodes[2] * max_neighbours + num_neighbours[nodes[2]])    ] = nodes[5];
            tuples[5 * (nodes[2] * max_neighbours + num_neighbours[nodes[2]]) + 1] = nodes[0];
            tuples[5 * (nodes[2] * max_neighbours + num_neighbours[nodes[2]]) + 2] = nodes[3];
            tuples[5 * (nodes[2] * max_neighbours + num_neighbours[nodes[2]]) + 3] = nodes[1];
            tuples[5 * (nodes[2] * max_neighbours + num_neighbours[nodes[2]]) + 4] = nodes[4];

            ++num_neighbours[nodes[0]];
            ++num_neighbours[nodes[1]];
            ++num_neighbours[nodes[2]];

            if (nodes[0] > num_edgenodes)
                num_edgenodes = nodes[0];
            if (nodes[1] > num_edgenodes)
                num_edgenodes = nodes[1];
            if (nodes[2] > num_edgenodes)
                num_edgenodes = nodes[2];
        }
        else
            fin.ignore(256, '\n');
    }
    num_edgenodes -= num_boundary_midnodes - 1; //TODO why must there be a -1 ?
    for (size_t i{0}; i < boundary_midnodes.size(); ++i)
    {
        num_neighbours.erase(num_neighbours.begin() + boundary_midnodes[i] - i);
        num_midpoints.erase(num_midpoints.begin() + boundary_midnodes[i] - i);
        gaps.erase(gaps.begin() + (boundary_midnodes[i] - i)*max_gaps, gaps.begin() + (boundary_midnodes[i] - i + 1)*max_gaps);
        tuples.erase(tuples.begin() + (boundary_midnodes[i] - i)*5*max_neighbours, tuples.begin() + (boundary_midnodes[i] - i + 1)*5*max_neighbours);
        // TODO not so good...
        for (size_t j{boundary_midnodes[i]-i}; j < nodes.size(); ++j)
            --nodes[j].ID;
        nodes.erase(nodes.begin() + boundary_midnodes[i] - i);
    }
    tuples.resize(5 * num_edgenodes * max_neighbours);
    num_neighbours.resize(num_edgenodes);
    gaps.resize(num_edgenodes * max_gaps);
    num_midpoints.resize(num_edgenodes);
    std::cout << ". done" << std::endl;

    // ----- change numbering begin ----- //
    //for (auto& nid : tuples)
    //for (size_t i{0}; i < num_edgenodes; ++i)
    for (auto& t : tuples)
    {
        size_t offset{0};
        bool is_bnid{false};
        for (auto bnid : boundary_midnodes)
        {
            if (bnid < t)
                ++offset;
            if (bnid == t)
                is_bnid = true;
        }
        if (t < num_edgenodes + num_boundary_midnodes)
        {
            if (!is_bnid)
                t -= offset;
            else
                t = num_edgenodes + offset;
        }
    }
    // ------ change numbering end ------ //

    std::cout << " - create adjacents graph" << std::flush;
    // count gaps
    std::vector<size_t> num_gaps(num_edgenodes, 0);
    for (size_t i{0}; i < num_edgenodes; ++i)
    {
        for (size_t a{1}; a < 5*num_neighbours[i]; a+=5)
        {
            bool gap{true};
            for (size_t b{3}; gap && b < 5*num_neighbours[i]; b+=5)
                if (tuples[5*max_neighbours*i + a] == tuples[5*max_neighbours*i + b])
                    gap = false;
            if (gap)
            {
                ++num_gaps[i];
                //--num_neighbours[i];
            }
        }
    }

    // first sorted version of nag but with wrong start
    // TODO TOCHECK what happens when num_gaps > 1 ?
    std::vector<size_t> nag(num_edgenodes * 3 * max_neighbours);
    for (size_t i{0}; i < num_edgenodes; ++i)
    {
//std::cout << "i = " << i << " (" << ", num gaps: " << num_gaps[i] << std::endl;
        if (num_gaps[i] == 0)
        {
//            std::cout << tuples[5*i*max_neighbours +  0] << " " << tuples[5*i*max_neighbours +  1] << " " << tuples[5*i*max_neighbours +  2] << " " << tuples[5*i*max_neighbours +  3] << " " << tuples[5*i*max_neighbours +  4] << std::endl;
//            std::cout << tuples[5*i*max_neighbours +  5] << " " << tuples[5*i*max_neighbours +  6] << " " << tuples[5*i*max_neighbours +  7] << " " << tuples[5*i*max_neighbours +  8] << " " << tuples[5*i*max_neighbours +  9] << std::endl;
//            std::cout << tuples[5*i*max_neighbours + 10] << " " << tuples[5*i*max_neighbours + 11] << " " << tuples[5*i*max_neighbours + 12] << " " << tuples[5*i*max_neighbours + 13] << " " << tuples[5*i*max_neighbours + 14] << std::endl;
//            std::cout << tuples[5*i*max_neighbours + 15] << " " << tuples[5*i*max_neighbours + 16] << " " << tuples[5*i*max_neighbours + 17] << " " << tuples[5*i*max_neighbours + 18] << " " << tuples[5*i*max_neighbours + 19] << std::endl;
            nag[i*3*max_neighbours    ] = tuples[5*i*max_neighbours + 1]; // edge
            nag[i*3*max_neighbours + 1] = tuples[5*i*max_neighbours + 3]; // edge
            nag[i*3*max_neighbours + num_neighbours[i]    ] = tuples[5*i*max_neighbours    ]; // mid
            nag[i*3*max_neighbours + num_neighbours[i] + 1] = tuples[5*i*max_neighbours + 2]; // mid
            size_t current{1};
            for (size_t k{0}; k < num_neighbours[i]-1; ++k)
            {
                for (size_t a{1}; a < 5*num_neighbours[i]; a+=5)
                    if (nag[i*max_neighbours + current] == tuples[5*i*max_neighbours + a])
                    {
                        ++current;
                        nag[i*3*max_neighbours + current] = tuples[5*i*max_neighbours + a + 2]; // edge
                        nag[i*3*max_neighbours + num_neighbours[i] + 2*current    ] = tuples[5*i*max_neighbours + a - 1]; // mid
                        nag[i*3*max_neighbours + num_neighbours[i] + 2*current + 1] = tuples[5*i*max_neighbours + a + 1]; // mid
                    }
            }
        }
        else if (num_gaps[i] == 1)
        {
            size_t start{0};
            bool unique{false};
            while (! unique)
            {
                unique = true;
                for (size_t l{start+1}; l < num_neighbours[i]; ++l)
                    if (tuples[5*(i*max_neighbours + l) + 3] == tuples[5*(i*max_neighbours+start) + 1])
                        unique = false;
                ++start;
            }
            --start;
            nag[i*3*max_neighbours    ] = tuples[5*(i*max_neighbours + start) + 1];
            nag[i*3*max_neighbours + 1] = tuples[5*(i*max_neighbours + start) + 3];
            nag[i*3*max_neighbours + num_neighbours[i]     + 1] = tuples[5*(i*max_neighbours + start)    ]; // mid
            nag[i*3*max_neighbours + num_neighbours[i] + 1 + 1] = tuples[5*(i*max_neighbours + start) + 2]; // mid
            size_t current{1};
            for (size_t k{0}; k < num_neighbours[i]-1; ++k)
            {
                for (size_t a{1}; a < 5*num_neighbours[i]; a+=5)
                    if (nag[i*3*max_neighbours + current] == tuples[5*i*max_neighbours + a])
                    {
                        nag[i*3*max_neighbours + current + 1] = tuples[5*i*max_neighbours + a + 2]; // edge
                        nag[i*3*max_neighbours + num_neighbours[i] + 2*current     + 1] = tuples[5*i*max_neighbours + a - 1]; // mid
                        nag[i*3*max_neighbours + num_neighbours[i] + 2*current + 1 + 1] = tuples[5*i*max_neighbours + a + 1]; // mid
                        if (k == num_neighbours[i] - 2 || k == gaps[i])
                            nag[i*3*max_neighbours + num_neighbours[i] + 2*current + 2 + 1] = tuples[5*i*max_neighbours + a + 3]; // mid
                        ++current;
                    }
            }
            ++num_neighbours[i];
            // gaps[i] is already 0
        }
        else
        {
            std::cout << "the special case of more then one disconnection is not implemented (yet)." << std::endl;
            return -2;
        }
    }
    std::cout << ". done" << std::endl;
/*
    std::cout << " - reorder graph entries" << std::flush;
    // sort to begin at x,0 or the nearest counterclockwise
    for (size_t i{0}; i < num_edgenodes; ++i)
    {
        // first step: check where to start
        bool top{false};
        bool right{false};
        bool left{false};
        size_t first{0};
        for (size_t neighbour{0}; neighbour < num_neighbours[i] && ! top; ++neighbour)
        {
            if (nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y)
            {
                top = true;
                first = neighbour;
            }
        }
        if (top)
        {
            for (size_t neighbour{first}; neighbour < num_neighbours[i] && ! right; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x >= nodes[i].x && nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y)
                {
                    right = true;
                    first = neighbour;
                }
        }
        else
        {
            for (size_t neighbour{first}; neighbour < num_neighbours[i] && ! left; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x <= nodes[i].x && nodes[nag[i*max_neighbours + neighbour]].y < nodes[i].y)
                {
                    left = true;
                    first = neighbour;
                }
        }

        if (top && right)
        {
            for (size_t neighbour{first+1}; neighbour < num_neighbours[i]; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x >= nodes[i].x && 
                    nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y && 
                    nodes[nag[i*max_neighbours + neighbour]].y < nodes[nag[i*max_neighbours + first]].y)
                    first = neighbour;
        }
        else if (top && left)
        {
            for (size_t neighbour{first+1}; neighbour < num_neighbours[i]; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x < nodes[i].x && 
                    nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y && 
                    nodes[nag[i*max_neighbours + neighbour]].x > nodes[nag[i*max_neighbours + first]].x)
                    first = neighbour;
        }
        else if (left) // && bottom
        {
            for (size_t neighbour{first+1}; neighbour < num_neighbours[i]; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x < nodes[i].x && 
                    nodes[nag[i*max_neighbours + neighbour]].y < nodes[i].y && 
                    nodes[nag[i*max_neighbours + neighbour]].x > nodes[nag[i*max_neighbours + first]].x)
                    first = neighbour;
        }
        else // right && bottom
        {
            for (size_t neighbour{first+1}; neighbour < num_neighbours[i]; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x < nodes[i].x && 
                    nodes[nag[i*max_neighbours + neighbour]].y < nodes[i].y && 
                    nodes[nag[i*max_neighbours + neighbour]].y > nodes[nag[i*max_neighbours + first]].y)
                    first = neighbour;
        }

//TODO BEGIN
//        // second step: reorder
//        size_t temporary_nag[max_neighbours];
//        for (size_t neighbour{0}; neighbour < first; ++neighbour)
//            temporary_nag[neighbour] = nag[i*max_neighbours + neighbour];
//        for (size_t neighbour{0}; neighbour < num_neighbours[i] - first; ++neighbour)
//            nag[i*max_neighbours + neighbour] = nag[i*max_neighbours + first + neighbour];
//        for (size_t neighbour{0}; neighbour < first; ++neighbour)
//            nag[i*max_neighbours + num_neighbours[i] - first + neighbour] = temporary_nag[neighbour];
//TODO END
//NEW
        // second step: reorder
        // TODO check outer midpoints
        size_t temporary_nag[3*max_neighbours];
        for (size_t neighbour{0}; neighbour < first; ++neighbour)
            temporary_nag[neighbour] = nag[i*3*max_neighbours + neighbour];
        for (size_t neighbour{0}; neighbour < num_neighbours[i] - first; ++neighbour)
            nag[i*max_neighbours + neighbour] = nag[i*3*max_neighbours + first + neighbour];
        for (size_t neighbour{0}; neighbour < first; ++neighbour)
            nag[i*max_neighbours + num_neighbours[i] - first + neighbour] = temporary_nag[neighbour];
//NEW END

        // third step: update gap(s)
        for (size_t gap{0}; gap < num_gaps[i]; ++gap)
            gaps[i*max_gaps + gap] = (gaps[i*max_gaps + gap] + num_neighbours[i] - first) % num_neighbours[i];
    }
    std::cout << ". done" << std::endl;
*/

    // count 'num midpoints to calc' (num nodes in the first two quarters: y >= 0 && x > 0 or y > 0 && x <= 0, with current node = 0,0)
    for (size_t i{0}; i < num_edgenodes; ++i)
    {
        for (size_t neighbour{0}; neighbour < num_neighbours[i]; ++neighbour)
        {
            if ((nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y && 
                 nodes[nag[i*max_neighbours + neighbour]].x >  nodes[i].x    )
             || (nodes[nag[i*max_neighbours + neighbour]].y >  nodes[i].y && 
                 nodes[nag[i*max_neighbours + neighbour]].x <= nodes[i].x    ))
                ++num_midpoints[i];
        }
    }

    std::cout << " - write output" << std::flush;
    // write into new file
    fout << "# nag-format (node adjacent graph)" << std::endl << std::endl;

    fout << "# nodecount" << std::endl;
    fout << num_nodes <<  std::endl;
    fout << "# edgenodecount" << std::endl;
    fout << num_edgenodes <<  std::endl;
    fout << "with IDs" << std::endl << std::endl;

    //fout << "# nodes : ID - xcoord - ycoord" << std::endl;
    fout << "# edgenodes" << std::endl;
    for (size_t i{0}; i < num_edgenodes; ++i)
        fout << nodes[i].ID << " " << nodes[i].x << " " << nodes[i].y << std::endl;
    fout << std::endl;

    //fout << "# adjacent-graph : node ID - num neighbour nodes - num midpoints to calc (num nodes in first two quarters) - num gaps - [gaps] - neighbour IDs" << std::endl;
    fout << "# adjacent-graph" << std::endl;
    for (size_t i{0}; i < num_edgenodes; ++i)
    {
        fout << nodes[i].ID << " " << num_neighbours[i] << " " << num_midpoints[i] << " " << num_gaps[i] << " ";
        for (size_t gap{0}; gap < num_gaps[i]; ++gap)
            fout << gaps[i*max_gaps + gap] << " ";
        //for (size_t neighbours{0}; neighbours < num_neighbours[i]; ++neighbours)
        //    fout << nag[i*3*max_neighbours + neighbours] << " ";
        //for (size_t midneighbours{num_neighbours[i]}; midneighbours < 3*num_neighbours[i] - num_gaps[i]; ++midneighbours)
        //    fout << nag[i*3*max_neighbours + midneighbours] << " ";
        for (size_t neighbours{0}; neighbours < 3*num_neighbours[i] - num_gaps[i]; ++neighbours)
            fout << nag[i*3*max_neighbours + neighbours] << " ";
        fout << std::endl;
    }
    std::cout << ". done" << std::endl;

    fin.close();
    fout.close();
    std::cout << "done converting" << std::endl;
    return 0;
}

// ***** // ***** // ***** // ***** // ***** msh_to_nag_q1 ***** // ***** // ***** // ***** // ***** //
int msh_to_nag_q1(std::ifstream& fin, std::ofstream& fout)
{
    std::cout << "start converting from 'msh' to 'nag'." << std::endl;

    const size_t max_neighbours{20};
    const size_t max_gaps{1};
    // read input msh file
    std::string tmp;
    do { fin >> tmp; } while(tmp != "$Nodes");
    size_t num_nodes; fin >> num_nodes;
    std::vector<Node> nodes(num_nodes);
    std::vector<size_t> pairs(2 * num_nodes * max_neighbours);
    std::vector<size_t> num_neighbours(num_nodes, 0);
    std::vector<size_t> gaps(num_nodes * max_gaps, 0);
    std::vector<size_t> num_midpoints(num_nodes, 0);

    std::cout << " - read input file" << std::flush;
    // read and save nodes
    double z_coord_trash;
    for (size_t i{0}; i < num_nodes; ++i)
    {
        fin >> nodes[i].ID >> nodes[i].x >> nodes[i].y >> z_coord_trash;
        --nodes[i].ID;
    }

    // read elements and save the pairs
    do { fin >> tmp; } while(tmp != "$Elements");
    size_t num_elements; fin >> num_elements;
    for (size_t i{0}; i < num_elements; ++i)
    {
        size_t ID; fin >> ID;
        size_t type; fin >> type;
        if (type == 2)
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j(0); j < number_of_tags; ++j) fin >> tagtrash;
            size_t nodes[3];
            fin >> nodes[0] >> nodes[1] >> nodes[2];
            --nodes[0]; --nodes[1]; --nodes[2];
            pairs[2 * (nodes[0] * max_neighbours + num_neighbours[nodes[0]])    ] = nodes[1];
            pairs[2 * (nodes[0] * max_neighbours + num_neighbours[nodes[0]]) + 1] = nodes[2];
            pairs[2 * (nodes[1] * max_neighbours + num_neighbours[nodes[1]])    ] = nodes[2];
            pairs[2 * (nodes[1] * max_neighbours + num_neighbours[nodes[1]]) + 1] = nodes[0];
            pairs[2 * (nodes[2] * max_neighbours + num_neighbours[nodes[2]])    ] = nodes[0];
            pairs[2 * (nodes[2] * max_neighbours + num_neighbours[nodes[2]]) + 1] = nodes[1];
            ++num_neighbours[nodes[0]];
            ++num_neighbours[nodes[1]];
            ++num_neighbours[nodes[2]];
        }
        else
            fin.ignore(256, '\n');
    }
    std::cout << ". done" << std::endl;

    std::cout << " - create adjacents graph" << std::flush;
    // count gaps
    std::vector<size_t> num_gaps(num_nodes, 0);
    for (size_t i{0}; i < num_nodes; ++i)
    {
        for (size_t a{0}; a < 2*num_neighbours[i]; a+=2)
        {
            bool gap{true};
            for (size_t b{1}; gap && b < 2*num_neighbours[i]; b+=2)
                if (pairs[2*max_neighbours*i + a] == pairs[2*max_neighbours*i + b])
                    gap = false;
            if (gap)
                ++num_gaps[i];
        }
    }

    // first sorted version of nag but with wrong start
    // TODO TOCHECK what happens when num_gaps > 1 ?
    std::vector<size_t> nag(num_nodes*max_neighbours);
    for (size_t i{0}; i < num_nodes; ++i)
    {
        if (num_gaps[i] == 0)
        {
        //std::cout << "no gap "<< i << ": " << num_neighbours[i] << std::endl;
            nag[i*max_neighbours    ] = pairs[2*i*max_neighbours];
            nag[i*max_neighbours + 1] = pairs[2*i*max_neighbours + 1];
            size_t current{1};
            for (size_t k{0}; k < num_neighbours[i]-1; ++k)
            {
                for (size_t a{2}; a < 2*num_neighbours[i]; a+=2)
                    if (nag[i*max_neighbours + current] == pairs[2*i*max_neighbours + a])
                    {
                        ++current;
                        nag[i*max_neighbours + current] = pairs[2*i*max_neighbours + a + 1];
                    }
            }
        }
        else if (num_gaps[i] == 1)
        {
        //std::cout << "gap "<< i << std::endl;
          size_t start{0};
            bool unique{false};
            while (! unique)
            {
                unique = true;
                for (size_t l{start+1}; l < num_neighbours[i]; ++l)
                    if (pairs[2*(i*max_neighbours + l) + 1] == pairs[2*(i*max_neighbours+start)])
                        unique = false;
                ++start;
            }
            --start;
            nag[i*max_neighbours    ] = pairs[2*(i*max_neighbours + start)];
            nag[i*max_neighbours + 1] = pairs[2*(i*max_neighbours + start) + 1];
            size_t current{1};
            for (size_t k{0}; k < num_neighbours[i]-1; ++k)
            {
                for (size_t a{0}; a < 2*num_neighbours[i]; a+=2)
                    if (nag[i*max_neighbours + current] == pairs[2*i*max_neighbours + a])
                    {
                        ++current;
                        nag[i*max_neighbours + current] = pairs[2*i*max_neighbours + a + 1];
                    }
            }
            ++num_neighbours[i];
            // gaps[i] is already 0
        }
        else
        {
            std::cout << "the special case of more then one disconnection is not implemented (yet)." << std::endl;
            return -2;
        }
    }
    std::cout << ". done" << std::endl;

    std::cout << " - reorder graph entries" << std::flush;
    // sort to begin at x,0 or the nearest counterclockwise
    for (size_t i{0}; i < num_nodes; ++i)
    {
        // first step: check where to start
        bool top{false};
        bool right{false};
        bool left{false};
        size_t first{0};
        for (size_t neighbour{0}; neighbour < num_neighbours[i] && ! top; ++neighbour)
        {
            if (nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y)
            {
                top = true;
                first = neighbour;
            }
        }
        if (top)
        {
            for (size_t neighbour{first}; neighbour < num_neighbours[i] && ! right; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x >= nodes[i].x && nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y)
                {
                    right = true;
                    first = neighbour;
                }
        }
        else
        {
            for (size_t neighbour{first}; neighbour < num_neighbours[i] && ! left; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x <= nodes[i].x && nodes[nag[i*max_neighbours + neighbour]].y < nodes[i].y)
                {
                    left = true;
                    first = neighbour;
                }
        }

        if (top && right)
        {
            for (size_t neighbour{first+1}; neighbour < num_neighbours[i]; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x >= nodes[i].x && 
                    nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y && 
                    nodes[nag[i*max_neighbours + neighbour]].y < nodes[nag[i*max_neighbours + first]].y)
                    first = neighbour;
        }
        else if (top && left)
        {
            for (size_t neighbour{first+1}; neighbour < num_neighbours[i]; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x < nodes[i].x && 
                    nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y && 
                    nodes[nag[i*max_neighbours + neighbour]].x > nodes[nag[i*max_neighbours + first]].x)
                    first = neighbour;
        }
        else if (left) // && bottom
        {
            for (size_t neighbour{first+1}; neighbour < num_neighbours[i]; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x < nodes[i].x && 
                    nodes[nag[i*max_neighbours + neighbour]].y < nodes[i].y && 
                    nodes[nag[i*max_neighbours + neighbour]].x > nodes[nag[i*max_neighbours + first]].x)
                    first = neighbour;
        }
        else // right && bottom
        {
            for (size_t neighbour{first+1}; neighbour < num_neighbours[i]; ++neighbour)
                if (nodes[nag[i*max_neighbours + neighbour]].x < nodes[i].x && 
                    nodes[nag[i*max_neighbours + neighbour]].y < nodes[i].y && 
                    nodes[nag[i*max_neighbours + neighbour]].y > nodes[nag[i*max_neighbours + first]].y)
                    first = neighbour;
        }

        // second step: reorder
        size_t temporary_nag[max_neighbours];
        for (size_t neighbour{0}; neighbour < first; ++neighbour)
            temporary_nag[neighbour] = nag[i*max_neighbours + neighbour];
        for (size_t neighbour{0}; neighbour < num_neighbours[i] - first; ++neighbour)
            nag[i*max_neighbours + neighbour] = nag[i*max_neighbours + first + neighbour];
        for (size_t neighbour{0}; neighbour < first; ++neighbour)
            nag[i*max_neighbours + num_neighbours[i] - first + neighbour] = temporary_nag[neighbour];

        // third step: update gap(s)
        for (size_t gap{0}; gap < num_gaps[i]; ++gap)
            gaps[i*max_gaps + gap] = (gaps[i*max_gaps + gap] + num_neighbours[i] - first) % num_neighbours[i];
    }
    std::cout << ". done" << std::endl;

    // count 'num midpoints to calc' (num nodes in the first two quarters: y >= 0 && x > 0 or y > 0 && x <= 0, with current node = 0,0)
    for (size_t i{0}; i < num_nodes; ++i)
    {
        for (size_t neighbour{0}; neighbour < num_neighbours[i]; ++neighbour)
        {
            if ((nodes[nag[i*max_neighbours + neighbour]].y >= nodes[i].y && 
                 nodes[nag[i*max_neighbours + neighbour]].x >  nodes[i].x    )
             || (nodes[nag[i*max_neighbours + neighbour]].y >  nodes[i].y && 
                 nodes[nag[i*max_neighbours + neighbour]].x <= nodes[i].x    ))
                ++num_midpoints[i];
        }
    }

    std::cout << " - write output" << std::flush;
    // write into new file
    fout << "# nag-format (node adjacent graph)" << std::endl << std::endl;

    fout << "# nodecount" << std::endl;
    fout << num_nodes <<  std::endl;
    fout << "with IDs" << std::endl << std::endl;

    //fout << "# nodes : ID - xcoord - ycoord" << std::endl;
    fout << "# nodes" << std::endl;
    for (size_t i{0}; i < num_nodes; ++i)
        fout << nodes[i].ID << " " << nodes[i].x << " " << nodes[i].y << std::endl;
    fout << std::endl;

    //fout << "# adjacent-graph : node ID - num neighbour nodes - num midpoints to calc (num nodes in first two quarters) - num gaps - [gaps] - neighbour IDs" << std::endl;
    fout << "# adjacent-graph" << std::endl;
    for (size_t i{0}; i < num_nodes; ++i)
    {
        fout << nodes[i].ID << " " << num_neighbours[i] << " " << num_midpoints[i] << " " << num_gaps[i] << " ";
        for (size_t gap{0}; gap < num_gaps[i]; ++gap)
            fout << gaps[i*max_gaps + gap] << " ";
        for (size_t neighbours{0}; neighbours < num_neighbours[i]; ++neighbours)
            fout << nag[i*max_neighbours + neighbours] << " ";
        fout << std::endl;
    }
    std::cout << ". done" << std::endl;

    fin.close();
    fout.close();
    std::cout << "done converting" << std::endl;
    return 0;
}
