#include "include/assemble.hpp"

void assemble_id(CsrMatrixCpu& matrix, pstd::vector<Node>& nodes, std::vector<TriangleQ2>& elements)
{
    for (const auto& elem : elements)
    {
        // B = [a b]
        //     [c d]
        const float a{elem.nodeB.x - elem.nodeA.x};
        const float c{elem.nodeB.y - elem.nodeA.y};
        const float b{elem.nodeC.x - elem.nodeA.x};
        const float d{elem.nodeC.y - elem.nodeA.y};
        const float detB(std::abs(a*d - b*c));

        const float bbdd{b*b + d*d};
        const float abcd{a*b + c*d};
        const float aacc{a*a + c*c};

        matrix.add_global(elem.nodeA.ID, elem.nodeA.ID, (  3.0*bbdd -   6.0*abcd +   3.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeA.ID, elem.nodeB.ID, (      bbdd -       abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeA.ID, elem.nodeC.ID, (           -       abcd +       aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeA.ID, elem.nodeD.ID, (  4.0*bbdd -   4.0*abcd             ) / (6.0 * detB);
        //trix.add_global(elem.nodeA.ID, elem.nodeE.ID, (                                    ) / (6.0 * detB);
        matrix.add_global(elem.nodeA.ID, elem.nodeF.ID, (           -   4.0*abcd +   4.0*aacc) / (6.0 * detB);

        matrix.add_global(elem.nodeB.ID, elem.nodeA.ID, (  3.0*bbdd -   6.0*abcd +   3.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeB.ID, elem.nodeB.ID, (  3.0*bbdd                          ) / (6.0 * detB);
        matrix.add_global(elem.nodeB.ID, elem.nodeC.ID, (                   abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeB.ID, elem.nodeD.ID, (-12.0*bbdd +   4.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeB.ID, elem.nodeE.ID, (           -   4.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeB.ID, elem.nodeF.ID, (               8.0*abcd             ) / (6.0 * detB);

        matrix.add_global(elem.nodeC.ID, elem.nodeA.ID, (           -       abcd +       aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeC.ID, elem.nodeB.ID, (                   abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeC.ID, elem.nodeC.ID, (                            3.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeC.ID, elem.nodeD.ID, (               8.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeC.ID, elem.nodeE.ID, (           -   4.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeC.ID, elem.nodeF.ID, (               4.0*abcd -  12.0*aacc) / (6.0 * detB);

        matrix.add_global(elem.nodeD.ID, elem.nodeA.ID, (  4.0*bbdd -   4.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeD.ID, elem.nodeB.ID, (-12.0*bbdd +   4.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeD.ID, elem.nodeC.ID, (               8.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeD.ID, elem.nodeD.ID, (200.0*bbdd -  72.0*abcd +   8.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeD.ID, elem.nodeE.ID, (-32.0*bbdd +  40.0*abcd -   8.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeD.ID, elem.nodeF.ID, ( 32.0*bbdd - 200.0*abcd +  32.0*aacc) / (6.0 * detB);

        //trix.add_global(elem.nodeE.ID, elem.nodeA.ID, (                                    ) / (6.0 * detB);
        matrix.add_global(elem.nodeE.ID, elem.nodeB.ID, (           -   4.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeE.ID, elem.nodeC.ID, (           -   4.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeE.ID, elem.nodeD.ID, (-32.0*bbdd +  40.0*abcd -   8.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeE.ID, elem.nodeE.ID, (  8.0*bbdd -   8.0*abcd +   8.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeE.ID, elem.nodeF.ID, ( -8.0*bbdd +  40.0*abcd -  32.0*aacc) / (6.0 * detB);

        matrix.add_global(elem.nodeF.ID, elem.nodeA.ID, (           -   4.0*abcd +   4.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeF.ID, elem.nodeB.ID, (               8.0*abcd             ) / (6.0 * detB);
        matrix.add_global(elem.nodeF.ID, elem.nodeC.ID, (               4.0*abcd -  12.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeF.ID, elem.nodeD.ID, ( 32.0*bbdd - 200.0*abcd +  32.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeF.ID, elem.nodeE.ID, ( -8.0*bbdd +  40.0*abcd -  32.0*aacc) / (6.0 * detB);
        matrix.add_global(elem.nodeF.ID, elem.nodeF.ID, (  8.0*bbdd -  72.0*abcd + 200.0*aacc) / (6.0 * detB);

    }
}


void mesh_q2(string filename, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements, size_t& highest_edgenode)
{
    ifstream fin(filename);

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

    highest_edgenode = 0;

    do { fin >> tmp; } while(tmp != "$Elements");
    size_t num_elements; fin >> num_elements;
    elements.resize(num_elements);
    size_t current{0};
    for (size_t i{0}; i < num_elements; ++i)
    {
        size_t ID; fin >> ID;
        size_t type; fin >> type;
        if (type == 9)
        {
            size_t number_of_tags; fin >> number_of_tags;
            size_t tagtrash;
            for (size_t j{0}; j < number_of_tags; ++j) fin >> tagtrash;
            elements[current].ID = ID-1;
            fin >> elements[current].nodeA >> elements[current].nodeB >> elements[current].nodeC
                >> elements[current].nodeD >> elements[current].nodeE >> elements[current].nodeF;
            elements[current].nodeA -= 1;
            elements[current].nodeB -= 1;
            elements[current].nodeC -= 1;
            elements[current].nodeD -= 1;
            elements[current].nodeE -= 1;
            elements[current].nodeF -= 1;
            if (elements[current].nodeA > highest_edgenode)
                highest_edgenode = elements[current].nodeA;
            if (elements[current].nodeB > highest_edgenode)
                highest_egdenode = elements[current].nodeB;
            if (elements[current].nodeC > highest_edgenode)
                highest_edgenode = elements[current].nodeC;
            ++current;
        }
        else
        {
            --num_triangles;
            fin.ignore(256, '\n');
        }
    }

    fin.close();
    elements.resize(current);
}
