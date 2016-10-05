#include "include/assemble.hpp"

void structure_id(CsrMatrixCpu& matrix, std::vector<TriangleQ2>& elements)
{
    const size_t max_rowlength{30};

    size_t* num_nonzeros = new size_t[matrix._numrows_local];
    for (size_t i{0}; i < matrix._numrows_local; ++i)
        num_nonzeros[i] = 0;

    size_t* colind = new size_t[max_rowlength*matrix._numrows_local];

    for (size_t i{0}; i < elements.size(); ++i)
    {
        size_t nodes[6];
        nodes[0] = elements[i].nodeA;
        nodes[1] = elements[i].nodeB;
        nodes[2] = elements[i].nodeC;
        nodes[3] = elements[i].nodeD;
        nodes[4] = elements[i].nodeE;
        nodes[5] = elements[i].nodeF;
        for (size_t node1{0}; node1 < 6; ++node1)
        {
            for (size_t node2{0}; node2 < 6; ++node2)
            {
                int a{nodes[node1] - matrix._firstrow_on_local};
                size_t b(nodes[node2]);
                if (a >= 0 && static_cast<size_t>(a) < matrix._numrows_local)
                {
                    size_t j{0};
                    while (j < num_nonzeros[a] && colind[a*max_rowlength + j] != b )
                        ++j;
                    if (num_nonzeros[a] == j)
                    {
                        ++(num_nonzeros[a]);
                        assert(num_nonzeros[a] <= max_rowlength);
                        colind[a*max_rowlength + j] = b;
                    }
                }
            }
        }
    }

    for (size_t i{0}; i < matrix._numrows_local; ++i)
        for (size_t a{num_nonzeros[i]-1}; a > 0; --a)
            for (size_t b{0}; b < a; ++b)
                if (colind[i*max_rowlength + b] > colind[i*max_rowlength + b+1])
                {
                    size_t tmp{colind[i*max_rowlength + b]};
                    colind[i*max_rowlength + b] = colind[i*max_rowlength + b+1];
                    colind[i*max_rowlength + b+1] = tmp;
                }

    size_t num_values{0};
    for (size_t i{0}; i < matrix._numrows_local; ++i)
    {
        matrix._rowptr[i] = num_values;
        num_values += num_nonzeros[i];
    }
    matrix._rowptr[matrix._numrows_local] = num_values;
    delete[] matrix._colind;
    delete[] matrix._values;
    matrix._colind = new size_t[num_values];
    matrix._values = new float[num_values];

    size_t current_pos{0};
    for (size_t row{0}; row < matrix._numrows_local; ++row)
        for (size_t col{0}; col < num_nonzeros[row]; ++col)
            matrix._colind[current_pos++] = colind[row*max_rowlength + col];
    for (size_t i{0}; i < num_values; ++i)
        matrix._values[i] = 0.0;

    delete[] num_nonzeros;
    delete[] colind;
}

void assemble_id(CsrMatrixCpu& matrix, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements)
{
    for (const auto& elem : elements)
    {
        // B = [a b]
        //     [c d]
        //const float a{elem.nodeB.x - elem.nodeA.x};
        //const float c{elem.nodeB.y - elem.nodeA.y};
        //const float b{elem.nodeC.x - elem.nodeA.x};
        //const float d{elem.nodeC.y - elem.nodeA.y};
        const float a{nodes[elem.nodeB].x - nodes[elem.nodeA].x};
        const float c{nodes[elem.nodeB].y - nodes[elem.nodeA].y};
        const float b{nodes[elem.nodeC].x - nodes[elem.nodeA].x};
        const float d{nodes[elem.nodeC].y - nodes[elem.nodeA].y};
        const float detB(std::abs(a*d - b*c));

        const float bbdd{b*b + d*d};
        const float abcd{a*b + c*d};
        const float aacc{a*a + c*c};

        matrix.add_global(elem.nodeA, elem.nodeA, (  3.0*bbdd -   6.0*abcd +   3.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeA, elem.nodeB, (      bbdd -       abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeA, elem.nodeC, (           -       abcd +       aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeA, elem.nodeD, (  4.0*bbdd -   4.0*abcd             ) / (6.0 * detB));
        //trix.add_global(elem.nodeA, elem.nodeE, (                                    ) / (6.0 * detB));
        matrix.add_global(elem.nodeA, elem.nodeF, (           -   4.0*abcd +   4.0*aacc) / (6.0 * detB));

        matrix.add_global(elem.nodeB, elem.nodeA, (  3.0*bbdd -   6.0*abcd +   3.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeB, elem.nodeB, (  3.0*bbdd                          ) / (6.0 * detB));
        matrix.add_global(elem.nodeB, elem.nodeC, (                   abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeB, elem.nodeD, (-12.0*bbdd +   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeB, elem.nodeE, (           -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeB, elem.nodeF, (               8.0*abcd             ) / (6.0 * detB));

        matrix.add_global(elem.nodeC, elem.nodeA, (           -       abcd +       aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeC, elem.nodeB, (                   abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeC, elem.nodeC, (                            3.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeC, elem.nodeD, (               8.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeC, elem.nodeE, (           -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeC, elem.nodeF, (               4.0*abcd -  12.0*aacc) / (6.0 * detB));

        matrix.add_global(elem.nodeD, elem.nodeA, (  4.0*bbdd -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeD, elem.nodeB, (-12.0*bbdd +   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeD, elem.nodeC, (               8.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeD, elem.nodeD, (200.0*bbdd -  72.0*abcd +   8.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeD, elem.nodeE, (-32.0*bbdd +  40.0*abcd -   8.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeD, elem.nodeF, ( 32.0*bbdd - 200.0*abcd +  32.0*aacc) / (6.0 * detB));

        //trix.add_global(elem.nodeE, elem.nodeA, (                                    ) / (6.0 * detB));
        matrix.add_global(elem.nodeE, elem.nodeB, (           -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeE, elem.nodeC, (           -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeE, elem.nodeD, (-32.0*bbdd +  40.0*abcd -   8.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeE, elem.nodeE, (  8.0*bbdd -   8.0*abcd +   8.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeE, elem.nodeF, ( -8.0*bbdd +  40.0*abcd -  32.0*aacc) / (6.0 * detB));

        matrix.add_global(elem.nodeF, elem.nodeA, (           -   4.0*abcd +   4.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeF, elem.nodeB, (               8.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeF, elem.nodeC, (               4.0*abcd -  12.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeF, elem.nodeD, ( 32.0*bbdd - 200.0*abcd +  32.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeF, elem.nodeE, ( -8.0*bbdd +  40.0*abcd -  32.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeF, elem.nodeF, (  8.0*bbdd -  72.0*abcd + 200.0*aacc) / (6.0 * detB));

    }
}


void mesh_q2(std::string filename, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements, size_t& highest_edgenode)
{
    std::ifstream fin(filename);

    std::string tmp;
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
                highest_edgenode = elements[current].nodeB;
            if (elements[current].nodeC > highest_edgenode)
                highest_edgenode = elements[current].nodeC;
            ++current;
        }
        else
        {
            fin.ignore(256, '\n');
        }
    }

    fin.close();
    elements.resize(current);
}
