#include "include/assemble.hpp"

/////////////////////////////////////////// nag id
void nag_q2_id(std::string filename,
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

    std::ifstream fin(filename);

    std::string tmp;
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

/////////////////////////////////////////// mesh full
void assemble_full(CsrMatrixCpu& matrix, std::vector<FullTriangleQ2>& elements)
{
    for (const auto& elem : elements)
    {
        // B = [a b]
        //     [c d]
        //const double a{elem.nodeB.x - elem.nodeA.x};
        //const double c{elem.nodeB.y - elem.nodeA.y};
        //const double b{elem.nodeC.x - elem.nodeA.x};
        //const double d{elem.nodeC.y - elem.nodeA.y};
        const double a{elem.nodeB.x - elem.nodeA.x};
        const double c{elem.nodeB.y - elem.nodeA.y};
        const double b{elem.nodeC.x - elem.nodeA.x};
        const double d{elem.nodeC.y - elem.nodeA.y};
        const double detB{std::abs(a*d - b*c)};

        const double bbdd{b*b + d*d};
        const double abcd{a*b + c*d};
        const double aacc{a*a + c*c};

        matrix.add_global(elem.nodeA.ID, elem.nodeA.ID, (  3.0*bbdd -   6.0*abcd +   3.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeA.ID, elem.nodeB.ID, (      bbdd -       abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeA.ID, elem.nodeC.ID, (           -       abcd +       aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeA.ID, elem.nodeD   , (  4.0*bbdd -   4.0*abcd             ) / (6.0 * detB));
        //trix.add_global(elem.nodeA.ID, elem.nodeE   , (                                    ) / (6.0 * detB));
        matrix.add_global(elem.nodeA.ID, elem.nodeF   , (           -   4.0*abcd +   4.0*aacc) / (6.0 * detB));

        matrix.add_global(elem.nodeB.ID, elem.nodeA.ID, (  3.0*bbdd -   6.0*abcd +   3.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeB.ID, elem.nodeB.ID, (  3.0*bbdd                          ) / (6.0 * detB));
        matrix.add_global(elem.nodeB.ID, elem.nodeC.ID, (                   abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeB.ID, elem.nodeD   , (-12.0*bbdd +   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeB.ID, elem.nodeE   , (           -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeB.ID, elem.nodeF   , (               8.0*abcd             ) / (6.0 * detB));

        matrix.add_global(elem.nodeC.ID, elem.nodeA.ID, (           -       abcd +       aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeC.ID, elem.nodeB.ID, (                   abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeC.ID, elem.nodeC.ID, (                            3.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeC.ID, elem.nodeD   , (               8.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeC.ID, elem.nodeE   , (           -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeC.ID, elem.nodeF   , (               4.0*abcd -  12.0*aacc) / (6.0 * detB));

        matrix.add_global(elem.nodeD   , elem.nodeA.ID, (  4.0*bbdd -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeD   , elem.nodeB.ID, (-12.0*bbdd +   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeD   , elem.nodeC.ID, (               8.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeD   , elem.nodeD   , (200.0*bbdd -  72.0*abcd +   8.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeD   , elem.nodeE   , (-32.0*bbdd +  40.0*abcd -   8.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeD   , elem.nodeF   , ( 32.0*bbdd - 200.0*abcd +  32.0*aacc) / (6.0 * detB));

        //trix.add_global(elem.nodeE   , elem.nodeA.ID, (                                    ) / (6.0 * detB));
        matrix.add_global(elem.nodeE   , elem.nodeB.ID, (           -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeE   , elem.nodeC.ID, (           -   4.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeE   , elem.nodeD   , (-32.0*bbdd +  40.0*abcd -   8.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeE   , elem.nodeE   , (  8.0*bbdd -   8.0*abcd +   8.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeE   , elem.nodeF   , ( -8.0*bbdd +  40.0*abcd -  32.0*aacc) / (6.0 * detB));

        matrix.add_global(elem.nodeF   , elem.nodeA.ID, (           -   4.0*abcd +   4.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeF   , elem.nodeB.ID, (               8.0*abcd             ) / (6.0 * detB));
        matrix.add_global(elem.nodeF   , elem.nodeC.ID, (               4.0*abcd -  12.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeF   , elem.nodeD   , ( 32.0*bbdd - 200.0*abcd +  32.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeF   , elem.nodeE   , ( -8.0*bbdd +  40.0*abcd -  32.0*aacc) / (6.0 * detB));
        matrix.add_global(elem.nodeF   , elem.nodeF   , (  8.0*bbdd -  72.0*abcd + 200.0*aacc) / (6.0 * detB));

    }
}

void structure_full(CsrMatrixCpu& matrix, std::vector<FullTriangleQ2>& elements)
{
    const size_t max_rowlength{40};

    size_t* num_nonzeros = new size_t[matrix._numrows_local];
    for (size_t i{0}; i < matrix._numrows_local; ++i)
        num_nonzeros[i] = 0;

    size_t* colind = new size_t[max_rowlength*matrix._numrows_local];

    for (size_t i{0}; i < elements.size(); ++i)
    {
        size_t nodes[6];
        nodes[0] = elements[i].nodeA.ID;
        nodes[1] = elements[i].nodeB.ID;
        nodes[2] = elements[i].nodeC.ID;
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
    matrix._values = new double[num_values];

    size_t current_pos{0};
    for (size_t row{0}; row < matrix._numrows_local; ++row)
        for (size_t col{0}; col < num_nonzeros[row]; ++col)
            matrix._colind[current_pos++] = colind[row*max_rowlength + col];
    for (size_t i{0}; i < num_values; ++i)
        matrix._values[i] = 0.0;

    delete[] num_nonzeros;
    delete[] colind;
}

void mesh_q2_full(std::string filename, std::vector<FullTriangleQ2>& elements, size_t& numnodes)
{
    std::ifstream fin(filename);

    std::string tmp;
    do { fin >> tmp; } while(tmp != "$Nodes");
    size_t num_nodes; fin >> num_nodes;
    std::vector<Node> nodes(num_nodes);
    double z_coord_trash;
    for (size_t i{0}; i < num_nodes; ++i)
    {
        fin >> nodes[i].ID >> nodes[i].x >> nodes[i].y >> z_coord_trash;
        --nodes[i].ID;
    }

    numnodes = num_nodes;

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
            fin >> elements[current].nodeA.ID >> elements[current].nodeB.ID >> elements[current].nodeC.ID
                >> elements[current].nodeD    >> elements[current].nodeE    >> elements[current].nodeF;
            elements[current].nodeA.ID -= 1;
            elements[current].nodeB.ID -= 1;
            elements[current].nodeC.ID -= 1;
            elements[current].nodeD    -= 1;
            elements[current].nodeE    -= 1;
            elements[current].nodeF    -= 1;
            ++current;
        }
        else
        {
            fin.ignore(256, '\n');
        }
    }

    fin.close();
    elements.resize(current);

    for (auto& elem : elements)
    {
        elem.nodeA.x = nodes[elem.nodeA.ID].x;
        elem.nodeA.y = nodes[elem.nodeA.ID].y;
        elem.nodeB.x = nodes[elem.nodeB.ID].x;
        elem.nodeB.y = nodes[elem.nodeB.ID].y;
        elem.nodeC.x = nodes[elem.nodeC.ID].x;
        elem.nodeC.y = nodes[elem.nodeC.ID].y;
    }
}

/////////////////////////////////////////// mesh ID
void assemble_id(CsrMatrixCpu& matrix, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements)
{
    for (const auto& elem : elements)
    {
        // B = [a b]
        //     [c d]
        const double a{nodes[elem.nodeB].x - nodes[elem.nodeA].x};
        const double c{nodes[elem.nodeB].y - nodes[elem.nodeA].y};
        const double b{nodes[elem.nodeC].x - nodes[elem.nodeA].x};
        const double d{nodes[elem.nodeC].y - nodes[elem.nodeA].y};
        const double detB{std::abs(a*d - b*c)};

        //const double A{ d/detB};
        //const double B{-c/detB};
        //const double C{-b/detB};
        //const double D{ a/detB};
        //const double bbdd{B*B + D*D};
        //const double abcd{A*B + C*D};
        //const double aacc{A*A + C*C};
        const double bbdd{ c*c + a*a};
        const double abcd{-c*d - a*b};
        const double aacc{ d*d + b*b};

        matrix.add_global(elem.nodeA, elem.nodeA, ( 3.0*aacc + 6.0*abcd + 3.0*bbdd ) / detB / 6.0);
        matrix.add_global(elem.nodeA, elem.nodeB, (     aacc +     abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeA, elem.nodeC, (                abcd +     bbdd ) / detB / 6.0);
        matrix.add_global(elem.nodeA, elem.nodeD, (-4.0*aacc - 4.0*abcd            ) / detB / 6.0);
        //trix.add_global(elem.nodeA, elem.nodeE, (                                ) / detB / 6.0);
        matrix.add_global(elem.nodeA, elem.nodeF, (          - 4.0*abcd - 4.0*bbdd ) / detB / 6.0);

        matrix.add_global(elem.nodeB, elem.nodeA, (     aacc +     abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeB, elem.nodeB, ( 3.0*aacc                       ) / detB / 6.0);
        matrix.add_global(elem.nodeB, elem.nodeC, (          -     abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeB, elem.nodeD, (-4.0*aacc - 4.0*abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeB, elem.nodeE, (            4.0*abcd            ) / detB / 6.0);
        //trix.add_global(elem.nodeB, elem.nodeF, (                                ) / detB / 6.0);

        matrix.add_global(elem.nodeC, elem.nodeA, (                abcd +     bbdd ) / detB / 6.0);
        matrix.add_global(elem.nodeC, elem.nodeB, (          -     abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeC, elem.nodeC, (                       3.0*bbdd ) / detB / 6.0);
        //trix.add_global(elem.nodeC, elem.nodeD, (                                ) / detB / 6.0);
        matrix.add_global(elem.nodeC, elem.nodeE, (            4.0*abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeC, elem.nodeF, (          - 4.0*abcd - 4.0*bbdd ) / detB / 6.0);

        matrix.add_global(elem.nodeD, elem.nodeA, (-4.0*aacc - 4.0*abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeD, elem.nodeB, (-4.0*aacc - 4.0*abcd            ) / detB / 6.0);
        //trix.add_global(elem.nodeD, elem.nodeC, (                                ) / detB / 6.0);
        matrix.add_global(elem.nodeD, elem.nodeD, ( 8.0*aacc + 8.0*abcd + 8.0*bbdd ) / detB / 6.0);
        matrix.add_global(elem.nodeD, elem.nodeE, (          - 8.0*abcd - 8.0*bbdd ) / detB / 6.0);
        matrix.add_global(elem.nodeD, elem.nodeF, (            8.0*abcd            ) / detB / 6.0);

        //trix.add_global(elem.nodeE, elem.nodeA, (                                ) / detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeB, (            4.0*abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeC, (            4.0*abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeD, (          - 8.0*abcd - 8.0*bbdd ) / detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeE, ( 8.0*aacc + 8.0*abcd + 8.0*bbdd ) / detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeF, (-8.0*aacc - 8.0*abcd            ) / detB / 6.0);

        matrix.add_global(elem.nodeF, elem.nodeA, (          - 4.0*abcd - 4.0*bbdd ) / detB / 6.0);
        //trix.add_global(elem.nodeF, elem.nodeB, (                                ) / detB / 6.0);
        matrix.add_global(elem.nodeF, elem.nodeC, (          - 4.0*abcd - 4.0*bbdd ) / detB / 6.0);
        matrix.add_global(elem.nodeF, elem.nodeD, (            8.0*abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeF, elem.nodeE, (-8.0*aacc - 8.0*abcd            ) / detB / 6.0);
        matrix.add_global(elem.nodeF, elem.nodeF, ( 8.0*aacc + 8.0*abcd + 8.0*bbdd ) / detB / 6.0);

        //matrix.add_global(elem.nodeA, elem.nodeA, ( 3.0*A*A + 6.0*A*B + 3.0*B*B
        //                                          + 3.0*C*C + 6.0*C*D + 3.0*D*D ) * detB / 6.0);
        //matrix.add_global(elem.nodeA, elem.nodeA, ( A*A + 2.0*A*B + B*B
        //                                          + C*C + 2.0*C*D + D*D ) * detB / 2.0);
        //matrix.add_global(elem.nodeA, elem.nodeA, ( (A+B)*(A+B)
        //                                          + (C+D)*(C+D) ) * detB / 2.0);
/*
        const double A{ d/detB};
        const double B{-c/detB};
        const double C{-b/detB};
        const double D{ a/detB};

        matrix.add_global(elem.nodeA, elem.nodeA, ( (-    A-    B)*(-    A-    B) + (-    C-    D)*(-    C-    D)
                                                  + (-    A-    B)*(-    A-    B) + (-    C-    D)*(-    C-    D)
                                                  + (     A+    B)*(     A+    B) + (     C+    D)*(     C+    D) ) * detB / 6.0);
        matrix.add_global(elem.nodeA, elem.nodeB, ( (-    A-    B)*(-    A      ) + (-    C-    D)*(-    C      )
                                                  + (-    A-    B)*(     A      ) + (-    C-    D)*(     C      )
                                                  + (     A+    B)*(     A      ) + (     C+    D)*(     C      ) ) * detB / 6.0);
        matrix.add_global(elem.nodeA, elem.nodeC, ( (-    A-    B)*(           B) + (-    C-    D)*(           D)
                                                  + (-    A-    B)*(      -    B) + (-    C-    D)*(      -    D)
                                                  + (     A+    B)*(           B) + (     C+    D)*(           D) ) * detB / 6.0);
        matrix.add_global(elem.nodeA, elem.nodeD, ( (-    A-    B)*( 2.0*A      ) + (-    C-    D)*( 2.0*C      )
                                                  + (-    A-    B)*(      -2.0*B) + (-    C-    D)*(      -2.0*D)
                                                  + (     A+    B)*(-2.0*A-2.0*B) + (     C+    D)*(-2.0*C-2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeA, elem.nodeE, ( (-    A-    B)*( 2.0*A      ) + (-    C-    D)*( 2.0*C      )
                                                  + (-    A-    B)*(       2.0*B) + (-    C-    D)*(       2.0*D)
                                                  + (     A+    B)*( 2.0*A+2.0*B) + (     C+    D)*( 2.0*C+2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeA, elem.nodeF, ( (-    A-    B)*(-2.0*A      ) + (-    C-    D)*(-2.0*C      )
                                                  + (-    A-    B)*(       2.0*B) + (-    C-    D)*(       2.0*D)
                                                  + (     A+    B)*(-2.0*A-2.0*B) + (     C+    D)*(-2.0*C-2.0*D) ) * detB / 6.0);

        matrix.add_global(elem.nodeB, elem.nodeA, ( (-    A      )*(-    A-    B) + (-    C      )*(-    C-    D)
                                                  + (     A      )*(-    A-    B) + (     C      )*(-    C-    D)
                                                  + (     A      )*(     A+    B) + (     C      )*(     C+    D) ) * detB / 6.0);
        matrix.add_global(elem.nodeB, elem.nodeB, ( (-    A      )*(-    A      ) + (-    C      )*(-    C      )
                                                  + (     A      )*(     A      ) + (     C      )*(     C      )
                                                  + (     A      )*(     A      ) + (     C      )*(     C      ) ) * detB / 6.0);
        matrix.add_global(elem.nodeB, elem.nodeC, ( (-    A      )*(           B) + (-    C      )*(           D)
                                                  + (     A      )*(      -    B) + (     C      )*(      -    D)
                                                  + (     A      )*(           B) + (     C      )*(           D) ) * detB / 6.0);
        matrix.add_global(elem.nodeB, elem.nodeD, ( (-    A      )*( 2.0*A      ) + (-    C      )*( 2.0*C      )
                                                  + (     A      )*(      -2.0*B) + (     C      )*(      -2.0*D)
                                                  + (     A      )*(-2.0*A-2.0*B) + (     C      )*(-2.0*C-2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeB, elem.nodeE, ( (-    A      )*( 2.0*A      ) + (-    C      )*( 2.0*C      )
                                                  + (     A      )*(       2.0*B) + (     C      )*(       2.0*D)
                                                  + (     A      )*( 2.0*A+2.0*B) + (     C      )*( 2.0*C+2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeB, elem.nodeF, ( (-    A      )*(-2.0*A      ) + (-    C      )*(-2.0*C      )
                                                  + (     A      )*(       2.0*B) + (     C      )*(       2.0*D)
                                                  + (     A      )*(-2.0*A-2.0*B) + (     C      )*(-2.0*C-2.0*D) ) * detB / 6.0);

        matrix.add_global(elem.nodeC, elem.nodeA, ( (           B)*(-    A-    B) + (           D)*(-    C-    D)
                                                  + (      -    B)*(-    A-    B) + (      -    D)*(-    C-    D)
                                                  + (           B)*(     A+    B) + (           D)*(     C+    D) ) * detB / 6.0);
        matrix.add_global(elem.nodeC, elem.nodeB, ( (           B)*(-    A      ) + (           D)*(-    C      )
                                                  + (      -    B)*(     A      ) + (      -    D)*(     C      )
                                                  + (           B)*(     A      ) + (           D)*(     C      ) ) * detB / 6.0);
        matrix.add_global(elem.nodeC, elem.nodeC, ( (           B)*(           B) + (           D)*(           D)
                                                  + (      -    B)*(      -    B) + (      -    D)*(      -    D)
                                                  + (           B)*(           B) + (           D)*(           D) ) * detB / 6.0);
        matrix.add_global(elem.nodeC, elem.nodeD, ( (           B)*( 2.0*A      ) + (           D)*( 2.0*C      )
                                                  + (      -    B)*(      -2.0*B) + (      -    D)*(      -2.0*D)
                                                  + (           B)*(-2.0*A-2.0*B) + (           D)*(-2.0*C-2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeC, elem.nodeE, ( (           B)*( 2.0*A      ) + (           D)*( 2.0*C      )
                                                  + (      -    B)*(       2.0*B) + (      -    D)*(       2.0*D)
                                                  + (           B)*( 2.0*A+2.0*B) + (           D)*( 2.0*C+2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeC, elem.nodeF, ( (           B)*(-2.0*A      ) + (           D)*(-2.0*C      )
                                                  + (      -    B)*(       2.0*B) + (      -    D)*(       2.0*D)
                                                  + (           B)*(-2.0*A-2.0*B) + (           D)*(-2.0*C-2.0*D) ) * detB / 6.0);

        matrix.add_global(elem.nodeD, elem.nodeA, ( ( 2.0*A      )*(-    A-    B) + ( 2.0*C      )*(-    C-    D)
                                                  + (      -2.0*B)*(-    A-    B) + (      -2.0*D)*(-    C-    D)
                                                  + (-2.0*A-2.0*B)*(     A+    B) + (-2.0*C-2.0*D)*(     C+    D) ) * detB / 6.0);
        matrix.add_global(elem.nodeD, elem.nodeB, ( ( 2.0*A      )*(-    A      ) + ( 2.0*C      )*(-    C      )
                                                  + (      -2.0*B)*(     A      ) + (      -2.0*D)*(     C      )
                                                  + (-2.0*A-2.0*B)*(     A      ) + (-2.0*C-2.0*D)*(     C      ) ) * detB / 6.0);
        matrix.add_global(elem.nodeD, elem.nodeC, ( ( 2.0*A      )*(           B) + ( 2.0*C      )*(           D)
                                                  + (      -2.0*B)*(      -    B) + (      -2.0*D)*(      -    D)
                                                  + (-2.0*A-2.0*B)*(           B) + (-2.0*C-2.0*D)*(           D) ) * detB / 6.0);
        matrix.add_global(elem.nodeD, elem.nodeD, ( ( 2.0*A      )*( 2.0*A      ) + ( 2.0*C      )*( 2.0*C      )
                                                  + (      -2.0*B)*(      -2.0*B) + (      -2.0*D)*(      -2.0*D)
                                                  + (-2.0*A-2.0*B)*(-2.0*A-2.0*B) + (-2.0*C-2.0*D)*(-2.0*C-2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeD, elem.nodeE, ( ( 2.0*A      )*( 2.0*A      ) + ( 2.0*C      )*( 2.0*C      )
                                                  + (      -2.0*B)*(       2.0*B) + (      -2.0*D)*(       2.0*D)
                                                  + (-2.0*A-2.0*B)*( 2.0*A+2.0*B) + (-2.0*C-2.0*D)*( 2.0*C+2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeD, elem.nodeF, ( ( 2.0*A      )*(-2.0*A      ) + ( 2.0*C      )*(-2.0*C      )
                                                  + (      -2.0*B)*(       2.0*B) + (      -2.0*D)*(       2.0*D)
                                                  + (-2.0*A-2.0*B)*(-2.0*A-2.0*B) + (-2.0*C-2.0*D)*(-2.0*C-2.0*D) ) * detB / 6.0);

        matrix.add_global(elem.nodeE, elem.nodeA, ( ( 2.0*A      )*(-    A-    B) + ( 2.0*C      )*(-    C-    D)
                                                  + (       2.0*B)*(-    A-    B) + (       2.0*D)*(-    C-    D)
                                                  + ( 2.0*A+2.0*B)*(     A+    B) + ( 2.0*C+2.0*D)*(     C+    D) ) * detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeB, ( ( 2.0*A      )*(-    A      ) + ( 2.0*C      )*(-    C      )
                                                  + (       2.0*B)*(     A      ) + (       2.0*D)*(     C      )
                                                  + ( 2.0*A+2.0*B)*(     A      ) + ( 2.0*C+2.0*D)*(     C      ) ) * detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeC, ( ( 2.0*A      )*(           B) + ( 2.0*C      )*(           D)
                                                  + (       2.0*B)*(      -    B) + (       2.0*D)*(      -    D)
                                                  + ( 2.0*A+2.0*B)*(           B) + ( 2.0*C+2.0*D)*(           D) ) * detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeD, ( ( 2.0*A      )*( 2.0*A      ) + ( 2.0*C      )*( 2.0*C      )
                                                  + (       2.0*B)*(      -2.0*B) + (       2.0*D)*(      -2.0*D)
                                                  + ( 2.0*A+2.0*B)*(-2.0*A-2.0*B) + ( 2.0*C+2.0*D)*(-2.0*C-2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeE, ( ( 2.0*A      )*( 2.0*A      ) + ( 2.0*C      )*( 2.0*C      )
                                                  + (       2.0*B)*(       2.0*B) + (       2.0*D)*(       2.0*D)
                                                  + ( 2.0*A+2.0*B)*( 2.0*A+2.0*B) + ( 2.0*C+2.0*D)*( 2.0*C+2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeE, elem.nodeF, ( ( 2.0*A      )*(-2.0*A      ) + ( 2.0*C      )*(-2.0*C      )
                                                  + (       2.0*B)*(       2.0*B) + (       2.0*D)*(       2.0*D)
                                                  + ( 2.0*A+2.0*B)*(-2.0*A-2.0*B) + ( 2.0*C+2.0*D)*(-2.0*C-2.0*D) ) * detB / 6.0);

        matrix.add_global(elem.nodeF, elem.nodeA, ( (-2.0*A      )*(-    A-    B) + (-2.0*C      )*(-    C-    D)
                                                  + (       2.0*B)*(-    A-    B) + (       2.0*D)*(-    C-    D)
                                                  + (-2.0*A-2.0*B)*(     A+    B) + (-2.0*C-2.0*D)*(     C+    D) ) * detB / 6.0);
        matrix.add_global(elem.nodeF, elem.nodeB, ( (-2.0*A      )*(-    A      ) + (-2.0*C      )*(-    C      )
                                                  + (       2.0*B)*(     A      ) + (       2.0*D)*(     C      )
                                                  + (-2.0*A-2.0*B)*(     A      ) + (-2.0*C-2.0*D)*(     C      ) ) * detB / 6.0);
        matrix.add_global(elem.nodeF, elem.nodeC, ( (-2.0*A      )*(           B) + (-2.0*C      )*(           D)
                                                  + (       2.0*B)*(      -    B) + (       2.0*D)*(      -    D)
                                                  + (-2.0*A-2.0*B)*(           B) + (-2.0*C-2.0*D)*(           D) ) * detB / 6.0);
        matrix.add_global(elem.nodeF, elem.nodeD, ( (-2.0*A      )*( 2.0*A      ) + (-2.0*C      )*( 2.0*C      )
                                                  + (       2.0*B)*(      -2.0*B) + (       2.0*D)*(      -2.0*D)
                                                  + (-2.0*A-2.0*B)*(-2.0*A-2.0*B) + (-2.0*C-2.0*D)*(-2.0*C-2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeF, elem.nodeE, ( (-2.0*A      )*( 2.0*A      ) + (-2.0*C      )*( 2.0*C      )
                                                  + (       2.0*B)*(       2.0*B) + (       2.0*D)*(       2.0*D)
                                                  + (-2.0*A-2.0*B)*( 2.0*A+2.0*B) + (-2.0*C-2.0*D)*( 2.0*C+2.0*D) ) * detB / 6.0);
        matrix.add_global(elem.nodeF, elem.nodeF, ( (-2.0*A      )*(-2.0*A      ) + (-2.0*C      )*(-2.0*C      )
                                                  + (       2.0*B)*(       2.0*B) + (       2.0*D)*(       2.0*D)
                                                  + (-2.0*A-2.0*B)*(-2.0*A-2.0*B) + (-2.0*C-2.0*D)*(-2.0*C-2.0*D) ) * detB / 6.0);
*/
    }
}

void structure_id(CsrMatrixCpu& matrix, std::vector<TriangleQ2>& elements)
{
    const size_t max_rowlength{40};

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
    matrix._values = new double[num_values];

    size_t current_pos{0};
    for (size_t row{0}; row < matrix._numrows_local; ++row)
        for (size_t col{0}; col < num_nonzeros[row]; ++col)
            matrix._colind[current_pos++] = colind[row*max_rowlength + col];
    for (size_t i{0}; i < num_values; ++i)
        matrix._values[i] = 0.0;

    delete[] num_nonzeros;
    delete[] colind;
}

void mesh_q2_id(std::string filename, std::vector<Node>& nodes, std::vector<TriangleQ2>& elements, size_t& highest_edgenode)
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
