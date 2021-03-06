#include "include/assemblecpu.hpp"

//TODO MPI: this is not mpi ready!
//TODO: midpoint nodes
void assemble_cpu_nag_id(CsrMatrixCpu<double>& matrix,
                         std::vector<size_t>& num_neighbours,
                         std::vector<size_t>& nag,
                         std::vector<size_t>& num_midpoints,
                         std::vector<size_t>& gaps,
                         std::vector<size_t>& num_gaps,
                         std::vector<Node>& nodes)
{
std::cout << "start assembling" << std::endl;
    assert(matrix._numrows == matrix._numcols &&
           matrix._numrows == nodes.size()           &&
           matrix._numrows == num_neighbours.size()    );

    // create structure
    const size_t num_nodes{nodes.size()};
    matrix._rowptr[0] = 0;
    for (size_t row{0}; row < num_nodes; ++row)
        matrix._rowptr[row + 1] = matrix._rowptr[row] + num_neighbours[row] + 1;
    matrix._colind = new size_t[matrix._rowptr[num_nodes]];
    matrix._values = new double[matrix._rowptr[num_nodes]];
    for (size_t i{0}; i < matrix._rowptr[num_nodes]; ++i)
        matrix._values[i] = 0.0;

    // fill _colind from nag
    for (size_t row{0}; row < num_nodes; ++row)
    {
        for (size_t colentry{0}; colentry < num_neighbours[row]; ++colentry)
            matrix._colind[matrix._rowptr[row] + colentry] = nag[matrix._rowptr[row] - row + colentry];
        matrix._colind[matrix._rowptr[row] + num_neighbours[row]] = row;
    }
    // sort _colind
    for (size_t row{0}; row < matrix._numrows; ++row)
        for (size_t a(num_neighbours[row]); a > 0; --a)
            for (size_t b{0}; b < a; ++b)
                if (matrix._colind[matrix._rowptr[row] + b] > matrix._colind[matrix._rowptr[row] + b+1])
                {
                    size_t tmp(matrix._colind[matrix._rowptr[row] + b]);
                    matrix._colind[matrix._rowptr[row] + b] = matrix._colind[matrix._rowptr[row] + b+1];
                    matrix._colind[matrix._rowptr[row] + b+1] = tmp;
                }


    // fill entries
    for (size_t row{0}; row < num_nodes; ++row)
    {
        if (num_gaps[row] == 0)
        {
            for (size_t e{0}; e < num_neighbours[row]-1; ++e)
            {
                // point A: actual point - row
                // point B: e's neighbour
                // point C: e+1's neighbour
                //         C
                //        / |
                //        |  |
                //       /    |
                //      A------B
                double B[2][2];
                B[0][0] = nodes[nag[matrix._rowptr[row] - row + e    ]].x - nodes[row].x;
                B[1][0] = nodes[nag[matrix._rowptr[row] - row + e    ]].y - nodes[row].y;
                B[0][1] = nodes[nag[matrix._rowptr[row] - row + e + 1]].x - nodes[row].x;
                B[1][1] = nodes[nag[matrix._rowptr[row] - row + e + 1]].y - nodes[row].y;
//std::cout << B[0][0] << ", " << B[0][1] << ", " << B[1][0] << ", " << B[1][1] << ", " << std::endl;
                double detB{std::abs(B[0][0]*B[1][1] - B[0][1]*B[1][0])};
                double gradA[2], gradB[2], gradC[2]; // in fact these are the gradients multplied by det(B)
                gradA[0] = B[1][0] - B[1][1];
                gradA[1] = B[0][1] - B[0][0];
                gradB[0] = B[1][1];
                gradB[1] = -B[0][1];
                gradC[0] = -B[1][0];
                gradC[1] = B[0][0];
                // TODO TOCHECK: dont add this directly into the matrix. do this at the end -> faster?
                matrix.add(row, row                                   , (gradA[0]*gradA[0] + gradA[1]*gradA[1]) / 2.0 / detB);
                matrix.add(row, nag[matrix._rowptr[row] - row + e    ], (gradA[0]*gradB[0] + gradA[1]*gradB[1]) / 2.0 / detB);
                matrix.add(row, nag[matrix._rowptr[row] - row + e + 1], (gradA[0]*gradC[0] + gradA[1]*gradC[1]) / 2.0 / detB);
//                std::cout << row << std::endl;
//                std::cout << row << ": " << (gradA[0]*gradA[0] + gradA[1]*gradA[1]) / 2.0 / detB << ", "
//                          << nag[matrix._rowptr[row] - row + e    ] << ": " << ((gradA[0]*gradB[0] + gradA[1]*gradB[1]) / 2.0 / detB) << ", "
//                          << nag[matrix._rowptr[row] - row + e + 1] << ": " << ((gradA[0]*gradC[0] + gradA[1]*gradC[1]) / 2.0 / detB) << std::endl;
            }
            // e = num_neighbours[row] - 1; e + 1 = 0;
                double B[2][2];
                B[0][0] = nodes[nag[matrix._rowptr[row] - row + num_neighbours[row] - 1]].x - nodes[row].x;
                B[1][0] = nodes[nag[matrix._rowptr[row] - row + num_neighbours[row] - 1]].y - nodes[row].y;
                B[0][1] = nodes[nag[matrix._rowptr[row] - row /* e+1 = 0 */            ]].x - nodes[row].x;
                B[1][1] = nodes[nag[matrix._rowptr[row] - row /* e+1 = 0 */            ]].y - nodes[row].y;
                double detB(std::abs(B[0][0]*B[1][1] - B[0][1]*B[1][0]));
                double gradA[2], gradB[2], gradC[2]; // in fact these are the gradients multplied by det(B)
                gradA[0] = B[1][0] - B[1][1];
                gradA[1] = B[0][1] - B[0][0];
                gradB[0] = B[1][1];
                gradB[1] = -B[0][1];
                gradC[0] = -B[1][0];
                gradC[1] = B[0][0];
                // TODO TOCHECK: dont add this directly into the matrix. do this at the end -> faster?
                matrix.add(row, row                                                     , (gradA[0]*gradA[0] + gradA[1]*gradA[1]) / 2.0 / detB);
                matrix.add(row, nag[matrix._rowptr[row] - row + num_neighbours[row] - 1], (gradA[0]*gradB[0] + gradA[1]*gradB[1]) / 2.0 / detB);
                matrix.add(row, nag[matrix._rowptr[row] - row /* e+1 = 0 */            ], (gradA[0]*gradC[0] + gradA[1]*gradC[1]) / 2.0 / detB);
        }
        else
        {
            //TODO
            for (size_t i{0}; i < num_neighbours[i]; ++i)
                matrix.set(row, nag[matrix._rowptr[row] - row + i], 0.0);
            matrix.set(row, row, 1.0);
        }
    }
}


void assemble_cpu_elem(CsrMatrixCpu<double>& matrix, std::vector<FullTriangle>& elements, std::vector<size_t>& boundaryNodes)
{
    for (const auto& elem : elements)
    {
        double B[2][2];
        B[0][0] = elem.nodeB.x - elem.nodeA.x;
        B[1][0] = elem.nodeB.y - elem.nodeA.y;
        B[0][1] = elem.nodeC.x - elem.nodeA.x;
        B[1][1] = elem.nodeC.y - elem.nodeA.y;
        //double d[2];
        //d[0] = elem.nodeA.x;
        //d[1] = elem.nodeA.y;
        double detB(std::abs(B[0][0]*B[1][1] - B[0][1]*B[1][0]));
        double gradA[2], gradB[2], gradC[2]; // in fact these are the gradients multplied by det(B)
        gradA[0] = B[1][0] - B[1][1];
        gradA[1] = B[0][1] - B[0][0];
        gradB[0] = B[1][1];
        gradB[1] = -B[0][1];
        gradC[0] = -B[1][0];
        gradC[1] = B[0][0];

        matrix.add(elem.nodeA.ID, elem.nodeA.ID, (gradA[0]*gradA[0] + gradA[1]*gradA[1]) / 2.0 / detB);
        matrix.add(elem.nodeA.ID, elem.nodeB.ID, (gradA[0]*gradB[0] + gradA[1]*gradB[1]) / 2.0 / detB);
        matrix.add(elem.nodeA.ID, elem.nodeC.ID, (gradA[0]*gradC[0] + gradA[1]*gradC[1]) / 2.0 / detB);
        matrix.add(elem.nodeB.ID, elem.nodeA.ID, (gradB[0]*gradA[0] + gradB[1]*gradA[1]) / 2.0 / detB);
        matrix.add(elem.nodeB.ID, elem.nodeB.ID, (gradB[0]*gradB[0] + gradB[1]*gradB[1]) / 2.0 / detB);
        matrix.add(elem.nodeB.ID, elem.nodeC.ID, (gradB[0]*gradC[0] + gradB[1]*gradC[1]) / 2.0 / detB);
        matrix.add(elem.nodeC.ID, elem.nodeA.ID, (gradC[0]*gradA[0] + gradC[1]*gradA[1]) / 2.0 / detB);
        matrix.add(elem.nodeC.ID, elem.nodeB.ID, (gradC[0]*gradB[0] + gradC[1]*gradB[1]) / 2.0 / detB);
        matrix.add(elem.nodeC.ID, elem.nodeC.ID, (gradC[0]*gradC[0] + gradC[1]*gradC[1]) / 2.0 / detB);
    }
    for (const auto bid : boundaryNodes)
        for (size_t i{matrix._rowptr[bid]}; i < matrix._rowptr[bid + 1]; ++i)
            matrix._values[i] = (matrix._colind[i] == bid ? 1.0 : 0.0);
}
