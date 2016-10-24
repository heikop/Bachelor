#ifndef __STRUCTURE_HPP_
#define __STRUCTURE_HPP_

#include "csrmatrixcpu.hpp"

template<typename datatype>
void structure(CsrMatrixCpu<datatype>& mat, const std::vector<Element<datatype>*>& elements)
{
    const size_t num_elem{elements.size()};
//    const size_t max_rowlength{40};
    const size_t max_rowlength{60};

    size_t* num_nonzeros = new size_t[mat._numrows];
    for (size_t i{0}; i < mat._numrows; ++i)
        num_nonzeros[i] = 0;

    size_t* colind = new size_t[max_rowlength*mat._numrows];

    for (size_t i{0}; i < num_elem; ++i)
    {
        size_t nodes[elements[i]->num_basis_functions()];
        for (size_t id{0}; id < elements[i]->num_basis_functions(); ++id)
            nodes[id] = elements[i]->vertexids()[id];
        for (size_t node1{0}; node1 < elements[i]->num_basis_functions(); ++node1)
        {
            for (size_t node2{0}; node2 < elements[i]->num_basis_functions(); ++node2)
            {
                size_t a{nodes[node1]};
                size_t b{nodes[node2]};
                if (a >= 0 && static_cast<size_t>(a) < mat._numrows)
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

    for (size_t i{0}; i < mat._numrows; ++i)
        for (size_t a{num_nonzeros[i]-1}; a > 0; --a)
            for (size_t b(0); b < a; ++b)
                if (colind[i*max_rowlength + b] > colind[i*max_rowlength + b+1])
                {
                    size_t tmp(colind[i*max_rowlength + b]);
                    colind[i*max_rowlength + b] = colind[i*max_rowlength + b+1];
                    colind[i*max_rowlength + b+1] = tmp;
                }

    size_t num_values{0};
    for (size_t i{0}; i < mat._numrows; ++i)
    {
        mat._rowptr[i] = num_values;
        num_values += num_nonzeros[i];
    }
    mat._rowptr[mat._numrows] = num_values;
    delete[] mat._colind;
    delete[] mat._values;
    mat._colind = new size_t[num_values];
    mat._values = new datatype[num_values];

    size_t current_pos{0};
    for (size_t row{0}; row < mat._numrows; ++row)
        for (size_t col{0}; col < num_nonzeros[row]; ++col)
            mat._colind[current_pos++] = colind[row*max_rowlength + col];
    for (size_t i{0}; i < num_values; ++i)
        mat._values[i] = 0.0;

    delete[] num_nonzeros;
    delete[] colind;
}

#endif
