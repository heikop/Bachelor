#ifndef __ASSEMBLEGPU_HPP_
#define __ASSEMBLEGPU_HPP_

#include <cmath>

#include "csrmatrix.hpp"
#include "global.hpp"

void assemble_atomic(size_t* rowptr, size_t* colind, float* values, size_t numrows, FullTriangle* elements, size_t numelem);
void assemble_gpu_atomic(CsrMatrix& matrix, std::vector<FullTriangle>& elements)
{
    assemble_atomic(matrix._rowptr, matrix._colind, matrix._values, matrix._numrows, elements.data(), elements.size());
}

#endif
