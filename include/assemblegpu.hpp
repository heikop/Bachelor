#ifndef __ASSEMBLEGPU_HPP_
#define __ASSEMBLEGPU_HPP_

#include <cmath>

#include "csrmatrixgpu.hpp"
#include "global.hpp"

void assemble_atomic(size_t* d_rowptr, size_t* d_colind, float* d_values, size_t numrows, FullTriangle* d_elements, size_t numelem);
void assemble_gpu_atomic(CsrMatrixGpu& matrix, std::vector<FullTriangle>& h_elements)
{
    assemble_atomic(matrix._rowptr, matrix._colind, matrix._values, matrix._numrows, h_elements.data(), h_elements.size());
}

#endif
