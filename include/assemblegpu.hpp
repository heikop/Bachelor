#ifndef __ASSEMBLEGPU_HPP_
#define __ASSEMBLEGPU_HPP_

#include <cmath>

#include "csrmatrixgpu.hpp"
#include "global.hpp"

void assemble_atomic(size_t* d_rowptr, size_t* d_colind, float* d_values, size_t numrows, FullTriangle* h_elements, size_t numelem, size_t* h_boundaryNodes, size_t numboundaryNodes);
void assemble_gpu_atomic(CsrMatrixGpu& matrix, std::vector<FullTriangle>& h_elements, std::vector<size_t>& h_boundaryNodes)
{
    assemble_atomic(matrix._rowptr, matrix._colind, matrix._values, matrix._numrows, h_elements.data(), h_elements.size(), h_boundaryNodes.data(), h_boundaryNodes.size());
}

#endif
