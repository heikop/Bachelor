#ifndef __ASSEMBLEGPU_HPP_
#define __ASSEMBLEGPU_HPP_

#include <cmath>

#include "csrmatrix.hpp"
#include "global.hpp"

void assemble_gpu_atomic(CsrMatrix& matrix, std::vector<FullTriangle>& elements);

#endif
