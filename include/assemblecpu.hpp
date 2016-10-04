#ifndef __ASSEMBLECPU_HPP_
#define __ASSEMBLECPU_HPP_

#include <vector>
#include <cmath>

#include "csrmatrixcpu.hpp"
#include "global.hpp"

void assemble_cpu_elem(CsrMatrixCpu& matrix, std::vector<FullTriangle>& elements, std::vector<size_t>& boundaryNodes);
void assemble_cpu_nag_id(CsrMatrixCpu& matrix,
                         std::vector<size_t>& num_neighbours,
                         std::vector<size_t>& nag,
                         std::vector<size_t>& num_midpoints,
                         std::vector<size_t>& gaps,
                         std::vector<size_t>& num_gaps,
                         std::vector<Node>& nodes);

#endif
