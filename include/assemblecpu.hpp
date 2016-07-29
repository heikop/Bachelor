#ifndef __ASSEMBLECPU_HPP_
#define __ASSEMBLECPU_HPP_

#include <vector>
#include <cmath>

#include "csrmatrixcpu.hpp"
#include "global.hpp"

void assemble_cpu_elem(CsrMatrixCpu& matrix, std::vector<FullTriangle>& elements);

#endif
