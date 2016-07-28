#ifndef __ASSEMBLECPU_HPP_
#define __ASSEMBLECPU_HPP_

#include <vector>
#include <cmath>

#include "csrmatrix.hpp"
#include "global.hpp"

void assemble_cpu_elem(CsrMatrix& matrix, std::vector<FullTriangle>& elements);

#endif
