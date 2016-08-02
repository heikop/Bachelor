#ifndef __CGSOLVER_HPP_
#define __CGSOLVER_HPP_

#include <cassert>
#include "matrix.hpp"
#include "vector.hpp"

template<typename mtype, typename vtype> //TODO not all should be allowed
class CgSolver
{
public:
    CgSolver(const mtype&, const vtype&);
    void solve(vtype&);
//private:
    const mtype _matrix; // TOCHECK why no references possible?
    const vtype _rhs;
};

#include "cgsolver.tpp"

#endif
