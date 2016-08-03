//#ifndef __CGSOLVER_TPP_
//#define __CGSOLVER_TPP_
//#include "cgsolver.hpp"
#include <iostream>

template<typename mtype, typename vtype>
CgSolver<mtype, vtype>::CgSolver(const mtype& matrix, const vtype& rhs):
    _matrix{matrix},
    _rhs{rhs}
{
    assert(_matrix._numrows == _matrix._numcols && _matrix._numrows == _rhs._size);
}

template<typename mtype, typename vtype>
void CgSolver<mtype, vtype>::solve(vtype& res)
{
    assert(res._size == _matrix._numcols);
    float tol{1.0e-8}; // TODO local is not good...
    size_t maxit{2000}; // the same...

    vtype residual(_rhs._size);
    _matrix.multvec(res, residual);
    residual.axpby(1.0, _rhs, -1.0);
    vtype direction(residual);
    vtype mat_mult_dir(_rhs._size);
    float alpha{residual.l2norm2()};
    float beta{0.0};
    for (size_t it{0}; alpha > tol && it < maxit; ++it)
    {
//std::cout << "after " << it << " iteratons" << ": " << alpha << std::endl;
        _matrix.multvec(direction, mat_mult_dir);
        beta = 1.0 / alpha;
        alpha /= direction.dot_vec(mat_mult_dir);
        res.axpy(alpha, direction);
        residual.axpy(-alpha, mat_mult_dir);
        alpha = residual.l2norm2();
        beta *= alpha;
        direction.axpby(1.0, residual, beta);
    }
std::cout << "stopped with an residual of " << std::sqrt(alpha) << std::endl;
}

//#endif
