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
//*
    double tol{1.0e-5}; // TODO local is not good...
    size_t maxit{200000}; // the same...

    vtype residual(_rhs._size);
    _matrix.multvec(res, residual);
    residual.axpby(1.0, _rhs, -1.0);

    vtype r_hat(residual);

    double rho{1.0};
    double alpha{1.0};
    double omega{1.0};
    double beta{0.0};

    vtype v(_rhs._size, 0.0);
    vtype p(_rhs._size, 0.0);
    vtype s(_rhs._size, 0.0);
    vtype t(_rhs._size, 0.0);

    for (size_t it{0}; std::sqrt(residual.l2norm2()) > tol && it < maxit; ++it)
    {
std::cout << "after " << it << " iteratons" << ": " << std::sqrt(residual.l2norm2()) << std::endl;
        beta = alpha / rho / omega;         // 2.
        rho = r_hat.dot_vec(residual);      // 1.
        beta *= rho;                        // 2.
        p.axpy(-omega, v);                  // 3.
        p.axpby(1.0, residual, beta);       // 3.
        _matrix.multvec(p, v);              // 4.
        alpha = rho / r_hat.dot_vec(v);     // 5.
        res.axpy(alpha, p);                 // 6.
        // 7. quit?
        s.copy(residual);                   // 8.
        s.axpy(-alpha, v);                  // 8.
        _matrix.multvec(s, t);              // 9.
        omega = t.dot_vec(s) / t.l2norm2(); // 10.
        res.axpy(omega, s);                 // 11.
        // 12. quit?
        residual.copy(s);                   // 13.
        residual.axpy(-omega, t);           // 13.
    }
std::cout << "stopped with an residual of " << std::sqrt(residual.l2norm2()) << std::endl;
//*/
/*
    assert(res._size == _matrix._numcols);
    double tol{1.0e-8}; // TODO local is not good...
    //size_t maxit{2000}; // the same...
    size_t maxit{200000}; // the same...
    //size_t maxit{24}; // the same...

    vtype residual(_rhs._size);
    _matrix.multvec(res, residual);
    residual.axpby(1.0, _rhs, -1.0);
    vtype direction(residual);
    vtype mat_mult_dir(_rhs._size);
    double alpha{residual.l2norm2()};
    double beta{0.0};
    for (size_t it{0}; alpha > tol && it < maxit; ++it)
    {
std::cout << "after " << it << " iteratons" << ": " << alpha << std::endl;
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
*/
}

//#endif
