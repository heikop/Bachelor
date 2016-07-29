#ifndef __CSRMATRIXCPU_HPP_
#define __CSRMATRIXCPU_HPP_

#include <cassert>
#include <vector>
#include <iostream>

#include "global.hpp"
#include "mpihandler.hpp"

class CsrMatrixCpu
{
public:
    CsrMatrixCpu(const size_t, const size_t);
    CsrMatrixCpu(const size_t);
    CsrMatrixCpu(const CsrMatrixCpu&);
    CsrMatrixCpu(CsrMatrixCpu&&);
    ~CsrMatrixCpu();
    CsrMatrixCpu operator=(const CsrMatrixCpu&);
    CsrMatrixCpu operator=(const CsrMatrixCpu&&);

    void createStructure(const Triangle* const elements, const size_t num_elem);

    void set_local(const size_t, const size_t, const float);
    void add_local(const size_t, const size_t, const float);
    void set_global(const size_t, const size_t, const float);
    void add_global(const size_t, const size_t, const float);

    void print_local_data(const size_t firstindex);
//private:
    size_t _numrows, _numcols;
    size_t* _rowptr;
    size_t* _colind;
    float* _values;
};

#endif
