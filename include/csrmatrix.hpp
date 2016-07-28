#ifndef __CSRMATRIX_HPP_
#define __CSRMATRIX_HPP_

#include <cassert>
#include <vector>
#include <iostream>

#include "global.hpp"
#include "mpihandler.hpp"

class CsrMatrix
{
public:
    CsrMatrix(const size_t, const size_t);
    CsrMatrix(const size_t);
    CsrMatrix(const CsrMatrix&);
    CsrMatrix(CsrMatrix&&);
    ~CsrMatrix();
    CsrMatrix operator=(const CsrMatrix&);
    CsrMatrix operator=(const CsrMatrix&&);

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
