#ifndef __CSRMATRIXGPU_HPP_
#define __CSRMATRIXGPU_HPP_

#include <cassert>
#include <vector>
#include <iostream>

#include "global.hpp"
#include "mpihandler.hpp"
#include "vectorgpu.hpp"

class CsrMatrixGpu
{
public:
    CsrMatrixGpu(const size_t, const size_t);
    CsrMatrixGpu(const size_t);
    CsrMatrixGpu(const CsrMatrixGpu&);
//    CsrMatrixGpu(CsrMatrixGpu&&);
    ~CsrMatrixGpu();
    CsrMatrixGpu operator=(const CsrMatrixGpu&);
//    CsrMatrixGpu operator=(const CsrMatrixGpu&&);

    void createStructure(const Triangle* const elements, const size_t num_elem);

    float get_local(const size_t, const size_t) const;
    void set_local(const size_t, const size_t, const float);
    void add_local(const size_t, const size_t, const float);
    void add_local_atomic(const size_t, const size_t, const float);
    float get_global(const size_t, const size_t) const;
    void set_global(const size_t, const size_t, const float);
    void add_global(const size_t, const size_t, const float);

    void multvec(const VectorGpu&, VectorGpu&) const;

    void print_local_data(const size_t firstindex=0) const;
//private:
    size_t _numrows, _numcols;
    size_t* _rowptr;
    size_t* _colind;
    float* _values;
};

#endif
