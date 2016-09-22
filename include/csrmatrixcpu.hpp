#ifndef __CSRMATRIXCPU_HPP_
#define __CSRMATRIXCPU_HPP_

#include <cassert>
#include <vector>
#include <iostream>

#include "global.hpp"
#include "mpihandler.hpp"
#include "vectorcpu.hpp"

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

    float get_local(const size_t, const size_t) const;
    void set_local(const size_t, const size_t, const float);
    void add_local(const size_t, const size_t, const float);
    float get_global(const size_t, const size_t) const;
    void set_global(const size_t, const size_t, const float);
    void add_global(const size_t, const size_t, const float);

    void multvec(const VectorCpu&, VectorCpu&) const;

    void print_local_data(const size_t firstindex=0);
    void print_global_data(const size_t firstindex=0);
//private:
    size_t _numrows_global, _numcols_global;
    size_t _numrows_local, _numcols_local;
    size_t _firstrow_on_local;
    //size_t* _rowptr_global;
    //size_t* _colind_global;
    //float* _values_global;
    size_t* _rowptr;
    size_t* _colind;
    float* _values;
};

#endif
