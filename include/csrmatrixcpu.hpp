#ifndef __CSRMATRIXCPU_HPP_
#define __CSRMATRIXCPU_HPP_

#include <cassert>
#include <vector>
#include <iostream>

#include "global.hpp"
#include "mpihandler.hpp"
#include "vectorcpu.hpp"

template <typename datatype>
class CsrMatrixCpu
{
public:
    CsrMatrixCpu(const size_t, const size_t);
    CsrMatrixCpu(const size_t);
    CsrMatrixCpu(const CsrMatrixCpu&);
//    CsrMatrixCpu(CsrMatrixCpu&&);
    ~CsrMatrixCpu();
//    CsrMatrixCpu operator=(const CsrMatrixCpu&);
//    CsrMatrixCpu operator=(const CsrMatrixCpu&&);

    void createStructure(const Triangle1* const elements, const size_t num_elem);

    datatype get(const size_t, const size_t) const;
    void set(const size_t, const size_t, const datatype);
    void add(const size_t, const size_t, const datatype);
    //double get_global(const size_t, const size_t) const;
    //void set_global(const size_t, const size_t, const double);
    //void add_global(const size_t, const size_t, const double);

    void multvec(const VectorCpu&, VectorCpu&) const;

    void print_data(const size_t firstindex=0);
    //void print_global_data(const size_t firstindex=0);
//private:
    size_t _numrows, _numcols;
    size_t* _rowptr;
    size_t* _colind;
    datatype* _values;
};

#include "csrmatrixcpu.tpp"

#endif
