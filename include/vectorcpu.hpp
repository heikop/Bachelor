#ifndef __VECTORCPU_HPP_
#define __VECTORCPU_HPP_

#include <cassert>
#include <vector>
#include <iostream>

#include "global.hpp"
#include "mpihandler.hpp"

class VectorCpu
{
public:
    VectorCpu(const size_t size, const double val=0.0);
    VectorCpu(const VectorCpu&);
//    VectorCpu(VectorCpu&&);
    ~VectorCpu();
    VectorCpu operator=(const VectorCpu&);
//    VectorCpu operator=(VectorCpu&&);
    void copy(const VectorCpu&);
    void copyscal(const double, const VectorCpu&);

    void set(const size_t, const double);
    void add(const size_t, const double);
    //void set_global(const size_t, const double);
    //void add_global(const size_t, const double);

    void scal(const double);
    void axpy(const double, const VectorCpu&);
    void axpby(const double, const VectorCpu&, const double);
    double dot_vec(const VectorCpu&) const;

    double l2norm2() const;

    void print_data(const size_t firstindex=0) const;
//private:
    size_t _size;
    double* _values;
};

#endif
