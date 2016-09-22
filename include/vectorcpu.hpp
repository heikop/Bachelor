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
    VectorCpu(const size_t size, const float val=0.0);
    VectorCpu(const VectorCpu&);
//    VectorCpu(VectorCpu&&);
    ~VectorCpu();
    VectorCpu operator=(const VectorCpu&);
//    VectorCpu operator=(VectorCpu&&);
    void copy(const VectorCpu&);
    void copyscal(const float, const VectorCpu&);

    void set_local(const size_t, const float);
    void add_local(const size_t, const float);
    void set_global(const size_t, const float);
    void add_global(const size_t, const float);

    void scal(const float);
    void axpy(const float, const VectorCpu&);
    void axpby(const float, const VectorCpu&, const float);
    float dot_vec(const VectorCpu&) const;

    float l2norm2() const;

    void print_local_data(const size_t firstindex=0) const;
//private:
    size_t _size_global;
    size_t _size_local;
    size_t _firstentry_on_local;
    float* _values;
};

#endif
