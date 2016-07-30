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
    VectorCpu(const size_t);
    VectorCpu(const VectorCpu&);
    VectorCpu(VectorCpu&&);
    ~VectorCpu();
    VectorCpu operator=(const VectorCpu&);
    VectorCpu operator=(VectorCpu&&);

    void set_local(const size_t, const float);
    void add_local(const size_t, const float);
    void set_global(const size_t, const float);
    void add_global(const size_t, const float);

    void scal_mul(const float);
    float dot_vec(const VectorCpu&) const;

    void print_local_data(const size_t firstindex) const;
//private:
    size_t _size;
    float* _values;
};

#endif
