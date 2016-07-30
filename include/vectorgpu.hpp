#ifndef __VECTORGPU_HPP_
#define __VECTORGPU_HPP_

#include <cassert>
#include <vector>
#include <iostream>

#include "global.hpp"
#include "mpihandler.hpp"

class VectorGpu
{
public:
    VectorGpu(const size_t);
    VectorGpu(const VectorGpu&);
    VectorGpu(VectorGpu&&);
    ~VectorGpu();
    VectorGpu operator=(const VectorGpu&);
    VectorGpu operator=(VectorGpu&&);

    void set_local(const size_t, const float);
    void add_local(const size_t, const float);
    void set_global(const size_t, const float);
    void add_global(const size_t, const float);

    void scal_mul(const float);
    float dot_vec(const VectorGpu&) const;

    void print_local_data(const size_t firstindex) const;
//private:
    size_t _size;
    float* _values;

    // pre kernel funcions
    //void scal_mul_kernel(const float scalar);
};

#endif
