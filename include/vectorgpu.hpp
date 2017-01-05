#ifndef __VECTORGPU_HPP_
#define __VECTORGPU_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cassert>
#include <vector>
#include <iostream>

#include "global.hpp"
#include "mpihandler.hpp"
//#include "csrmatrixgpu.hpp"

class VectorGpu
{
public:
    VectorGpu(const size_t size, const float val=0.0);
    VectorGpu(const VectorGpu&);
//    VectorGpu(VectorGpu&&);
    ~VectorGpu();
    VectorGpu operator=(const VectorGpu&);
//    VectorGpu operator=(VectorGpu&&);
    void copy(const VectorGpu&);
    void copyscal(const float, const VectorGpu&);

    void set_local(const size_t, const float);
    void add_local(const size_t, const float);
    void set_global(const size_t, const float);
    void add_global(const size_t, const float);

    void scal(const float);
    void axpy(const float, const VectorGpu&);
    void axpby(const float, const VectorGpu&, const float);
    float dot_vec(const VectorGpu&) const;

    float l2norm2() const;

    void print_local_data(const size_t firstindex) const;
//private:
    size_t _size;
    float* _values;
};

#endif
