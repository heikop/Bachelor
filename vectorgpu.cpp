#include "include/vectorgpu.hpp"

VectorGpu::VectorGpu(const size_t size):
    _size(size)
{
    malloc_cuda(&_values, _size*sizeof(float));
    //TODISCUSS: set to zero?
}

VectorGpu::VectorGpu(const VectorGpu& other):
    _size(other._size)
{
    malloc_cuda(&_values, _size*sizeof(float));
    memcpy_cuda(_values, other._values, _size*sizeof(float), d2d);
}

VectorGpu::VectorGpu(VectorGpu&& other):
    _size(other._size),
    _values(other._values)
{
    other._values = nullptr;
    other._size = 0;
}

VectorGpu::~VectorGpu()
{
    free_cuda(_values);
}

VectorGpu VectorGpu::operator=(const VectorGpu& other)
{
    free_cuda(_values);
    _size = other._size;
    malloc_cuda(&_values, _size*sizeof(float));
    memcpy_cuda(_values, other._values, _size*sizeof(float), d2d);
    return *this;
}

VectorGpu VectorGpu::operator=(VectorGpu&& other)
{
    free_cuda(_values);
    _size = other._size;
    _values = other._values;
    other._values = nullptr;
    other._size = 0;
    return *this;
}

void VectorGpu::set_local(const size_t pos, const float val)
{
    assert(pos < _size);
    //TODO
}

void VectorGpu::add_local(const size_t pos, const float val)
{
    assert(pos < _size);
    //TODO
}

//void VectorGpu::set_global(const size_t, const float); //TODO later
//void VectorGpu::add_global(const size_t, const float); //TODO later

//void VectorGpu::scal_mul(const float scalar)
//{
//    //TODO
//}
//
//void scal_mul_kernel(const float scalar)
//{
//}

float VectorGpu::dot_vec(const VectorGpu& other) const
{
    assert(_size == other._size);
    float res(0.0);
    //TODO
    return res;
}

void VectorGpu::print_local_data(const size_t firstindex=0) const
{
    float *h_values = new float[_size];
    memcpy_cuda(h_values, _values, _size*sizeof(float), d2h);
    for (size_t i(0); i < _size; ++i)
        std::cout << i + firstindex << ": " << h_values[i] << std::endl;
    delete[] h_values;
}
