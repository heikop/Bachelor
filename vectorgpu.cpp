#include "include/vectorgpu.hpp"

VectorGpu::VectorGpu(const VectorGpu& other):
    _size(other._size)
{
    malloc_cuda(&_values, _size*sizeof(float));
    memcpy_cuda(_values, other._values, _size*sizeof(float), d2d);
}

/*
VectorGpu::VectorGpu(VectorGpu&& other):
    _size(other._size),
    _values(other._values)
{
    other._values = nullptr;
    other._size = 0;
}
*/

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

/*
VectorGpu VectorGpu::operator=(VectorGpu&& other)
{
    free_cuda(_values);
    _size = other._size;
    _values = other._values;
    other._values = nullptr;
    other._size = 0;
    return *this;
}
*/

void VectorGpu::copy(const VectorGpu& other)
{
    assert(_size == other._size);
    memcpy_cuda(this, &other, _size*sizeof(float), d2d);
}

void VectorGpu::print_local_data(const size_t firstindex=0) const
{
    float *h_values = new float[_size];
    memcpy_cuda(h_values, _values, _size*sizeof(float), d2h);
    for (size_t i(0); i < _size; ++i)
        std::cout << i + firstindex << ": " << h_values[i] << std::endl;
    delete[] h_values;
}
