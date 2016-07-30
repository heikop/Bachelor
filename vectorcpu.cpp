#include "include/vectorcpu.hpp"

VectorCpu::VectorCpu(const size_t size):
    _size(size)
{
    _values = new float[_size];
    for (size_t i(0); i < _size; ++i)
        _values[i] = 0.0;
}

VectorCpu::VectorCpu(const VectorCpu& other):
    _size(other._size)
{
    _values = new float[_size];
    for (size_t i(0); i < _size; ++i)
        _values[i] = other._values[i];
}

VectorCpu::VectorCpu(VectorCpu&& other):
    _size(other._size),
    _values(other._values)
{
    other._values = nullptr;
    other._size = 0;
}

VectorCpu::~VectorCpu()
{
    delete[] _values;
}

VectorCpu VectorCpu::operator=(const VectorCpu& other)
{
    delete[] _values;
    _size = other._size;
    _values = new float[_size];
    for (size_t i(0); i < _size; ++i)
        _values[i] = other._values[i];
    return *this;
}

VectorCpu VectorCpu::operator=(VectorCpu&& other)
{
    delete[] _values;
    _size = other._size;
    _values = other._values;
    other._values = nullptr;
    other._size = 0;
    return *this;
}

void VectorCpu::set_local(const size_t pos, const float val)
{
    assert(pos < _size);
    _values[pos] = val;
}

void VectorCpu::add_local(const size_t pos, const float val)
{
    assert(pos < _size);
    _values[pos] += val;
}

//void VectorCpu::set_global(const size_t, const float); //TODO later
//void VectorCpu::add_global(const size_t, const float); //TODO later

void VectorCpu::scal_mul(const float scalar)
{
    if (scalar == 0.0)
    {
        for (size_t i(0); i < _size; ++i)
            _values[i] = 0.0;
    }
    else if (scalar != 1.0)
    {
        for (size_t i(0); i < _size; ++i)
            _values[i] *= scalar;
    }
}

float VectorCpu::dot_vec(const VectorCpu& other) const
{
    assert(_size == other._size);
    float res(0.0);
    if (this == &other)
    {
        for (size_t i(0); i < _size; ++i)
            res += _values[i]*_values[i];
    }
    else
    {
        for (size_t i(0); i < _size; ++i)
            res += _values[i]*other._values[i];
    }
    return res;
}

void VectorCpu::print_local_data(const size_t firstindex=0) const
{
    for (size_t i(0); i < _size; ++i)
        std::cout << i + firstindex << ": " << _values[i] << std::endl;
}
