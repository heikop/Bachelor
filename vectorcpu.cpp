#include "include/vectorcpu.hpp"
#include <cstring>

VectorCpu::VectorCpu(const size_t size, const float value):
    _size(size)
{
    _values = new float[_size];
    for (size_t i(0); i < _size; ++i)
        _values[i] = value;
    // TODISCUSS is the following faster for value == 0.0
    //if (value == 0.0)
    //    for (size_t i(0); i < _size; ++i)
    //        _values[i] = 0.0;
    //else
    //    for (size_t i(0); i < _size; ++i)
    //        _values[i] = value;
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

void VectorCpu::copy(const VectorCpu& other)
{
    assert(_size == other._size);
    for (size_t i{0}; i < _size; ++i)
        _values[i] = other._values[i];
}

void VectorCpu::copyscal(const float scal, const VectorCpu& other)
{
    assert(_size == other._size);
    if (scal == 0.0)
        for (size_t i{0}; i < _size; ++i)
            _values[i] = 0.0;
    else if (scal == 1.0)
        std::memcpy(_values, other._values, _size*sizeof(float));
    else
        for (size_t i{0}; i < _size; ++i)
            _values[i] = scal * other._values[i];
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

void VectorCpu::scal(const float scalar)
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
    for (size_t i(0); i < _size; ++i)
        res += _values[i]*other._values[i];
    return res;
}

void VectorCpu::axpy(const float a, const VectorCpu& x)
{
    assert(_size == x._size);
    if (a == 1.0)
        for (size_t i{0}; i < _size; ++i)
            _values[i] += x._values[i];
    else if (a != 0.0)
        for (size_t i{0}; i < _size; ++i)
            _values[i] += a * x._values[i];
}

void VectorCpu::axpby(const float a, const VectorCpu& x, const float b)
{
    assert(_size == x._size);
    if (b == 1.0)
        axpy(a, x);
    else if (b == 0.0)
        copyscal(a, x);
    else if (a == 0.0)
        scal(b);
    else if (a == 1.0)
        for (size_t i{0}; i < _size; ++i)
            _values[i] = b*_values[i] + x._values[i];
    else
        for (size_t i{0}; i < _size; ++i)
            _values[i] = b*_values[i] + a*x._values[i];
}

float VectorCpu::l2norm2() const
{
    float res{0.0};
    for (size_t i{0}; i < _size; ++i)
        res += _values[i] * _values[i];
    return res;
}

void VectorCpu::print_local_data(const size_t firstindex=0) const
{
    for (size_t i(0); i < _size; ++i)
        std::cout << i + firstindex << ": " << _values[i] << std::endl;
}
