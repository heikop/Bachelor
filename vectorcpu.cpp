#include "include/vectorcpu.hpp"
#include <cstring>

VectorCpu::VectorCpu(const size_t size, const double value):
    _size(size)
{
    _values = new double[_size];
    for (size_t i{0}; i < _size; ++i) 
        _values[i] = value;
    //TODO error handling allocating
}

VectorCpu::VectorCpu(const VectorCpu& other):
    _size(other._size)
{
    _values = new double[_size];
    for (size_t i(0); i < _size; ++i)
        _values[i] = other._values[i];
}

//VectorCpu::VectorCpu(VectorCpu&& other):
//    _size(other._size),
//    _values(other._values)
//{
//    other._values = nullptr;
//    other._size = 0;
//}

VectorCpu::~VectorCpu()
{
    delete[] _values;
}

VectorCpu VectorCpu::operator=(const VectorCpu& other)
{
    delete[] _values;
    _size = other._size;
    _values = new double[_size];
    for (size_t i(0); i < _size; ++i)
        _values[i] = other._values[i];
    return *this;
}

//VectorCpu VectorCpu::operator=(VectorCpu&& other)
//{
//    delete[] _values;
//    _size = other._size;
//    _values = other._values;
//    other._values = nullptr;
//    other._size = 0;
//    return *this;
//}

void VectorCpu::copy(const VectorCpu& other)
{
    assert(_size == other._size);
    std::memcpy(_values, other._values, _size*sizeof(double));
}

void VectorCpu::copyscal(const double scal, const VectorCpu& other)
{
    assert(_size == other._size);
    if (scal == 0.0)
        for (size_t i{0}; i < _size; ++i)
            _values[i] = 0.0;
    else if (scal == 1.0)
        std::memcpy(_values, other._values, _size*sizeof(double));
    else
        for (size_t i{0}; i < _size; ++i)
            _values[i] = scal * other._values[i];
}

void VectorCpu::set(const size_t pos, const double val)
{
    assert(pos < _size);
    _values[pos] = val;
}

void VectorCpu::add(const size_t pos, const double val)
{
    assert(pos < _size);
    _values[pos] += val;
}

//void VectorCpu::set(const size_t, const double); //TODO later
//void VectorCpu::add(const size_t, const double); //TODO later

void VectorCpu::scal(const double scalar)
{
    if (scalar == 0.0)
        for (size_t i(0); i < _size; ++i)
            _values[i] = 0.0;
    else if (scalar != 1.0)
        for (size_t i(0); i < _size; ++i)
            _values[i] *= scalar;
}

double VectorCpu::dot_vec(const VectorCpu& other) const
{
    assert(_size == other._size);

    double res(0.0);
    for (size_t i(0); i < _size; ++i)
        res += _values[i] * other._values[i];

    return res;
}

void VectorCpu::axpy(const double a, const VectorCpu& x)
{
    assert(_size == x._size);

    if (a == 1.0)
        for (size_t i{0}; i < _size; ++i)
            _values[i] += x._values[i];
    else if (a != 0.0)
        for (size_t i{0}; i < _size; ++i)
            _values[i] += a * x._values[i];
}

void VectorCpu::axpby(const double a, const VectorCpu& x, const double b)
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

double VectorCpu::l2norm2() const
{
    double res{0.0};
    for (size_t i{0}; i < _size; ++i)
        res += _values[i] * _values[i];

    return res;
}

void VectorCpu::print_data(const size_t firstindex) const
{
    for (size_t i(0); i < _size; ++i)
        std::cout << i + firstindex << ": " << _values[i] << std::endl;
}
