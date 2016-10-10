#include "include/vectorcpu.hpp"
#include <cstring>

VectorCpu::VectorCpu(const size_t size, const float value):
    _size_global(size)
{
    _size_local = _size_global / __mpi_instance__.get_global_size();
    if (_size_global % __mpi_instance__.get_global_size() > __mpi_instance__.get_global_rank())
    {
        ++_size_local;
        _firstentry_on_local = _size_local * __mpi_instance__.get_global_rank();
    }
    else 
        _firstentry_on_local = _size_local * __mpi_instance__.get_global_rank() + _size_global % __mpi_instance__.get_global_size();

    _values = new float[_size_local];
    for (size_t i{0}; i < _size_local; ++i) 
        _values[i] = value;
    //TODO error handling allocating
}

VectorCpu::VectorCpu(const VectorCpu& other):
    _size_global(other._size_global), _size_local(other._size_local),
    _firstentry_on_local(other._firstentry_on_local)
{
    _values = new float[_size_local];
    for (size_t i(0); i < _size_local; ++i)
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
    _size_global = other._size_global;
    _size_local = other._size_local;
    _firstentry_on_local = other._firstentry_on_local;
    _values = new float[_size_local];
    for (size_t i(0); i < _size_local; ++i)
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
    assert(_size_global == other._size_global);
    assert(_size_local == other._size_local);
    assert(_firstentry_on_local == other._firstentry_on_local);
    std::memcpy(_values, other._values, _size_local*sizeof(float));
}

void VectorCpu::copyscal(const float scal, const VectorCpu& other)
{
    assert(_size_global == other._size_global);
    assert(_size_local == other._size_local);
    assert(_firstentry_on_local = other._firstentry_on_local);
    if (scal == 0.0)
        for (size_t i{0}; i < _size_local; ++i)
            _values[i] = 0.0;
    else if (scal == 1.0)
        std::memcpy(_values, other._values, _size_local*sizeof(float));
    else
        for (size_t i{0}; i < _size_local; ++i)
            _values[i] = scal * other._values[i];
}

void VectorCpu::set_local(const size_t pos, const float val)
{
    assert(pos < _size_local);
    _values[pos] = val;
}

void VectorCpu::add_local(const size_t pos, const float val)
{
    assert(pos < _size_local);
    _values[pos] += val;
}

//void VectorCpu::set_global(const size_t, const float); //TODO later
//void VectorCpu::add_global(const size_t, const float); //TODO later

void VectorCpu::scal(const float scalar)
{
    if (scalar == 0.0)
        for (size_t i(0); i < _size_local; ++i)
            _values[i] = 0.0;
    else if (scalar != 1.0)
        for (size_t i(0); i < _size_local; ++i)
            _values[i] *= scalar;
}

float VectorCpu::dot_vec(const VectorCpu& other) const
{
    assert(_size_global == other._size_global);
    assert(_size_local == other._size_local);
    assert(_firstentry_on_local == other._firstentry_on_local);

    float res_local(0.0);
    for (size_t i(0); i < _size_local; ++i)
        res_local += _values[i] * other._values[i];

    float res_global(0.0);
    MPICALL(MPI::COMM_WORLD.Allreduce(&res_local, &res_global, 1, MPI_FLOAT, MPI_SUM);)
    return res_global;
}

void VectorCpu::axpy(const float a, const VectorCpu& x)
{
    assert(_size_global == x._size_global);
    assert(_size_local == x._size_local);
    assert(_firstentry_on_local == x._firstentry_on_local);

    if (a == 1.0)
        for (size_t i{0}; i < _size_local; ++i)
            _values[i] += x._values[i];
    else if (a != 0.0)
        for (size_t i{0}; i < _size_local; ++i)
            _values[i] += a * x._values[i];
}

void VectorCpu::axpby(const float a, const VectorCpu& x, const float b)
{
    assert(_size_global == x._size_global);
    assert(_size_local == x._size_local);
    assert(_firstentry_on_local == x._firstentry_on_local);

    if (b == 1.0)
        axpy(a, x);
    else if (b == 0.0)
        copyscal(a, x);
    else if (a == 0.0)
        scal(b);
    else if (a == 1.0)
        for (size_t i{0}; i < _size_local; ++i)
            _values[i] = b*_values[i] + x._values[i];
    else
        for (size_t i{0}; i < _size_local; ++i)
            _values[i] = b*_values[i] + a*x._values[i];
}

float VectorCpu::l2norm2() const
{
    float res_local{0.0};
    for (size_t i{0}; i < _size_local; ++i)
        res_local += _values[i] * _values[i];

    float res_global(0.0);
    MPICALL(MPI::COMM_WORLD.Allreduce(&res_local, &res_global, 1, MPI_FLOAT, MPI_SUM);)
    return res_global;
}

void VectorCpu::print_local_data(const size_t firstindex) const
{
    for (size_t i(0); i < _size_local; ++i)
        std::cout << _firstentry_on_local + i + firstindex << ": " << _values[i] << std::endl;
}
