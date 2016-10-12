#include "include/csrmatrixcpu.hpp"
#include <cstring>

CsrMatrixCpu::CsrMatrixCpu(const size_t numrows, const size_t numcols):
    _numrows_global(numrows), _numcols_global(numcols),
    _numcols_local(numcols),
    _colind(nullptr), _values(nullptr)
{
    _numrows_local = _numrows_global / __mpi_instance__.get_global_size();
    if (_numrows_global % __mpi_instance__.get_global_size() > __mpi_instance__.get_global_rank())
    {
        ++_numrows_local;
        _firstrow_on_local = _numrows_local * __mpi_instance__.get_global_rank();
    }
    else
        _firstrow_on_local = _numrows_local * __mpi_instance__.get_global_rank() + _numrows_global % __mpi_instance__.get_global_size();

    _rowptr = new size_t[_numrows_local + 1];
    for (size_t i{0}; i <= _numrows_local; ++i)
        _rowptr[i] = 0;
}

CsrMatrixCpu::CsrMatrixCpu(const size_t size):
    _numrows_global(size), _numcols_global(size),
    _numcols_local(size),
    _colind(nullptr), _values(nullptr)
{
    _numrows_local = _numrows_global / __mpi_instance__.get_global_size();
    if (_numrows_global % __mpi_instance__.get_global_size() > __mpi_instance__.get_global_rank())
    {
        ++_numrows_local;
        _firstrow_on_local = _numrows_local * __mpi_instance__.get_global_rank();
    }
    else
        _firstrow_on_local = _numrows_local * __mpi_instance__.get_global_rank() + _numrows_global % __mpi_instance__.get_global_size();

    _rowptr = new size_t[_numrows_local + 1];
    for (size_t i{0}; i <= _numrows_local; ++i)
        _rowptr[i] = 0;
}

CsrMatrixCpu::CsrMatrixCpu(const CsrMatrixCpu& other):
    _numrows_global(other._numrows_global), _numcols_global(other._numcols_global),
    _numrows_local(other._numrows_local), _numcols_local(other._numcols_local)
{
    _rowptr = new size_t[_numrows_local + 1];
    for (size_t i(0); i <= _numrows_local; ++i)
        _rowptr[i] = other._rowptr[i];
    _colind = new size_t[_rowptr[_numrows_local]];
    _values = new double[_rowptr[_numrows_local]];
    for (size_t i(0); i < _rowptr[_numrows_local]; ++i)
    {
        _colind[i] = other._colind[i];
        _values[i] = other._values[i];
    }
}

CsrMatrixCpu::~CsrMatrixCpu()
{
    delete[] _rowptr;
    delete[] _colind;
    delete[] _values;
}

void CsrMatrixCpu::createStructure(const Triangle* const elements, const size_t num_elem)
{
    const size_t max_rowlength(20);

    size_t* num_nonzeros = new size_t[_numrows_local];
    for (size_t i(0); i < _numrows_local; ++i)
        num_nonzeros[i] = 0;

    size_t* colind = new size_t[max_rowlength*_numrows_local];

    for (size_t i(0); i < num_elem; ++i)
    {
        size_t nodes[3];
        nodes[0] = elements[i].nodeA;
        nodes[1] = elements[i].nodeB;
        nodes[2] = elements[i].nodeC;
        for (size_t node1(0); node1 < 3; ++node1)
        {
            for (size_t node2(0); node2 < 3; ++node2)
            {
                int a(nodes[node1] - _firstrow_on_local);
                size_t b(nodes[node2]);
                if (a >= 0 && static_cast<size_t>(a) < _numrows_local)
                {
                    size_t j(0);
                    while (j < num_nonzeros[a] && colind[a*max_rowlength + j] != b )
                        ++j;
                    if (num_nonzeros[a] == j)
                    {
                        ++(num_nonzeros[a]);
                        assert(num_nonzeros[a] <= max_rowlength);
                        colind[a*max_rowlength + j] = b;
                    }
                }
            }
        }
    }

    for (size_t i(0); i < _numrows_local; ++i)
        for (size_t a(num_nonzeros[i]-1); a > 0; --a)
            for (size_t b(0); b < a; ++b)
                if (colind[i*max_rowlength + b] > colind[i*max_rowlength + b+1])
                {
                    size_t tmp(colind[i*max_rowlength + b]);
                    colind[i*max_rowlength + b] = colind[i*max_rowlength + b+1];
                    colind[i*max_rowlength + b+1] = tmp;
                }

    size_t num_values{0};
    for (size_t i{0}; i < _numrows_local; ++i)
    {
        _rowptr[i] = num_values;
        num_values += num_nonzeros[i];
    }
    _rowptr[_numrows_local] = num_values;
    delete[] _colind;
    delete[] _values;
    _colind = new size_t[num_values];
    _values = new double[num_values];

    size_t current_pos{0};
    for (size_t row{0}; row < _numrows_local; ++row)
        for (size_t col{0}; col < num_nonzeros[row]; ++col)
            _colind[current_pos++] = colind[row*max_rowlength + col];
    for (size_t i{0}; i < num_values; ++i)
        _values[i] = 0.0;

    delete[] num_nonzeros;
    delete[] colind;
}

double CsrMatrixCpu::get_local(const size_t row, const size_t col) const
{
    assert(row < _numrows_local && col < _numcols_local);
    size_t pos_to_get(_rowptr[row]);
    while (_colind[pos_to_get] < col && pos_to_get < _rowptr[row+1])
        ++pos_to_get;
    return (_colind[pos_to_get] == col ? _values[pos_to_get] : 0.0);
}

void CsrMatrixCpu::set_local(const size_t row, const size_t col, const double val)
{
    assert(row < _numrows_local && col < _numcols_local);
    size_t pos_to_insert(_rowptr[row]);
    while (_colind[pos_to_insert] < col && pos_to_insert < _rowptr[row+1])
        ++pos_to_insert;
    assert(_colind[pos_to_insert] == col && pos_to_insert < _rowptr[row+1]);
    _values[pos_to_insert] = val;
}

void CsrMatrixCpu::add_local(const size_t row, const size_t col, const double val)
{
    assert(row < _numrows_local && col < _numcols_local);
    size_t pos_to_insert(_rowptr[row]);
    while (_colind[pos_to_insert] < col && pos_to_insert < _rowptr[row+1])
        ++pos_to_insert;
    assert(_colind[pos_to_insert] == col);// && pos_to_insert < _rowptr[row+1]);
    _values[pos_to_insert] += val;
}

double CsrMatrixCpu::get_global(const size_t row, const size_t col) const
{
    assert(row < _numrows_global && col < _numcols_global);
    double val{0.0};
    if (row >= _firstrow_on_local && row < _firstrow_on_local + _numrows_local)
        val = get_local(row - _firstrow_on_local, col);
    double val_global{0.0};
    MPICALL(MPI::COMM_WORLD.Allreduce(&val, &val_global, 1, MPI_DOUBLE, MPI_SUM);) //TODO should work with copying and not adding it up!
    //MPICALL(MPI::COMM_WORLD.Bcast(&val, 1, MPI_DOUBLE, __mpi_instance__.get_global_rank());) // somehow like this, I think
    return val_global;
}

void CsrMatrixCpu::set_global(const size_t row, const size_t col, const double val)
{
    assert(row < _numrows_global && col < _numcols_global);
    if (row >= _firstrow_on_local && row < _firstrow_on_local + _numrows_local)
        set_local(row - _firstrow_on_local, col, val);
    MPICALL(MPI::COMM_WORLD.Barrier();)
}

void CsrMatrixCpu::add_global(const size_t row, const size_t col, const double val)
{
    assert(row < _numrows_global && col < _numcols_global);
    if (row >= _firstrow_on_local && row < _firstrow_on_local + _numrows_local)
        add_local(row - _firstrow_on_local, col, val);
    MPICALL(MPI::COMM_WORLD.Barrier();)
}

void CsrMatrixCpu::multvec(const VectorCpu& vec, VectorCpu& res) const
{
    assert(_numcols_global == vec._size_global && _numrows_global == res._size_global); //TODISCUSS or reallocate when res has a different size?
    assert(_numcols_local == vec._size_local && _numrows_local == res._size_local);     //TODISCUSS or reallocate when res has a different size?
    for (size_t row{0}; row < _numrows_local; ++row)
    {
        res._values[row] = 0.0;
        for (size_t col{_rowptr[row]}; col < _rowptr[row+1]; ++col)
            res._values[row] += _values[col] * vec._values[_colind[col]];
    }
}

void CsrMatrixCpu::print_local_data(const size_t firstindex)
{
    for (size_t row(0), current_pos(0); row < _numrows_local; ++row)
    {
        std::cout << _firstrow_on_local + row + firstindex << ": ";
        for (size_t col(_rowptr[row]); col < _rowptr[row+1]; ++col, ++current_pos)
            std::cout << _values[current_pos] << "(" << _colind[current_pos] + firstindex << "), ";
        std::cout << std::endl;
    }
}

void CsrMatrixCpu::print_global_data(const size_t firstindex)
{
    for (size_t rank(0); rank < __mpi_instance__.get_global_size(); ++rank)
    {
        if (__mpi_instance__.get_global_rank() == rank)
            for (size_t row(0), current_pos(0); row < _numrows_local; ++row)
            {
                std::cout << _firstrow_on_local + row + firstindex << ": ";
                for (size_t col(_rowptr[row]); col < _rowptr[row+1]; ++col, ++current_pos)
                    std::cout << _values[current_pos] << "(" << _colind[current_pos] + firstindex << "), ";
                std::cout << std::endl;
            }
        MPICALL(MPI::COMM_WORLD.Barrier();)
    }
}
