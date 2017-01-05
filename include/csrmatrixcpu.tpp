#include <cstring>

template <typename datatype>
CsrMatrixCpu<datatype>::CsrMatrixCpu(const size_t numrows, const size_t numcols):
    _numrows(numrows), _numcols(numcols),
    _colind(nullptr), _values(nullptr)
{
    _rowptr = new size_t[_numrows + 1];
    for (size_t i{0}; i <= _numrows; ++i)
        _rowptr[i] = 0;
}

template <typename datatype>
CsrMatrixCpu<datatype>::CsrMatrixCpu(const size_t size):
    _numrows(size), _numcols(size),
    _colind(nullptr), _values(nullptr)
{
    _rowptr = new size_t[_numrows + 1];
    for (size_t i{0}; i <= _numrows; ++i)
        _rowptr[i] = 0;
}

template <typename datatype>
CsrMatrixCpu<datatype>::CsrMatrixCpu(const CsrMatrixCpu& other):
    _numrows(other._numrows), _numcols(other._numcols)
{
    _rowptr = new size_t[_numrows + 1];
    for (size_t i(0); i <= _numrows; ++i)
        _rowptr[i] = other._rowptr[i];
    _colind = new size_t[_rowptr[_numrows]];
    _values = new datatype[_rowptr[_numrows]];
    for (size_t i(0); i < _rowptr[_numrows]; ++i)
    {
        _colind[i] = other._colind[i];
        _values[i] = other._values[i];
    }
}

template <typename datatype>
CsrMatrixCpu<datatype>::~CsrMatrixCpu()
{
    delete[] _rowptr;
    delete[] _colind;
    delete[] _values;
}

template <typename datatype>
void CsrMatrixCpu<datatype>::createStructure(const Triangle1* const elements, const size_t num_elem)
{
    const size_t max_rowlength(20);

    size_t* num_nonzeros = new size_t[_numrows];
    for (size_t i(0); i < _numrows; ++i)
        num_nonzeros[i] = 0;

    size_t* colind = new size_t[max_rowlength*_numrows];

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
                size_t a{nodes[node1]};
                size_t b{nodes[node2]};
                if (a >= 0 && static_cast<size_t>(a) < _numrows)
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

    for (size_t i(0); i < _numrows; ++i)
        for (size_t a(num_nonzeros[i]-1); a > 0; --a)
            for (size_t b(0); b < a; ++b)
                if (colind[i*max_rowlength + b] > colind[i*max_rowlength + b+1])
                {
                    size_t tmp(colind[i*max_rowlength + b]);
                    colind[i*max_rowlength + b] = colind[i*max_rowlength + b+1];
                    colind[i*max_rowlength + b+1] = tmp;
                }

    size_t num_values{0};
    for (size_t i{0}; i < _numrows; ++i)
    {
        _rowptr[i] = num_values;
        num_values += num_nonzeros[i];
    }
    _rowptr[_numrows] = num_values;
    delete[] _colind;
    delete[] _values;
    _colind = new size_t[num_values];
    _values = new datatype[num_values];

    size_t current_pos{0};
    for (size_t row{0}; row < _numrows; ++row)
        for (size_t col{0}; col < num_nonzeros[row]; ++col)
            _colind[current_pos++] = colind[row*max_rowlength + col];
    for (size_t i{0}; i < num_values; ++i)
        _values[i] = 0.0;

    delete[] num_nonzeros;
    delete[] colind;
}

template <typename datatype>
datatype CsrMatrixCpu<datatype>::get(const size_t row, const size_t col) const
{
    assert(row < _numrows && col < _numcols);
    size_t pos_to_get(_rowptr[row]);
    while (_colind[pos_to_get] < col && pos_to_get < _rowptr[row+1])
        ++pos_to_get;
    return (_colind[pos_to_get] == col ? _values[pos_to_get] : 0.0);
}

template <typename datatype>
void CsrMatrixCpu<datatype>::set(const size_t row, const size_t col, const datatype val)
{
    assert(row < _numrows && col < _numcols);
    size_t pos_to_insert(_rowptr[row]);
    while (_colind[pos_to_insert] < col && pos_to_insert < _rowptr[row+1])
        ++pos_to_insert;
    assert(_colind[pos_to_insert] == col && pos_to_insert < _rowptr[row+1]);
    _values[pos_to_insert] = val;
}

template <typename datatype>
void CsrMatrixCpu<datatype>::add(const size_t row, const size_t col, const datatype val)
{
    assert(row < _numrows && col < _numcols);
    size_t pos_to_insert(_rowptr[row]);
    while (_colind[pos_to_insert] < col && pos_to_insert < _rowptr[row+1])
        ++pos_to_insert;
    assert(_colind[pos_to_insert] == col);// && pos_to_insert < _rowptr[row+1]);

    #pragma omp atomic
    _values[pos_to_insert] += val;
}

//template <typename datatype>
//datatype CsrMatrixCpu<datatype>::get(const size_t row, const size_t col) const
//{
//    assert(row < _numrows && col < _numcols);
//    datatype val{0.0};
//    if (row >= _firstrow_on && row < _firstrow_on + _numrows)
//        val = get(row - _firstrow_on, col);
//    datatype val{0.0};
//    //MPICALL(MPI::COMM_WORLD.Allreduce(&val, &val, 1, MPI_DOUBLE, MPI_SUM);) //TODO should work with copying and not adding it up!
//    //MPICALL(MPI::COMM_WORLD.Bcast(&val, 1, MPI_DOUBLE, __mpi_instance__.get_rank());) // somehow like this, I think
//    return val;
//}
//
//template <typename datatype>
//void CsrMatrixCpu<datatype>::set(const size_t row, const size_t col, const datatype val)
//{
//    assert(row < _numrows && col < _numcols);
//    if (row >= _firstrow_on && row < _firstrow_on + _numrows)
//        set(row - _firstrow_on, col, val);
//    //MPICALL(MPI::COMM_WORLD.Barrier();)
//}
//
//template <typename datatype>
//void CsrMatrixCpu<datatype>::add(const size_t row, const size_t col, const datatype val)
//{
//    assert(row < _numrows && col < _numcols);
//    if (row >= _firstrow_on && row < _firstrow_on + _numrows)
//        add(row - _firstrow_on, col, val);
//    //MPICALL(MPI::COMM_WORLD.Barrier();)
//}

template <typename datatype>
void CsrMatrixCpu<datatype>::multvec(const VectorCpu& vec, VectorCpu& res) const
{
    assert(_numcols == vec._size && _numrows == res._size); //TODISCUSS or reallocate when res has a different size?
    assert(_numcols == vec._size && _numrows == res._size);     //TODISCUSS or reallocate when res has a different size?
    for (size_t row{0}; row < _numrows; ++row)
    {
        res._values[row] = 0.0;
        for (size_t col{_rowptr[row]}; col < _rowptr[row+1]; ++col)
            res._values[row] += _values[col] * vec._values[_colind[col]];
    }
}

template <typename datatype>
void CsrMatrixCpu<datatype>::print_data(const size_t firstindex)
{
    for (size_t row(0), current_pos(0); row < _numrows; ++row)
    {
        std::cout << row + firstindex << ": ";
        for (size_t col(_rowptr[row]); col < _rowptr[row+1]; ++col, ++current_pos)
            std::cout << _values[current_pos] << "(" << _colind[current_pos] + firstindex << "), ";
        std::cout << std::endl;
    }
}

//template <typename datatype>
//void CsrMatrixCpu<datatype>::print_data(const size_t firstindex)
//{
//    //for (size_t rank(0); rank < __mpi_instance__.get_size(); ++rank)
//    //{
//    //    if (__mpi_instance__.get_rank() == rank)
//            for (size_t row(0), current_pos(0); row < _numrows; ++row)
//            {
//                std::cout << _firstrow_on + row + firstindex << ": ";
//                for (size_t col(_rowptr[row]); col < _rowptr[row+1]; ++col, ++current_pos)
//                    std::cout << _values[current_pos] << "(" << _colind[current_pos] + firstindex << "), ";
//                std::cout << std::endl;
//            }
//    //    //MPICALL(MPI::COMM_WORLD.Barrier();)
//    //}
//}
