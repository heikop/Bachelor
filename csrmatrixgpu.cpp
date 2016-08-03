#include "include/csrmatrixgpu.hpp"

CsrMatrixGpu::CsrMatrixGpu(const size_t numrows, const size_t numcols):
    _numrows(numrows), _numcols(numcols),
    _colind(nullptr), _values(nullptr)
{
    malloc_cuda(&_rowptr, (numrows+1)*sizeof(size_t));
}

CsrMatrixGpu::CsrMatrixGpu(const size_t size):
    _numrows(size), _numcols(size),
    _colind(nullptr), _values(nullptr)
{
    malloc_cuda(&_rowptr, (size+1)*sizeof(size_t));
}

CsrMatrixGpu::CsrMatrixGpu(const CsrMatrixGpu& other):
    _numrows(other._numrows), _numcols(other._numcols)
{
    size_t numvalues{0};
    memcpy_cuda(&numvalues, other._rowptr+_numrows, sizeof(size_t), d2h);

    malloc_cuda(&_rowptr, (_numrows+1)*sizeof(size_t));
    memcpy_cuda(_rowptr, other._rowptr, (_numrows+1)*sizeof(size_t), d2d);
    malloc_cuda(&_colind, numvalues*sizeof(size_t));
    memcpy_cuda(_colind, other._colind, numvalues*sizeof(size_t), d2d);
    malloc_cuda(&_values, numvalues*sizeof(float));
    memcpy_cuda(_values, other._values, numvalues*sizeof(float), d2d);
}

CsrMatrixGpu::CsrMatrixGpu(CsrMatrixGpu&& other):
    _numrows(other._numrows), _numcols(other._numcols)
{
std::cout << "copy and delete" << std::endl;
    _rowptr = other._rowptr;
    _colind = other._colind;
    _values = other._values;

    other._numrows = 0;
    other._numcols = 0;
    other._rowptr = nullptr;
    other._colind = nullptr;
    other._values = nullptr;
}

CsrMatrixGpu::~CsrMatrixGpu()
{
    free_cuda(_rowptr);
    free_cuda(_colind);
    free_cuda(_values);
}


void CsrMatrixGpu::set_local(const size_t row, const size_t col, const float val)
{
    assert(row < _numrows && col < _numcols);
    size_t pos_to_insert(_rowptr[row]);
    while (_colind[pos_to_insert] < col && pos_to_insert < _rowptr[row+1])
        ++pos_to_insert;
    assert(_colind[pos_to_insert] == col && pos_to_insert < _rowptr[row+1]);
    _values[pos_to_insert] = val;
}

void CsrMatrixGpu::add_local(const size_t row, const size_t col, const float val)
{
    assert(row < _numrows && col < _numcols);
    size_t pos_to_insert(_rowptr[row]);
    while (_colind[pos_to_insert] < col && pos_to_insert < _rowptr[row+1])
        ++pos_to_insert;
    assert(_colind[pos_to_insert] == col && pos_to_insert < _rowptr[row+1]);
    _values[pos_to_insert] += val;
}

void CsrMatrixGpu::add_local_atomic(const size_t row, const size_t col, const float val)
{
    assert(row < _numrows && col < _numcols);
    size_t pos_to_insert(_rowptr[row]);
    while (_colind[pos_to_insert] < col && pos_to_insert < _rowptr[row+1])
        ++pos_to_insert;
    assert(_colind[pos_to_insert] == col && pos_to_insert < _rowptr[row+1]);
    _values[pos_to_insert] += val;
    //atomicAdd(&_values[pos_to_insert], val);
}

void CsrMatrixGpu::print_local_data(const size_t firstindex=0) const
{
    size_t* h_rowptr = new size_t[_numrows+1];
    memcpy_cuda(h_rowptr, _rowptr, (_numrows+1)*sizeof(size_t), d2h);
    size_t* h_colind = new size_t[h_rowptr[_numrows]];
    float* h_values = new float[h_rowptr[_numrows]];
    memcpy_cuda(h_colind, _colind, h_rowptr[_numrows]*sizeof(size_t), d2h);
    memcpy_cuda(h_values, _values, h_rowptr[_numrows]*sizeof(float), d2h);

    for (size_t row(0), current_pos(0); row < _numrows; ++row)
    {
        std::cout << row+firstindex << ": ";
        for (size_t col(h_rowptr[row]); col < h_rowptr[row+1]; ++col, ++current_pos)
            std::cout << h_values[current_pos] << "(" << h_colind[current_pos]+firstindex << "), ";
        std::cout << std::endl;
    }
    delete[] h_rowptr;
    delete[] h_colind;
    delete[] h_values;
}
