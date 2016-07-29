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

CsrMatrixGpu::~CsrMatrixGpu()
{
    free_cuda(_rowptr);
    free_cuda(_colind);
    free_cuda(_values);
}

void CsrMatrixGpu::createStructure(const Triangle* const elements, const size_t num_elem)
{
    std::vector<std::vector<size_t>> lol(_numcols, std::vector<size_t>(0));
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
                size_t a(nodes[node1]);
                size_t b(nodes[node2]);
                size_t j(0);
                while (j < lol[a].size() && lol[a][j] < b)
                    ++j;
                if (j == lol[a].size())
                    lol[a].push_back(b);
                else if (lol[a][j] != b)
                    lol[a].insert(lol[a].begin()+j, b);
            }
        }
    }

    size_t* h_rowptr = new size_t[_numrows+1];
    size_t num_values(0);
    for (size_t i(0); i < _numrows; ++i)
    {
        h_rowptr[i] = num_values;
        num_values += lol[i].size();
    }
    h_rowptr[_numrows] = num_values;

    size_t* h_colind = new size_t[num_values];
    size_t current_pos(0);
    for (const auto& row : lol)
        for (const auto col : row)
            h_colind[current_pos++] = col;
    float* h_values = new float[num_values];
    for (size_t i(0); i < num_values; ++i)
        h_values[i] = 0.0;

    // test output
    /*
    for (size_t row(0); row < lol.size(); ++row)
    {
        std::cout << row << "(): ";
        for (size_t col(0); col < lol[row].size(); ++col)
        {
            std::cout << lol[row][col] << ", ";
        }
        std::cout << std::endl;
    }

    for (size_t i(0); i <= _numrows; ++i)
        std::cout << _rowptr[i] << ", ";
    std::cout << std::endl;
    for (size_t i(0); i < num_values; ++i)
        std::cout << _colind[i] << ", ";
    std::cout << std::endl;
    for (size_t i(0); i < num_values; ++i)
        std::cout << _values[i] << ", ";
    std::cout << std::endl;
    */

    free_cuda(_colind);
    free_cuda(_values);
    malloc_cuda(&_colind, num_values*sizeof(size_t));
    malloc_cuda(&_values, num_values*sizeof(float));
    memcpy_cuda(_rowptr, h_rowptr, (_numrows+1)*sizeof(size_t), h2d);
    memcpy_cuda(_colind, h_colind, num_values*sizeof(size_t), h2d);
    memcpy_cuda(_values, h_values, num_values*sizeof(float), h2d);
    delete[] h_rowptr;
    delete[] h_colind;
    delete[] h_values;
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

void CsrMatrixGpu::print_local_data(const size_t firstindex=0)
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
