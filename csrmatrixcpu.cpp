#include "include/csrmatrixcpu.hpp"

CsrMatrixCpu::CsrMatrixCpu(const size_t numrows, const size_t numcols):
    _numrows(numrows), _numcols(numcols),
    _colind(nullptr), _values(nullptr)
{
    _rowptr = new size_t[numrows+1];
}

CsrMatrixCpu::CsrMatrixCpu(const size_t size):
    _numrows(size), _numcols(size),
    _colind(nullptr), _values(nullptr)
{
    _rowptr = new size_t[size+1];
}

CsrMatrixCpu::~CsrMatrixCpu()
{
    delete[] _rowptr;
    delete[] _colind;
    delete[] _values;
}

void CsrMatrixCpu::createStructure(const Triangle* const elements, const size_t num_elem)
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

    size_t num_values(0);
    for (size_t i(0); i < _numrows; ++i)
    {
        _rowptr[i] = num_values;
        num_values += lol[i].size();
    }
    _rowptr[_numrows] = num_values;
    delete[] _colind;
    delete[] _values;
    _colind = new size_t[num_values];
    _values = new float[num_values];

    size_t current_pos(0);
    for (const auto& row : lol)
        for (const auto col : row)
            _colind[current_pos++] = col;
    for (size_t i(0); i < num_values; ++i)
        _values[i] = 0.0;


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
}

void CsrMatrixCpu::set_local(const size_t row, const size_t col, const float val)
{
    assert(row < _numrows && col < _numcols);
    size_t pos_to_insert(_rowptr[row]);
    while (_colind[pos_to_insert] < col && pos_to_insert < _rowptr[row+1])
        ++pos_to_insert;
    assert(_colind[pos_to_insert] == col && pos_to_insert < _rowptr[row+1]);
    _values[pos_to_insert] = val;
}

void CsrMatrixCpu::add_local(const size_t row, const size_t col, const float val)
{
    assert(row < _numrows && col < _numcols);
    size_t pos_to_insert(_rowptr[row]);
    while (_colind[pos_to_insert] < col && pos_to_insert < _rowptr[row+1])
        ++pos_to_insert;
    assert(_colind[pos_to_insert] == col && pos_to_insert < _rowptr[row+1]);
    _values[pos_to_insert] += val;
}

void CsrMatrixCpu::multvec(const VectorCpu& vec, VectorCpu& res) const
{
    assert(_numcols == vec._size && _numrows == res._size); //TODISCUSS or reallocate when res has a different size?
    for (size_t row{0}; row < _numrows; ++row)
    {
        res._values[row] = 0.0;
        for (size_t col{_rowptr[row]}; col < _rowptr[row+1]; ++col)
            res._values[row] += _values[col] * vec._values[_colind[col]];
    }
}

void CsrMatrixCpu::print_local_data(const size_t firstindex=0)
{
    for (size_t row(0), current_pos(0); row < _numrows; ++row)
    {
        std::cout << row+firstindex << ": ";
        for (size_t col(_rowptr[row]); col < _rowptr[row+1]; ++col, ++current_pos)
            std::cout << _values[current_pos] << "(" << _colind[current_pos]+firstindex << "), ";
        std::cout << std::endl;
    }
}
