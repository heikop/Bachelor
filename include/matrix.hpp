#ifndef __MATRIX_HPP_
#define __MATRIX_HPP_

#include <cstddef> // size_t
#include "vector.hpp"

class Matrix
{
public:
    virtual ~Matrix() {}
    Matrix operator=(const Matrix&);
    Matrix operator=(const Matrix&&);

    //void createStructure(const Triangle* const elements, const size_t num_elem);

    void set_local(const size_t, const size_t, const double);
    void add_local(const size_t, const size_t, const double);
    void set_global(const size_t, const size_t, const double);
    void add_global(const size_t, const size_t, const double);

    void multvec(const Vector&, Vector&) const;

    void print_local_data(const size_t firstindex);
};

#endif
