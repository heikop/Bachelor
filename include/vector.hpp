#ifndef __VECTOR_HPP_
#define __VECTOR_HPP_

#include <cstddef> // size_t

class Vector
{
public:
    virtual ~Vector() {}
    Vector operator=(const Vector&);
    Vector operator=(Vector&&);
    void copy(const Vector&);
    void copyscal(const double, const Vector&);

    virtual void set_local(const size_t, const double);
    virtual void add_local(const size_t, const double);
    virtual void set_global(const size_t, const double);
    virtual void add_global(const size_t, const double);

    void scal(const double);
    void axpy(const double, const Vector&);
    void axpby(const double, const Vector&, const double);
    double dot_vec(const Vector&) const;

    double l2norm2() const;

    void print_local_data(const size_t firstindex) const;
};

#endif
