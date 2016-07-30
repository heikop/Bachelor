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
    void copyscal(const float, const Vector&);

    virtual void set_local(const size_t, const float);
    virtual void add_local(const size_t, const float);
    virtual void set_global(const size_t, const float);
    virtual void add_global(const size_t, const float);

    void scal(const float);
    void axpy(const float, const Vector&);
    void axpby(const float, const Vector&, const float);
    float dot_vec(const Vector&) const;

    float l2norm2() const;

    void print_local_data(const size_t firstindex) const;
};

#endif
