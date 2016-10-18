#ifndef __ELEMENTS_HPP_
#define __ELEMENTS_HPP_

#include <cassert>
#include <array>
#include <cmath>

//TODO TOREMOVE
template<typename datatype>
struct Vertex { size_t id; datatype x; datatype y; };
// all elements are in a 2-dimensional room

template<typename datatype>
class Element
{
protected:
    Element(const unsigned int dim,
            const unsigned int deg,
            const unsigned int numbf):
        _dimension{dim}, _degree{deg}, _num_basis_functions{numbf} {}
public:
    unsigned int dimension() const { return _dimension; }
    unsigned int degree() const { return _degree; }
    unsigned int num_basis_functions() const { return _num_basis_functions; }

protected:
    const unsigned int _dimension;
    const unsigned int _degree;
    const unsigned int _num_basis_functions; // num basis functions
};


// ***** // ***** // ***** // ***** 1D ***** // ***** // ***** // ***** //

template<typename datatype>
class Element1D : public Element<datatype>
{
protected:
    Element1D(const unsigned int deg,
              const unsigned int numbf):
        Element<datatype>(1, deg, numbf) {}

public:
    virtual datatype evaluate_ref(const unsigned int basis_function,
                                  const datatype x) = 0;
    virtual datatype derivate_ref(const unsigned int basis_function,
                                  const unsigned int direction,
                                  const datatype x) = 0;
};

// ***** // ***** Line ***** // ***** //

template<typename datatype>
class Line : public Element1D<datatype>
{
protected:
    Line(const unsigned int deg,
         const unsigned int numbf,
         const Vertex<datatype> p0,
         const Vertex<datatype> p1):
        Element1D<datatype>(deg, numbf), _p0{p0}, _p1{p1} {}

public:
    //const std::array<std::array<datatype, 2>, 2> transformation_matrix() const; //TODO

protected:
    const Vertex<datatype> _p0, _p1;
};

template<typename datatype>
class LineQ1 : public Line<datatype>
{
public:
    LineQ1(const Vertex<datatype> p0,
           const Vertex<datatype> p1):
        Line<datatype>(1, 2, p0, p1) {}

    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x);
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x);
};

template<typename datatype>
class LineQ2 : public Line<datatype>
{
public:
    LineQ2(const Vertex<datatype> p0,
           const Vertex<datatype> p1,
           const size_t p2_id):
        Line<datatype>(1, 2, p0, p1), _p2_id{p2_id} {}

    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x);
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x);
private:
    const size_t _p2_id;
};

// ***** // ***** // ***** // ***** 2D ***** // ***** // ***** // ***** //

template<typename datatype>
class Element2D : public Element<datatype>
{
protected:
    Element2D(const unsigned int deg,
              const unsigned int numbf):
        Element<datatype>(2, deg, numbf) {}

public:
    virtual datatype evaluate_ref(const unsigned int basis_function,
                                  const datatype x,
                                  const datatype y) = 0;
    virtual datatype derivate_ref(const unsigned int basis_function,
                                  const unsigned int direction,
                                  const datatype x,
                                  const datatype y) = 0;
    virtual std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                                 const datatype x,
                                                 const datatype y) = 0;
};

// ***** // ***** Triangle ***** // ***** //

template<typename datatype>
class Triangle : public Element2D<datatype>
{
//protected:
public:
    Triangle(const unsigned int deg,
             const unsigned int numbf,
             const Vertex<datatype> p0,
             const Vertex<datatype> p1,
             const Vertex<datatype> p2):
        Element2D<datatype>(deg, numbf),
        _p0{p0}, _p1{p1}, _p2{p2} {}

public:
    const std::array<std::array<datatype, 2>, 2> transformation_matrix() const;

//protected:
public:
    const Vertex<datatype> _p0, _p1, _p2;
};

template<typename datatype>
class TriangleQ1 : public Triangle<datatype>
{
public:
    TriangleQ1(const Vertex<datatype> p0,
               const Vertex<datatype> p1,
               const Vertex<datatype> p2):
        Triangle<datatype>(1, 3, p0, p1, p2) {}

    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x,
                          const datatype y);
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x,
                          const datatype y);
    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                         const datatype x,
                                         const datatype y);
};

template<typename datatype>
class TriangleQ2 : public Triangle<datatype>
{
public:
    TriangleQ2(const Vertex<datatype> p0,
               const Vertex<datatype> p1,
               const Vertex<datatype> p2,
               const size_t p3_id,
               const size_t p4_id,
               const size_t p5_id):
        Triangle<datatype>(1, 3, p0, p1, p2),
        _p3_id{p3_id}, _p4_id{p4_id}, _p5_id{p5_id} {}

    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x,
                          const datatype y);
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x,
                          const datatype y);
    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                         const datatype x,
                                         const datatype y);

//private:
public:
    const size_t _p3_id, _p4_id, _p5_id;
};

// ***** // ***** Rectangle ***** // ***** //

template<typename datatype>
class Rectangle : public Element2D<datatype>
{
protected:
    Rectangle(const unsigned int deg,
              const unsigned int numbf,
              const Vertex<datatype> p0,
              const Vertex<datatype> p1,
              const Vertex<datatype> p2,
              const Vertex<datatype> p3):
        Element2D<datatype>(deg, numbf),
        _p0{p0}, _p1{p1}, _p2{p2}, _p3{p3} {}

public:
    //const std::array<std::array<datatype, 2>, 2> transformation_matrix() const; //TODO

protected:
    const Vertex<datatype> _p0, _p1, _p2, _p3;
};

template<typename datatype>
class RectangleQ1 : public Rectangle<datatype>
{
public:
    RectangleQ1(const Vertex<datatype> p0,
                const Vertex<datatype> p1,
                const Vertex<datatype> p2,
                const Vertex<datatype> p3):
        Rectangle<datatype>(1, 4, p0, p1, p2, p3) {}

    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x,
                          const datatype y);
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x,
                          const datatype y);
};

template<typename datatype>
class RectangleQ2 : public Rectangle<datatype>
{
public:
    RectangleQ2(const Vertex<datatype> p0,
                const Vertex<datatype> p1,
                const Vertex<datatype> p2,
                const Vertex<datatype> p3,
                const size_t p4_id,
                const size_t p5_id,
                const size_t p6_id,
                const size_t p7_id):
        Rectangle<datatype>(1, 4, p0, p1, p2, p3),
        _p4_id{p4_id}, _p5_id{p5_id}, _p6_id{p6_id}, _p7_id{p7_id} {}

    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x,
                          const datatype y);
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x,
                          const datatype y);

private:
    const size_t _p4_id, _p5_id, _p6_id, _p7_id;
};

#include "elements.tpp"

#endif
