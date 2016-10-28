#ifndef __ELEMENTS_HPP_
#define __ELEMENTS_HPP_

#include <cassert>
#include <array>
#include <cmath>
#include <utility>
#include <functional>

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

    virtual const std::vector<size_t> vertexids() const = 0;

    virtual datatype evaluate_ref(const unsigned int basis_function,
                                  const datatype x) const = 0;
    virtual datatype evaluate_ref(const unsigned int basis_function,
                                  const datatype x,
                                  const datatype y) const = 0;
    virtual std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                                 const datatype x) = 0;
    virtual std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                                 const datatype x,
                                                 const datatype y) const = 0;
    virtual const std::array<std::array<datatype, 2>, 2> transformation_matrix() const;
    virtual const datatype area() const;
    virtual const datatype trafo_determinant(const datatype xi,
                                             const datatype eta) const;

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
                                  const datatype x) const = 0;
    virtual datatype derivate_ref(const unsigned int basis_function,
                                  const unsigned int direction,
                                  const datatype x) const = 0;
    virtual const std::vector<size_t> vertexids() const = 0;

private:
    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                         const datatype x,
                                         const datatype y) const
        { assert(false); }
    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x,
                          const datatype y) const
        { assert(false); }
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
    virtual const std::vector<size_t> vertexids() const = 0;

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
                          const datatype x) const;
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x) const;
    const std::vector<size_t> vertexids() const
        { return std::vector<size_t>{this->_p0.id, this->_p1.id}; }
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
                          const datatype x) const;
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x) const;
    const std::vector<size_t> vertexids() const
        { return std::vector<size_t>{this->_p0.id, this->_p1.id, _p2_id}; }
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
                                  const datatype y) const = 0;
    virtual datatype derivate_ref(const unsigned int basis_function,
                                  const unsigned int direction,
                                  const datatype x,
                                  const datatype y) const = 0;
    virtual std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                                 const datatype x,
                                                 const datatype y) const = 0;
    virtual const std::vector<size_t> vertexids() const = 0;

private:
    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                         const datatype x)
        { assert(false); }
    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x) const
        { assert(false); }

//    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
//                                         const datatype x) const = delete;
//private:
//    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
//                                         const datatype x)
//        { assert(false); return {0,0}; }
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
    virtual const std::vector<size_t> vertexids() const = 0;
    virtual const datatype area() const {
            return 0; // TODO
            }
    virtual const datatype trafo_determinant(const datatype xi,
                                             const datatype eta) const {
            return 0; // TODO
            }

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
                          const datatype y) const;
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x,
                          const datatype y) const;
    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                         const datatype x,
                                         const datatype y) const;
    const std::vector<size_t> vertexids() const
        { return std::vector<size_t>{this->_p0.id, this->_p1.id, this->_p2.id}; }
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
        Triangle<datatype>(2, 6, p0, p1, p2),
        _p3_id{p3_id}, _p4_id{p4_id}, _p5_id{p5_id} {}

    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x,
                          const datatype y) const;
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x,
                          const datatype y) const;
    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                         const datatype x,
                                         const datatype y) const;
    const std::vector<size_t> vertexids() const
        { return std::vector<size_t>{this->_p0.id, this->_p1.id, this->_p2.id, _p3_id, _p4_id, _p5_id}; }

private:
    const size_t _p3_id, _p4_id, _p5_id;
};

// ***** // ***** Quadrilateral ***** // ***** //

template<typename datatype>
class Quadrilateral : public Element2D<datatype>
{
protected:
    Quadrilateral(const unsigned int deg,
                  const unsigned int numbf,
                  const Vertex<datatype> p0,
                  const Vertex<datatype> p1,
                  const Vertex<datatype> p2,
                  const Vertex<datatype> p3):
        Element2D<datatype>(deg, numbf),
        _p0{p0}, _p1{p1}, _p2{p2}, _p3{p3} {}
    virtual const std::vector<size_t> vertexids() const = 0;

public:
    const datatype trafo_determinant(const datatype xi,
                                     const datatype eta) const {
        std::array<std::array<datatype, 2>, 2> B;
        //B[0][0] = this->_p1.x + this->_p3.x * eta;
        //B[0][1] = this->_p2.x + this->_p3.x * xi;
        //B[1][0] = this->_p1.y + this->_p3.y * eta;
        //B[1][1] = this->_p2.y + this->_p3.y * xi;
        //return {std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0])};
        B[0][0] = -(1.0-eta)/4.0*this->_p0.x + (1.0-eta)/4.0*this->_p1.x + (1.0+eta)/4.0*this->_p2.x - (1.0+eta)/4.0*this->_p3.x;
        B[0][1] = -(1.0-xi )/4.0*this->_p0.x - (1.0+xi )/4.0*this->_p1.x + (1.0+xi )/4.0*this->_p2.x + (1.0-xi )/4.0*this->_p3.x;
        B[1][0] = -(1.0-eta)/4.0*this->_p0.y + (1.0-eta)/4.0*this->_p1.y + (1.0+eta)/4.0*this->_p2.y - (1.0+eta)/4.0*this->_p3.y;
        B[1][1] = -(1.0-xi )/4.0*this->_p0.y - (1.0+xi )/4.0*this->_p1.y + (1.0+xi )/4.0*this->_p2.y + (1.0-xi )/4.0*this->_p3.y;
        return {std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0])};
            }
    const datatype area() const {
    return std::abs((_p2.x - _p0.x) * (_p3.y - _p1.y)
           -(_p2.y - _p0.y) * (_p3.x - _p1.x))
                                / static_cast<datatype>(2);
    }
    const std::array<std::array<datatype, 2>, 2> transformation_matrix() const {assert(false);} //TODO

protected:
    const Vertex<datatype> _p0, _p1, _p2, _p3;
};

template<typename datatype>
class QuadrilateralQ1 : public Quadrilateral<datatype>
{
public:
    QuadrilateralQ1(const Vertex<datatype> p0,
                    const Vertex<datatype> p1,
                    const Vertex<datatype> p2,
                    const Vertex<datatype> p3):
        Quadrilateral<datatype>(1, 4, p0, p1, p2, p3) {}

    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x,
                          const datatype y) const;
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x,
                          const datatype y) const;
    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                         const datatype x,
                                         const datatype y) const;
    const std::vector<size_t> vertexids() const
        { return std::vector<size_t>{this->_p0.id, this->_p1.id, this->_p2.id, this->_p3.id}; }
};

template<typename datatype>
class QuadrilateralQ2 : public Quadrilateral<datatype>
{
public:
    QuadrilateralQ2(const Vertex<datatype> p0,
                    const Vertex<datatype> p1,
                    const Vertex<datatype> p2,
                    const Vertex<datatype> p3,
                    const size_t p4_id,
                    const size_t p5_id,
                    const size_t p6_id,
                    const size_t p7_id,
                    const size_t p8_id):
        Quadrilateral<datatype>(2, 9, p0, p1, p2, p3),
        _p4_id{p4_id}, _p5_id{p5_id}, _p6_id{p6_id}, _p7_id{p7_id}, _p8_id{p8_id} {}

    datatype evaluate_ref(const unsigned int basis_function,
                          const datatype x,
                          const datatype y) const;
    datatype derivate_ref(const unsigned int basis_function,
                          const unsigned int direction,
                          const datatype x,
                          const datatype y) const;
    std::array<datatype, 2> gradient_ref(const unsigned int basis_function,
                                         const datatype x,
                                         const datatype y) const;
    const std::vector<size_t> vertexids() const
        { return std::vector<size_t>{this->_p0.id, this->_p1.id, this->_p2.id, this->_p3.id, _p4_id, _p5_id, _p6_id, _p7_id, _p8_id}; }

private:
    const size_t _p4_id, _p5_id, _p6_id, _p7_id, _p8_id;
};

#include "elements.tpp"

#endif
