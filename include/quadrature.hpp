#ifndef __QUADRATURE_HPP_
#define __QUADRATURE_HPP_

#include <cassert>
#include <cmath>
#include <array>
#include <typeinfo>

#include "elements.hpp"

//TODO double to datatype
template<typename elemtype>
class Quadrature
{
public:
    Quadrature(const elemtype& element) :
        _element{element} {}
    double integrate(unsigned int order, unsigned int firstbf, unsigned int secondbf);

private:
    const elemtype& _element;
};

#include "quadrature.tpp"

#endif
