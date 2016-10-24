#ifndef __QUADRATURE_HPP_
#define __QUADRATURE_HPP_

#include <cassert>
#include <cmath>
#include <array>
#include <typeinfo>
#include <functional>

#include "elements.hpp"

template<template<typename> class elemtype, typename datatype>
class Quadrature
{
public:
    Quadrature(const elemtype<datatype>* const element) :
        _element{element} {}
    const datatype integrate_laplace(const unsigned int order, const unsigned int firstbf, const unsigned int secondbf) const;
    const datatype integrate_basisfunction(const unsigned int order, const unsigned int basisfunction) const;

private:
    const elemtype<datatype>* const _element;
};

#include "quadrature.tpp"

#endif
