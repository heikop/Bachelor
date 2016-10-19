#ifndef __QUADRATURE_HPP_
#define __QUADRATURE_HPP_

#include <cassert>
#include <cmath>
#include <array>
#include <typeinfo>

#include "elements.hpp"

template<template<typename> class elemtype, typename datatype>
class Quadrature
{
public:
    Quadrature(const elemtype<datatype>& element) :
        _element{element} {}
    const datatype integrate(unsigned int order, unsigned int firstbf, unsigned int secondbf) const;

private:
    const elemtype<datatype>& _element;
};

#include "quadrature.tpp"

#endif
