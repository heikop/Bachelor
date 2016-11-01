#ifndef __ASSEMBLE_HPP_
#define __ASSEMBLE_HPP_

//#include <cassert>
#include <vector>
#include "csrmatrixcpu.hpp"
#include "elements.hpp"
#include "quadrature.hpp"

#include <omp.h>
#include <stdio.h>

template<template<typename> class elemtype, typename datatype>
void assemble(CsrMatrixCpu<datatype>& matrix, std::vector<elemtype<datatype>*>& elements)
{
/*
    for (const auto& elem : elements)
    {
        const std::vector<size_t> vertexids{elem->vertexids()};
        Quadrature<elemtype, datatype> quad(elem);
        for (size_t i{0}; i < vertexids.size(); ++i)
            for (size_t j{0}; j < vertexids.size(); ++j)
//                matrix.add(vertexids[i], vertexids[j], quad.integrate_laplace(2, i, j));
                matrix.add(vertexids[i], vertexids[j], quad.integrate_laplace(3, i, j));
//                matrix.add(vertexids[i], vertexids[j], quad.integrate_laplace(4, i, j));
    }
*/

    #pragma omp parallel for
    for (int e=0; e < static_cast<int>(elements.size()); e++)
    {
        const std::vector<size_t> vertexids{elements[e]->vertexids()};
        Quadrature<elemtype, datatype> quad(elements[e]);
        for (size_t i{0}; i < vertexids.size(); ++i)
            for (size_t j{0}; j < vertexids.size(); ++j)
//                matrix.add(vertexids[i], vertexids[j], quad.integrate_laplace(2, i, j));
                matrix.add(vertexids[i], vertexids[j], quad.integrate_laplace(3, i, j));
//                quad.integrate_laplace(3, i, j);
//                matrix.add(vertexids[i], vertexids[j], quad.integrate_laplace(4, i, j));
    }
}

#endif
