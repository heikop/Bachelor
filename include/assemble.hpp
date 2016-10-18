#ifndef __ASSEMBLE_HPP_
#define __ASSEMBLE_HPP_

#include <cassert>
#include <vector>
#include <cmath>
#include "csrmatrixcpu.hpp"
#include "elements.hpp"

#include <array>

template<typename elemtype>
//template<typename datatype, typename elemtype>
void assemble(CsrMatrixCpu<double>& matrix, std::vector<elemtype>& elements)
{
    for (const auto& elem : elements)
    {
        //const std::array<std::array<datatype, 2>, 2> B = elem.transformation_matrix();
        const std::array<std::array<double, 2>, 2> B = elem.transformation_matrix();
        const double detB{std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0])};

        const double bbdd{ B[1][0] * B[1][0] + B[0][0] * B[0][0]};
        const double abcd{-B[1][0] * B[1][1] - B[0][0] * B[0][1]};
        const double aacc{ B[1][1] * B[1][1] + B[0][1] * B[0][1]};

        matrix.add(elem._p0.id, elem._p0.id, ( 3.0*aacc + 6.0*abcd + 3.0*bbdd ) / detB / 6.0);
        matrix.add(elem._p0.id, elem._p1.id, (     aacc +     abcd            ) / detB / 6.0);
        matrix.add(elem._p0.id, elem._p2.id, (                abcd +     bbdd ) / detB / 6.0);
        matrix.add(elem._p0.id, elem._p3_id, (-4.0*aacc - 4.0*abcd            ) / detB / 6.0);
        //trix.add(elem._p0.id, elem._p4_id, (                                ) / detB / 6.0);
        matrix.add(elem._p0.id, elem._p5_id, (          - 4.0*abcd - 4.0*bbdd ) / detB / 6.0);

        matrix.add(elem._p1.id, elem._p0.id, (     aacc +     abcd            ) / detB / 6.0);
        matrix.add(elem._p1.id, elem._p1.id, ( 3.0*aacc                       ) / detB / 6.0);
        matrix.add(elem._p1.id, elem._p2.id, (          -     abcd            ) / detB / 6.0);
        matrix.add(elem._p1.id, elem._p3_id, (-4.0*aacc - 4.0*abcd            ) / detB / 6.0);
        matrix.add(elem._p1.id, elem._p4_id, (            4.0*abcd            ) / detB / 6.0);
        //trix.add(elem._p1.id, elem._p5_id, (                                ) / detB / 6.0);

        matrix.add(elem._p2.id, elem._p0.id, (                abcd +     bbdd ) / detB / 6.0);
        matrix.add(elem._p2.id, elem._p1.id, (          -     abcd            ) / detB / 6.0);
        matrix.add(elem._p2.id, elem._p2.id, (                       3.0*bbdd ) / detB / 6.0);
        //trix.add(elem._p2.id, elem._p3_id, (                                ) / detB / 6.0);
        matrix.add(elem._p2.id, elem._p4_id, (            4.0*abcd            ) / detB / 6.0);
        matrix.add(elem._p2.id, elem._p5_id, (          - 4.0*abcd - 4.0*bbdd ) / detB / 6.0);

        matrix.add(elem._p3_id, elem._p0.id, (-4.0*aacc - 4.0*abcd            ) / detB / 6.0);
        matrix.add(elem._p3_id, elem._p1.id, (-4.0*aacc - 4.0*abcd            ) / detB / 6.0);
        //trix.add(elem._p3_id, elem._p2.id, (                                ) / detB / 6.0);
        matrix.add(elem._p3_id, elem._p3_id, ( 8.0*aacc + 8.0*abcd + 8.0*bbdd ) / detB / 6.0);
        matrix.add(elem._p3_id, elem._p4_id, (          - 8.0*abcd - 8.0*bbdd ) / detB / 6.0);
        matrix.add(elem._p3_id, elem._p5_id, (            8.0*abcd            ) / detB / 6.0);

        //trix.add(elem._p4_id, elem._p0.id, (                                ) / detB / 6.0);
        matrix.add(elem._p4_id, elem._p1.id, (            4.0*abcd            ) / detB / 6.0);
        matrix.add(elem._p4_id, elem._p2.id, (            4.0*abcd            ) / detB / 6.0);
        matrix.add(elem._p4_id, elem._p3_id, (          - 8.0*abcd - 8.0*bbdd ) / detB / 6.0);
        matrix.add(elem._p4_id, elem._p4_id, ( 8.0*aacc + 8.0*abcd + 8.0*bbdd ) / detB / 6.0);
        matrix.add(elem._p4_id, elem._p5_id, (-8.0*aacc - 8.0*abcd            ) / detB / 6.0);

        matrix.add(elem._p5_id, elem._p0.id, (          - 4.0*abcd - 4.0*bbdd ) / detB / 6.0);
        //trix.add(elem._p5_id, elem._p1.id, (                                ) / detB / 6.0);
        matrix.add(elem._p5_id, elem._p2.id, (          - 4.0*abcd - 4.0*bbdd ) / detB / 6.0);
        matrix.add(elem._p5_id, elem._p3_id, (            8.0*abcd            ) / detB / 6.0);
        matrix.add(elem._p5_id, elem._p4_id, (-8.0*aacc - 8.0*abcd            ) / detB / 6.0);
        matrix.add(elem._p5_id, elem._p5_id, ( 8.0*aacc + 8.0*abcd + 8.0*bbdd ) / detB / 6.0);
    }
}

#endif
