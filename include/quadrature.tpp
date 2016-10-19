template<template<typename> class elemtype, typename datatype>
const datatype Quadrature<elemtype, datatype>::integrate(unsigned int order, unsigned int firstbf, unsigned int secondbf) const
{
    double res{0.0};

    if (typeid(_element) == typeid(Triangle<datatype>)
     || typeid(_element) == typeid(TriangleQ1<datatype>)
     || typeid(_element) == typeid(TriangleQ2<datatype>))
    {
        std::array<std::array<datatype, 2>, 2> B = _element.transformation_matrix();
        std::array<std::array<datatype, 2>, 2> B_inv_t;
        datatype detB{std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0])};
        B_inv_t[0][0] =  B[1][1] / detB;
        B_inv_t[0][1] = -B[1][0] / detB;
        B_inv_t[1][0] = -B[0][1] / detB;
        B_inv_t[1][1] =  B[0][0] / detB;

        std::array<datatype, 2> grad1;
        std::array<datatype, 2> grad2;

        switch (order)
        {
        case 1:
            grad1 = _element.gradient_ref(firstbf, 1.0/3.0, 1.0/3.0);
            grad2 = _element.gradient_ref(secondbf, 1.0/3.0, 1.0/3.0);
            res = (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[1][0] * grad2[0] + B_inv_t[1][1] * grad2[1]);
            res *= detB / 2.0; // 1/2 is the size of the referenceelement
            break;
        case 2:
            grad1 = _element.gradient_ref(firstbf, 0.0, 0.5);
            grad2 = _element.gradient_ref(secondbf, 0.0, 0.5);
            res  = (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                 + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[1][0] * grad2[0] + B_inv_t[1][1] * grad2[1]);
            grad1 = _element.gradient_ref(firstbf, 0.5, 0.0);
            grad2 = _element.gradient_ref(secondbf, 0.5, 0.0);
            res += (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                 + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[1][0] * grad2[0] + B_inv_t[1][1] * grad2[1]);
            grad1 = _element.gradient_ref(firstbf, 0.5, 0.5);
            grad2 = _element.gradient_ref(secondbf, 0.5, 0.5);
            res += (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                 + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[1][0] * grad2[0] + B_inv_t[1][1] * grad2[1]);
            res *= detB / 6.0; // 1/3 is weight of all points; 1/2 is the size of the referenceelement
            break;
        default:
            assert(false);
        }
    }
    else if (typeid(_element) == typeid(Quadrilateral<datatype>)
          || typeid(_element) == typeid(QuadrilateralQ1<datatype>)
          || typeid(_element) == typeid(QuadrilateralQ2<datatype>))
    {
        //TODO
        assert(false);
    }
    else
        assert(false);
    return res;
}
