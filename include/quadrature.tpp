template<template<typename> class elemtype, typename datatype>
const datatype Quadrature<elemtype, datatype>::integrate_laplace(const unsigned int order, const unsigned int firstbf, const unsigned int secondbf) const
{
    double res{0.0};

    if (typeid(*_element) == typeid(Triangle<datatype>)
     || typeid(*_element) == typeid(TriangleQ1<datatype>)
     || typeid(*_element) == typeid(TriangleQ2<datatype>))
    {
        std::array<std::array<datatype, 2>, 2> B = _element->transformation_matrix();
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
            grad1 = _element->gradient_ref(firstbf, 1.0/3.0, 1.0/3.0);
            grad2 = _element->gradient_ref(secondbf, 1.0/3.0, 1.0/3.0);
            res = (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[1][0] * grad2[0] + B_inv_t[1][1] * grad2[1]);
            res *= detB / static_cast<datatype>(2.0); // 1/2 is the size of the referenceelement
            break;
        case 2:
            grad1 = _element->gradient_ref(firstbf, 0.0, 0.5);
            grad2 = _element->gradient_ref(secondbf, 0.0, 0.5);
            res  = (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                 + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[1][0] * grad2[0] + B_inv_t[1][1] * grad2[1]);
            grad1 = _element->gradient_ref(firstbf, 0.5, 0.0);
            grad2 = _element->gradient_ref(secondbf, 0.5, 0.0);
            res += (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                 + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[1][0] * grad2[0] + B_inv_t[1][1] * grad2[1]);
            grad1 = _element->gradient_ref(firstbf, 0.5, 0.5);
            grad2 = _element->gradient_ref(secondbf, 0.5, 0.5);
            res += (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                 + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[1][0] * grad2[0] + B_inv_t[1][1] * grad2[1]);
            res *= detB / static_cast<datatype>(6.0); // 1/3 is weight of all points; 1/2 is the size of the referenceelement
            break;
        default:
            assert(false);
        }
    }
    else if (typeid(*_element) == typeid(Quadrilateral<datatype>)
          || typeid(*_element) == typeid(QuadrilateralQ1<datatype>)
          || typeid(*_element) == typeid(QuadrilateralQ2<datatype>))
    {
        std::array<datatype, 2> grad1;
        std::array<datatype, 2> grad2;

        switch (order)
        {
        case 1:
            grad1 = _element->gradient_ref(firstbf, 0.0, 0.0);
            grad2 = _element->gradient_ref(secondbf, 0.0, 0.0);
            res = grad1[0] * grad2[0] + grad1[1] * grad2[1];
            res *= static_cast<datatype>(4.0); // weight is 4
            break;
        case 2:
        {   datatype qp{std::sqrt(static_cast<datatype>(1.0/3.0))};
            grad1 = _element->gradient_ref(firstbf, -qp, -qp);
            grad2 = _element->gradient_ref(secondbf, -qp, -qp);
            res = grad1[0] * grad2[0] + grad1[1] * grad2[1];
            grad1 = _element->gradient_ref(firstbf, -qp, qp);
            grad2 = _element->gradient_ref(secondbf, -qp, qp);
            res += grad1[0] * grad2[0] + grad1[1] * grad2[1];
            grad1 = _element->gradient_ref(firstbf, qp, -qp);
            grad2 = _element->gradient_ref(secondbf, qp, -qp);
            res += grad1[0] * grad2[0] + grad1[1] * grad2[1];
            grad1 = _element->gradient_ref(firstbf, qp, qp);
            grad2 = _element->gradient_ref(secondbf, qp, qp);
            res += grad1[0] * grad2[0] + grad1[1] * grad2[1];
            //res *= 1; // weights of all points are 1
            break; }
        case 3:
        {   datatype qp{std::sqrt(static_cast<datatype>(3.0/5.0))};

            grad1 = _element->gradient_ref(firstbf , -qp, -qp);
            grad2 = _element->gradient_ref(secondbf, -qp, -qp);
            res   = static_cast<datatype>(25) * (grad1[0] * grad2[0] + grad1[1] * grad2[1]) ;//* _element->trafo_determinant(-qp, -qp);
            grad1 = _element->gradient_ref(firstbf , -qp, 0);
            grad2 = _element->gradient_ref(secondbf, -qp, 0);
            res  += static_cast<datatype>(40) * (grad1[0] * grad2[0] + grad1[1] * grad2[1]) ;//* _element->trafo_determinant(-qp, 0);
            grad1 = _element->gradient_ref(firstbf , -qp, qp);
            grad2 = _element->gradient_ref(secondbf, -qp, qp);
            res  += static_cast<datatype>(25) * (grad1[0] * grad2[0] + grad1[1] * grad2[1]) ;//* _element->trafo_determinant(-qp, qp);

            grad1 = _element->gradient_ref(firstbf , 0, -qp);
            grad2 = _element->gradient_ref(secondbf, 0, -qp);
            res  += static_cast<datatype>(40) * (grad1[0] * grad2[0] + grad1[1] * grad2[1]) ;//* _element->trafo_determinant(0, -qp);
            grad1 = _element->gradient_ref(firstbf , 0, 0);
            grad2 = _element->gradient_ref(secondbf, 0, 0);
            res  += static_cast<datatype>(64) * (grad1[0] * grad2[0] + grad1[1] * grad2[1]) ;//* _element->trafo_determinant(0, 0);
            grad1 = _element->gradient_ref(firstbf , 0, qp);
            grad2 = _element->gradient_ref(secondbf, 0, qp);
            res  += static_cast<datatype>(40) * (grad1[0] * grad2[0] + grad1[1] * grad2[1]) ;//* _element->trafo_determinant(0, qp);

            grad1 = _element->gradient_ref(firstbf , qp, -qp);
            grad2 = _element->gradient_ref(secondbf, qp, -qp);
            res  += static_cast<datatype>(25) * (grad1[0] * grad2[0] + grad1[1] * grad2[1]) ;//* _element->trafo_determinant(qp, -qp);
            grad1 = _element->gradient_ref(firstbf , qp, 0);
            grad2 = _element->gradient_ref(secondbf, qp, 0);
            res  += static_cast<datatype>(40) * (grad1[0] * grad2[0] + grad1[1] * grad2[1]) ;//* _element->trafo_determinant(qp, 0);
            grad1 = _element->gradient_ref(firstbf , qp, qp);
            grad2 = _element->gradient_ref(secondbf, qp, qp);
            res  += static_cast<datatype>(25) * (grad1[0] * grad2[0] + grad1[1] * grad2[1]) ;//* _element->trafo_determinant(qp, qp);

            res /= static_cast<datatype>(81); // all weights are .../81
            break; }
        default:
            assert(false);
        }

        //res /= _element->area();
        //res /= _element->area() / static_cast<datatype>(4);
    }
    else
        assert(false);
    return res;
}

template<template<typename> class elemtype, typename datatype>
const datatype Quadrature<elemtype, datatype>::integrate_basisfunction(const unsigned int order, const unsigned int basisfunction) const
{
    double res{0.0};

    if (typeid(*_element) == typeid(Triangle<datatype>)
     || typeid(*_element) == typeid(TriangleQ1<datatype>)
     || typeid(*_element) == typeid(TriangleQ2<datatype>))
    {
        std::array<std::array<datatype, 2>, 2> B = _element->transformation_matrix();
        datatype detB{std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0])};
        switch (order)
        {
        case 1:
            res = _element->evaluate_ref(basisfunction, 1.0/3.0, 1.0/3.0);
            res *= detB / static_cast<datatype>(2.0); // 1/2 is the size of the referenceelement
            break;
        case 2:
            res = _element->evaluate_ref(basisfunction, 0.0, 0.5)
                + _element->evaluate_ref(basisfunction, 0.5, 0.0)
                + _element->evaluate_ref(basisfunction, 0.5, 0.5);
            res *= detB / static_cast<datatype>(6.0); // 1/3 is weight of all points; 1/2 is the size of the referenceelement
            break;
        default:
            assert(false);
        }
    }
    else if (typeid(*_element) == typeid(Quadrilateral<datatype>)
          || typeid(*_element) == typeid(QuadrilateralQ1<datatype>)
          || typeid(*_element) == typeid(QuadrilateralQ2<datatype>))
    {
        switch (order)
        {
        case 1:
            res = _element->evaluate_ref(basisfunction, 0.0, 0.0);
            res *= static_cast<datatype>(4.0); // weight is 4
            break;
        case 2:
        {   datatype qp{std::sqrt(static_cast<datatype>(1.0/3.0))};
            res = _element->evaluate_ref(basisfunction, -qp, -qp)
                + _element->evaluate_ref(basisfunction, -qp, qp)
                + _element->evaluate_ref(basisfunction, qp, -qp)
                + _element->evaluate_ref(basisfunction, qp, qp);
            //res *= 1; // weights of all points are 1
            break; }
        case 3:
        {   datatype qp{std::sqrt(static_cast<datatype>(3.0/5.0))};
            res = static_cast<datatype>(25) /* _element->trafo_determinant(-qp, -qp) */* _element->evaluate_ref(basisfunction, -qp, -qp)
                + static_cast<datatype>(40) /* _element->trafo_determinant(-qp,   0) */* _element->evaluate_ref(basisfunction, -qp, static_cast<datatype>(0))
                + static_cast<datatype>(25) /* _element->trafo_determinant(-qp,  qp) */* _element->evaluate_ref(basisfunction, -qp, qp)
                + static_cast<datatype>(40) /* _element->trafo_determinant(  0, -qp) */* _element->evaluate_ref(basisfunction, static_cast<datatype>(0), -qp)
                + static_cast<datatype>(64) /* _element->trafo_determinant(  0,   0) */* _element->evaluate_ref(basisfunction, static_cast<datatype>(0), static_cast<datatype>(0))
                + static_cast<datatype>(40) /* _element->trafo_determinant(  0,  qp) */* _element->evaluate_ref(basisfunction, static_cast<datatype>(0), qp)
                + static_cast<datatype>(25) /* _element->trafo_determinant( qp, -qp) */* _element->evaluate_ref(basisfunction, qp, -qp)
                + static_cast<datatype>(40) /* _element->trafo_determinant( qp,   0) */* _element->evaluate_ref(basisfunction, qp, static_cast<datatype>(0))
                + static_cast<datatype>(25) /* _element->trafo_determinant( qp,  qp) */* _element->evaluate_ref(basisfunction, qp, qp);
            res /= static_cast<datatype>(81); // all weights are .../81
            break; }
        default:
            assert(false);
        }
        //res /= _element->area();
        //res /= _element->area() / static_cast<datatype>(4);
    }
    else
        assert(false);
    return res;
}
