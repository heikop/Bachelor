template<typename elemtype>
double Quadrature<elemtype>::integrate(unsigned int order, unsigned int firstbf, unsigned int secondbf)
{
    if (typeid(_element) == typeid(Triangle<double>)
     || typeid(_element) == typeid(TriangleQ1<double>)
     || typeid(_element) == typeid(TriangleQ2<double>))
    {
        std::array<std::array<double, 2>, 2> B = _element.transformation_matrix();
        std::array<std::array<double, 2>, 2> B_inv_t;
        double detB{std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0])};
        B_inv_t[0][0] =  B[1][1] / detB;
        B_inv_t[0][1] = -B[1][0] / detB;
        B_inv_t[1][0] = -B[0][1] / detB;
        B_inv_t[1][1] =  B[0][0] / detB;

        std::array<double, 2> grad1;
        std::array<double, 2> grad2;

        double res{0.0};

        switch (order)
        {
        case 1:
            grad1 = _element.gradient_ref(firstbf, 1.0/3.0, 1.0/3.0);
            grad2 = _element.gradient_ref(secondbf, 1.0/3.0, 1.0/3.0);

            res = (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1]);
            res *= detB;
            break;
        case 2:
            grad1 = _element.gradient_ref(firstbf, 0, 0.5);
            grad2 = _element.gradient_ref(secondbf, 0, 0.5);
            res  = (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                 + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1]);
            grad1 = _element.gradient_ref(firstbf, 0.5, 0.0);
            grad2 = _element.gradient_ref(secondbf, 0.5, 0.0);
            res += (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                 + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1]);
            grad1 = _element.gradient_ref(firstbf, 0.5, 0.5);
            grad2 = _element.gradient_ref(secondbf, 0.5, 0.5);
            res += (B_inv_t[0][0] * grad1[0] + B_inv_t[0][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1])
                 + (B_inv_t[1][0] * grad1[0] + B_inv_t[1][1] * grad1[1]) * (B_inv_t[0][0] * grad2[0] + B_inv_t[0][1] * grad2[1]);
            res *= detB / 3.0; // 1/3 is weight of all points
            break;
        default:
            ;
        }
//    const std::array<std::array<datatype, 2>, 2> transformation_matrix() const;
    }
    return res;
}
