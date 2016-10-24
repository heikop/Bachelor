// ***** // ***** // ***** // ***** 1D ***** // ***** // ***** // ***** //

// ***** // ***** LineQ1 ***** // ***** //
template<typename datatype>
datatype LineQ1<datatype>::evaluate_ref(const unsigned int basis_function,
                                        const datatype x) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(0) && x <= static_cast<datatype>(1));
    if (basis_function == 0)
        return static_cast<datatype>(1) - x;
    else //if (basis_function == 1)
        return x;
}

template<typename datatype>
datatype LineQ1<datatype>::derivate_ref(const unsigned int basis_function,
                                        const unsigned int direction,
                                        const datatype x) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(0) && x <= static_cast<datatype>(1));
    if (basis_function == 0)
        return static_cast<datatype>(-1);
    else //if (basis_function == 1)
        return static_cast<datatype>(1);
}

// ***** // ***** LineQ2 ***** // ***** //
template<typename datatype>
datatype LineQ2<datatype>::evaluate_ref(const unsigned int basis_function,
                                        const datatype x) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(0) && x <= static_cast<datatype>(1));
    if (basis_function == 0)
        return (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) - static_cast<datatype>(2) * x);
    else if (basis_function == 1)
        return x * (static_cast<datatype>(2) * x - static_cast<datatype>(1));
    else //if (basis_function == 2)
        return static_cast<datatype>(4) * x * (static_cast<datatype>(1) - x);
}

template<typename datatype>
datatype LineQ2<datatype>::derivate_ref(const unsigned int basis_function,
                                        const unsigned int direction,
                                        const datatype x) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(0) && x <= static_cast<datatype>(1));
    if (basis_function == 0)
        return static_cast<datatype>(4) * x - static_cast<datatype>(3);
    else if (basis_function == 1)
        return x * static_cast<datatype>(4) * x - static_cast<datatype>(1);
    else //if (basis_function == 2)
        return static_cast<datatype>(4) - static_cast<datatype>(8) * x;
}


// ***** // ***** // ***** // ***** 2D ***** // ***** // ***** // ***** //


// ***** // ***** Triangle ***** // ***** //
template<typename datatype>
const std::array<std::array<datatype, 2>, 2> Triangle<datatype>::transformation_matrix() const
{
    //return {_p1.x - _p0.x, _p2.x - _p0.x,
    //        _p1.y - _p0.y, _p2.y - _p0.y};
    std::array<std::array<datatype, 2>, 2> res;
    res[0][0] = _p1.x - _p0.x; res[0][1] = _p2.x - _p0.x;
    res[1][0] = _p1.y - _p0.y; res[1][1] = _p2.y - _p0.y;
    return res;
    // TODO change back
}

// ***** // ***** TriangleQ1 ***** // ***** //
template<typename datatype>
datatype TriangleQ1<datatype>::evaluate_ref(const unsigned int basis_function,
                                            const datatype x,
                                            const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(0)
        && y >= static_cast<datatype>(0)
        && x+y <= static_cast<datatype>(1));
    if (basis_function == 0)
        return static_cast<datatype>(1) - x - y;
    else if (basis_function == 1)
        return x;
    else //if (basis_function == 2)
        return y;
}

template<typename datatype>
datatype TriangleQ1<datatype>::derivate_ref(const unsigned int basis_function,
                                            const unsigned int direction,
                                            const datatype x,
                                            const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(direction == 0 || direction == 1);
    assert(x >= static_cast<datatype>(0)
        && y >= static_cast<datatype>(0)
        && x+y <= static_cast<datatype>(1));
    if (basis_function == 0)
        return static_cast<datatype>(-1);
    else if (basis_function == 1)
        return (direction == 0 ? static_cast<datatype>(1) : static_cast<datatype>(0));
    else //if (basis_function == 2)
        return (direction == 1 ? static_cast<datatype>(1) : static_cast<datatype>(0));
}

template<typename datatype>
std::array<datatype, 2> TriangleQ1<datatype>::gradient_ref(const unsigned int basis_function,
                                                           const datatype x,
                                                           const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(0)
        && y >= static_cast<datatype>(0)
        && x+y <= static_cast<datatype>(1));
    if (basis_function == 0)
        return {static_cast<datatype>(-1),
                static_cast<datatype>(-1)};
    else if (basis_function == 1)
        return {static_cast<datatype>(1),
                static_cast<datatype>(0)};
    else //if (basis_function == 2)
        return {static_cast<datatype>(0),
                static_cast<datatype>(1)};
}

// ***** // ***** TriangleQ2 ***** // ***** //
template<typename datatype>
datatype TriangleQ2<datatype>::evaluate_ref(const unsigned int basis_function,
                                            const datatype x,
                                            const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(0)
        && y >= static_cast<datatype>(0)
        && x+y <= static_cast<datatype>(1));
    if (basis_function == 0)
        return (static_cast<datatype>(1) - x - y) * (static_cast<datatype>(1) - static_cast<datatype>(2)*x - static_cast<datatype>(2)*y);
    else if (basis_function == 1)
        return x * (static_cast<datatype>(2)*x - static_cast<datatype>(1));
    else if (basis_function == 2)
        return y * (static_cast<datatype>(2)*y - static_cast<datatype>(1));
    else if (basis_function == 3)
        return static_cast<datatype>(4) * x * (static_cast<datatype>(1) - x - y);
    else if (basis_function == 4)
        return static_cast<datatype>(4) * x * y;
    else //if (basis_function == 5)
        return static_cast<datatype>(4) * y * (static_cast<datatype>(1) - x - y);
}

template<typename datatype>
datatype TriangleQ2<datatype>::derivate_ref(const unsigned int basis_function,
                                            const unsigned int direction,
                                            const datatype x,
                                            const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(direction == 0 || direction == 1);
    assert(x >= static_cast<datatype>(0)
        && y >= static_cast<datatype>(0)
        && x+y <= static_cast<datatype>(1));
    if (basis_function == 0)
        return static_cast<datatype>(4) * (x + y) - static_cast<datatype>(3);
    else if (basis_function == 1)
        return (direction == 0 ? static_cast<datatype>(4) * x - static_cast<datatype>(1) :
                                 static_cast<datatype>(0) );
    else if (basis_function == 2)
        return (direction == 0 ? static_cast<datatype>(0) :
                                 static_cast<datatype>(4) * y - static_cast<datatype>(1) );
    else if (basis_function == 3)
        return (direction == 0 ? static_cast<datatype>(4) * (static_cast<datatype>(1) - static_cast<datatype>(2) * x - y) :
                                 static_cast<datatype>(-4) * x );
    else if (basis_function == 4)
        return (direction == 0 ? static_cast<datatype>(4) * y :
                                 static_cast<datatype>(4) * x);
    else //if (basis_function == 5)
        return (direction == 0 ? static_cast<datatype>(-4) * y :
                                 static_cast<datatype>(4) * (static_cast<datatype>(1) - x - static_cast<datatype>(2) * y) );
}

template<typename datatype>
std::array<datatype, 2> TriangleQ2<datatype>::gradient_ref(const unsigned int basis_function,
                                                           const datatype x,
                                                           const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(0)
        && y >= static_cast<datatype>(0)
        && x+y <= static_cast<datatype>(1));

    if (basis_function == 0)
        return {static_cast<datatype>(4) * (x + y) - static_cast<datatype>(3),
                static_cast<datatype>(4) * (x + y) - static_cast<datatype>(3)};
    else if (basis_function == 1)
        return {static_cast<datatype>(4) * x - static_cast<datatype>(1),
                static_cast<datatype>(0)};
    else if (basis_function == 2)
        return {static_cast<datatype>(0),
                static_cast<datatype>(4) * y - static_cast<datatype>(1)};
    else if (basis_function == 3)
        return {static_cast<datatype>(4) * (static_cast<datatype>(1) - static_cast<datatype>(2) * x - y),
                static_cast<datatype>(-4) * x};
    else if (basis_function == 4)
        return {static_cast<datatype>(4) * y,
                static_cast<datatype>(4) * x};
    else //if (basis_function == 5)
        return {static_cast<datatype>(-4) * y,
                static_cast<datatype>(4) * (static_cast<datatype>(1) - x - static_cast<datatype>(2) * y )};
}

// ***** // ***** Quadrilateral ***** // ***** //
// TODO trafo

// ***** // ***** QuadrilateralQ1 ***** // ***** //
template<typename datatype>
datatype QuadrilateralQ1<datatype>::evaluate_ref(const unsigned int basis_function,
                                                 const datatype xi,
                                                 const datatype eta) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(xi  >= static_cast<datatype>(-1)
        && eta >= static_cast<datatype>(-1)
        && xi  <= static_cast<datatype>( 1)
        && eta <= static_cast<datatype>( 1));

    std::array<std::array<datatype, 2>, 2> B_inv_t;
    B_inv_t[0][0] = -(1.0-eta)/4.0*this->_p0.x + (1.0-eta)/4.0*this->_p1.x + (1.0+eta)/4.0*this->_p2.x - (1.0+eta)/4.0*this->_p3.x;
    B_inv_t[0][1] = -(1.0-xi )/4.0*this->_p0.x - (1.0+xi )/4.0*this->_p1.x + (1.0+xi )/4.0*this->_p2.x + (1.0-xi )/4.0*this->_p3.x;
    B_inv_t[1][0] = -(1.0-eta)/4.0*this->_p0.y + (1.0-eta)/4.0*this->_p1.y + (1.0+eta)/4.0*this->_p2.y - (1.0+eta)/4.0*this->_p3.y;
    B_inv_t[1][1] = -(1.0-xi )/4.0*this->_p0.y - (1.0+xi )/4.0*this->_p1.y + (1.0+xi )/4.0*this->_p2.y + (1.0-xi )/4.0*this->_p3.y;
    datatype detB = std::abs(B_inv_t[0][0] * B_inv_t[1][1] - B_inv_t[0][1] * B_inv_t[1][0]);

    if (basis_function == 0)
        return (static_cast<datatype>(1) - xi) * (static_cast<datatype>(1) - eta) / static_cast<datatype>(4) * detB;
    else if (basis_function == 1)
        return (static_cast<datatype>(1) + xi) * (static_cast<datatype>(1) - eta) / static_cast<datatype>(4) * detB;
    else if (basis_function == 2)
        return (static_cast<datatype>(1) + xi) * (static_cast<datatype>(1) + eta) / static_cast<datatype>(4) * detB;
    else //if (basis_function == 3)
        return (static_cast<datatype>(1) - xi) * (static_cast<datatype>(1) + eta) / static_cast<datatype>(4) * detB;
}

template<typename datatype>
datatype QuadrilateralQ1<datatype>::derivate_ref(const unsigned int basis_function,
                                                 const unsigned int direction,
                                                 const datatype x,
                                                 const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(direction == 0 || direction == 1);
    assert(x >= static_cast<datatype>(-1)
        && y >= static_cast<datatype>(-1)
        && x <= static_cast<datatype>( 1)
        && y <= static_cast<datatype>( 1));
    if (basis_function == 0)
        return (direction == 0 ? (static_cast<datatype>(1) - y) / static_cast<datatype>(-4) :
                                 (static_cast<datatype>(1) - x) / static_cast<datatype>(-4) );
    else if (basis_function == 1)
        return (direction == 0 ? (static_cast<datatype>(1) - y) / static_cast<datatype>( 4) :
                                 (static_cast<datatype>(1) + x) / static_cast<datatype>(-4) );
    else if (basis_function == 2)
        return (direction == 0 ? (static_cast<datatype>(1) + y) / static_cast<datatype>( 4) :
                                 (static_cast<datatype>(1) + x) / static_cast<datatype>( 4) );
    else //if (basis_function == 3)
        return (direction == 0 ? (static_cast<datatype>(1) + y) / static_cast<datatype>(-4) :
                                 (static_cast<datatype>(1) - x) / static_cast<datatype>( 4) );
}

template<typename datatype>
std::array<datatype, 2> QuadrilateralQ1<datatype>::gradient_ref(const unsigned int basis_function,
                                                                const datatype xi,
                                                                const datatype eta) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(xi >= static_cast<datatype>(-1)
        && eta >= static_cast<datatype>(-1)
        && xi <= static_cast<datatype>( 1)
        && eta <= static_cast<datatype>( 1));

//*
    std::array<std::array<datatype, 2>, 2> B;
    B[0][0] = this->_p1.x + this->_p3.x * eta;
    B[0][1] = this->_p2.x + this->_p3.x * xi;
    B[1][0] = this->_p1.y + this->_p3.y * eta;
    B[1][1] = this->_p2.y + this->_p3.y * xi;
    datatype detB{std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0])};
    std::array<std::array<datatype, 2>, 2> B_inv_t;
    B_inv_t[0][0] =  B[1][1] / detB; B_inv_t[0][1] = -B[1][0] / detB;
    B_inv_t[1][0] = -B[0][1] / detB; B_inv_t[1][1] =  B[0][0] / detB;

    B_inv_t[0][0] = -(1.0-eta)/4.0*this->_p0.x + (1.0-eta)/4.0*this->_p1.x + (1.0+eta)/4.0*this->_p2.x - (1.0+eta)/4.0*this->_p3.x;
    B_inv_t[0][1] = -(1.0-xi )/4.0*this->_p0.x - (1.0+xi )/4.0*this->_p1.x + (1.0+xi )/4.0*this->_p2.x + (1.0-xi )/4.0*this->_p3.x;
    B_inv_t[1][0] = -(1.0-eta)/4.0*this->_p0.y + (1.0-eta)/4.0*this->_p1.y + (1.0+eta)/4.0*this->_p2.y - (1.0+eta)/4.0*this->_p3.y;
    B_inv_t[1][1] = -(1.0-xi )/4.0*this->_p0.y - (1.0+xi )/4.0*this->_p1.y + (1.0+xi )/4.0*this->_p2.y + (1.0-xi )/4.0*this->_p3.y;
    detB = std::abs(B_inv_t[0][0] * B_inv_t[1][1] - B_inv_t[0][1] * B_inv_t[1][0]);
    B_inv_t[0][0] *= std::sqrt(detB);
    B_inv_t[0][1] *= std::sqrt(detB);
    B_inv_t[1][0] *= std::sqrt(detB);
    B_inv_t[1][1] *= std::sqrt(detB);

    std::array<datatype, 2> grad;
    if (basis_function == 0)
        grad = {(static_cast<datatype>(1) - eta) / static_cast<datatype>(-4) ,
                (static_cast<datatype>(1) - xi ) / static_cast<datatype>(-4) };
    else if (basis_function == 1)
        grad = {(static_cast<datatype>(1) - eta) / static_cast<datatype>( 4) ,
                (static_cast<datatype>(1) + xi ) / static_cast<datatype>(-4) };
    else if (basis_function == 2)
        grad = {(static_cast<datatype>(1) + eta) / static_cast<datatype>( 4) ,
                (static_cast<datatype>(1) + xi ) / static_cast<datatype>( 4) };
    else //if (basis_function == 3)
        grad = {(static_cast<datatype>(1) + eta) / static_cast<datatype>(-4) ,
                (static_cast<datatype>(1) - xi ) / static_cast<datatype>( 4) };
    return {B_inv_t[0][0] * grad[0] + B_inv_t[0][1] * grad[1],
            B_inv_t[1][0] * grad[0] + B_inv_t[1][1] * grad[1]};
//*/
/*

    if (basis_function == 0)
        return {(static_cast<datatype>(1) - eta) / static_cast<datatype>(-4) ,
                (static_cast<datatype>(1) - xi ) / static_cast<datatype>(-4) };
    else if (basis_function == 1)
        return {(static_cast<datatype>(1) - eta) / static_cast<datatype>( 4) ,
                (static_cast<datatype>(1) + xi ) / static_cast<datatype>(-4) };
    else if (basis_function == 2)
        return {(static_cast<datatype>(1) + eta) / static_cast<datatype>( 4) ,
                (static_cast<datatype>(1) + xi ) / static_cast<datatype>( 4) };
    else //if (basis_function == 3)
        return {(static_cast<datatype>(1) + eta) / static_cast<datatype>(-4) ,
                (static_cast<datatype>(1) - xi ) / static_cast<datatype>( 4) };
*/
}

// ***** // ***** QuadrilateralQ2 ***** // ***** //
// TODO
template<typename datatype>
datatype QuadrilateralQ2<datatype>::evaluate_ref(const unsigned int basis_function,
                                                 const datatype x,
                                                 const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(-1)
        && y >= static_cast<datatype>(-1)
        && x <= static_cast<datatype>( 1)
        && y <= static_cast<datatype>( 1));
    if (basis_function == 0)
        return x * (static_cast<datatype>(1) - x) * y * (static_cast<datatype>(1) - y) / static_cast<datatype>(4);
    else if (basis_function == 1)
        return x * (static_cast<datatype>(1) + x) * y * (static_cast<datatype>(1) - y) / static_cast<datatype>(4);
    else if (basis_function == 2)
        return x * (static_cast<datatype>(1) + x) * y * (static_cast<datatype>(1) + y) / static_cast<datatype>(4);
    else if (basis_function == 3)
        return x * (static_cast<datatype>(1) - x) * y * (static_cast<datatype>(1) + y) / static_cast<datatype>(4);
    else if (basis_function == 4)
        return (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) + x) * y * (static_cast<datatype>(1) - y) / static_cast<datatype>(-2);
    else if (basis_function == 5)
        return x * (static_cast<datatype>(1) + x) * (static_cast<datatype>(1) - y) * (static_cast<datatype>(1) + y) / static_cast<datatype>(-2);
    else if (basis_function == 6)
        return (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) + x) * y * (static_cast<datatype>(1) + y) / static_cast<datatype>(-2);
    else if (basis_function == 7)
        return x * (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) - y) * (static_cast<datatype>(1) + y) / static_cast<datatype>(-2);
    else //if (basis_function == 8)
        return (static_cast<datatype>(1) - x*x) * (static_cast<datatype>(1) - y*y);
}

template<typename datatype>
datatype QuadrilateralQ2<datatype>::derivate_ref(const unsigned int basis_function,
                                                 const unsigned int direction,
                                                 const datatype x,
                                                 const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(direction == 0 || direction == 1);
    assert(x >= static_cast<datatype>(-1)
        && y >= static_cast<datatype>(-1)
        && x <= static_cast<datatype>( 1)
        && y <= static_cast<datatype>( 1));
    if (basis_function == 0)
        return (direction == 0 ? (static_cast<datatype>(1) - static_cast<datatype>(2) * x) * y * (static_cast<datatype>(1) - y) / static_cast<datatype>(4) :
                                 x * (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) - static_cast<datatype>(2) * y) / static_cast<datatype>(4) );
    else if (basis_function == 1)
        return (direction == 0 ? (static_cast<datatype>(1) + static_cast<datatype>(2) * x) * y * (static_cast<datatype>(1) - y) / static_cast<datatype>(4) :
                                 x * (static_cast<datatype>(1) + x) * (static_cast<datatype>(1) - static_cast<datatype>(2) * y) / static_cast<datatype>(4) );
    else if (basis_function == 2)
        return (direction == 0 ? (static_cast<datatype>(1) + static_cast<datatype>(2) * x) * y * (static_cast<datatype>(1) + y) / static_cast<datatype>(4) :
                                 x * (static_cast<datatype>(1) + x) * (static_cast<datatype>(1) + static_cast<datatype>(2) * y) / static_cast<datatype>(4) );
    else if (basis_function == 3)
        return (direction == 0 ? (static_cast<datatype>(1) - static_cast<datatype>(2) * x) * y * (static_cast<datatype>(1) + y) / static_cast<datatype>(4) :
                                 x * (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) + static_cast<datatype>(2) * y) / static_cast<datatype>(4) );
    else if (basis_function == 4)
        return (direction == 0 ? x * y * (static_cast<datatype>(1) - y) / static_cast<datatype>(-2) :
                                 (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) + x) * (static_cast<datatype>(1) - static_cast<datatype>(2) * y) / static_cast<datatype>(-2) );
    else if (basis_function == 5)
        return (direction == 0 ? (static_cast<datatype>(1) + static_cast<datatype>(2) * x) * (static_cast<datatype>(1) - y) * (static_cast<datatype>(1) + y) / static_cast<datatype>(-2) :
                                 x * (static_cast<datatype>(1) + x) * y / static_cast<datatype>(-2) );
    else if (basis_function == 6)
        return (direction == 0 ? x * y * (static_cast<datatype>(1) + y) / static_cast<datatype>(-2) :
                                 (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) + x) * (static_cast<datatype>(1) + static_cast<datatype>(2) * y) / static_cast<datatype>(-2) );
    else if (basis_function == 7)
        return (direction == 0 ? (static_cast<datatype>(1) - static_cast<datatype>(2) * x) * (static_cast<datatype>(1) - y) * (static_cast<datatype>(1) + y) / static_cast<datatype>(-2) :
                                 x * (static_cast<datatype>(1) - x) * y / static_cast<datatype>(-2) );
    else //if (basis_function == 8)
        return (direction == 0 ? static_cast<datatype>(-2) * x * (static_cast<datatype>(1) - y*y) :
                                 static_cast<datatype>(-2) * (static_cast<datatype>(1) - x*x) * y );
}

template<typename datatype>
std::array<datatype, 2> QuadrilateralQ2<datatype>::gradient_ref(const unsigned int basis_function,
                                                                const datatype xi,
                                                                const datatype eta) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(xi >= static_cast<datatype>(-1)
        && eta >= static_cast<datatype>(-1)
        && xi <= static_cast<datatype>( 1)
        && eta <= static_cast<datatype>( 1));

    std::array<std::array<datatype, 2>, 2> B;
    B[0][0] = this->_p1.x + this->_p3.x * eta;
    B[0][1] = this->_p2.x + this->_p3.x * xi;
    B[1][0] = this->_p1.y + this->_p3.y * eta;
    B[1][1] = this->_p2.y + this->_p3.y * xi;
    datatype detB{std::abs(B[0][0] * B[1][1] - B[0][1] * B[1][0])};
    std::array<std::array<datatype, 2>, 2> B_inv_t;
    B_inv_t[0][0] =  B[1][1] / detB; B_inv_t[0][1] = -B[1][0] / detB;
    B_inv_t[1][0] = -B[0][1] / detB; B_inv_t[1][1] =  B[0][0] / detB;

    std::array<datatype, 2> grad;

    if (basis_function == 0)
    {
        grad = {(static_cast<datatype>(1) - static_cast<datatype>(2) * xi) * eta * (static_cast<datatype>(1) - eta) / static_cast<datatype>(4) ,
                xi * (static_cast<datatype>(1) - xi) * (static_cast<datatype>(1) - static_cast<datatype>(2) * eta) / static_cast<datatype>(4) };
    }
    else if (basis_function == 1)
    {
        grad = {(static_cast<datatype>(1) + static_cast<datatype>(2) * xi) * eta * (static_cast<datatype>(1) - eta) / static_cast<datatype>(4) ,
                xi * (static_cast<datatype>(1) + xi) * (static_cast<datatype>(1) - static_cast<datatype>(2) * eta) / static_cast<datatype>(4) };
    }
    else if (basis_function == 2)
    {
        grad = {(static_cast<datatype>(1) + static_cast<datatype>(2) * xi) * eta * (static_cast<datatype>(1) + eta) / static_cast<datatype>(4) ,
                xi * (static_cast<datatype>(1) + xi) * (static_cast<datatype>(1) + static_cast<datatype>(2) * eta) / static_cast<datatype>(4) };
    }
    else if (basis_function == 3)
    {
        grad = {(static_cast<datatype>(1) - static_cast<datatype>(2) * xi) * eta * (static_cast<datatype>(1) + eta) / static_cast<datatype>(4) ,
                xi * (static_cast<datatype>(1) - xi) * (static_cast<datatype>(1) + static_cast<datatype>(2) * eta) / static_cast<datatype>(4) };
    }
    else if (basis_function == 4)
    {
        //grad = {xi * eta * (static_cast<datatype>(1) - eta) / static_cast<datatype>(-2) ,
        grad = {static_cast<datatype>(-2) * xi * eta * (static_cast<datatype>(1) - eta) / static_cast<datatype>(-2) ,
                (static_cast<datatype>(1) - xi) * (static_cast<datatype>(1) + xi) * (static_cast<datatype>(1) - static_cast<datatype>(2) * eta) / static_cast<datatype>(-2) };
    }
    else if (basis_function == 5)
    {
        grad = {(static_cast<datatype>(1) + static_cast<datatype>(2) * xi) * (static_cast<datatype>(1) - eta) * (static_cast<datatype>(1) + eta) / static_cast<datatype>(-2) ,
                xi * (static_cast<datatype>(1) + xi) * static_cast<datatype>(-2) * eta / static_cast<datatype>(-2) };
                //xi * (static_cast<datatype>(1) + xi) * eta / static_cast<datatype>(-2) };
    }
    else if (basis_function == 6)
    {
        //grad = {xi * eta * (static_cast<datatype>(1) + eta) / static_cast<datatype>(-2) ,
        grad = {static_cast<datatype>(-2) * xi * eta * (static_cast<datatype>(1) + eta) / static_cast<datatype>(-2) ,
                (static_cast<datatype>(1) - xi) * (static_cast<datatype>(1) + xi) * (static_cast<datatype>(1) + static_cast<datatype>(2) * eta) / static_cast<datatype>(-2) };
    }
    else if (basis_function == 7)
    {
        grad = {(static_cast<datatype>(1) - static_cast<datatype>(2) * xi) * (static_cast<datatype>(1) - eta) * (static_cast<datatype>(1) + eta) / static_cast<datatype>(-2) ,
                xi * (static_cast<datatype>(1) - xi) * static_cast<datatype>(-2) * eta / static_cast<datatype>(-2) };
                //xi * (static_cast<datatype>(1) - xi) * eta / static_cast<datatype>(-2) };
    }
    else //if (basis_function == 8)
    {
        grad = {static_cast<datatype>(-2) * xi * (static_cast<datatype>(1) - eta*eta) ,
                static_cast<datatype>(-2) * (static_cast<datatype>(1) - xi*xi) * eta };
    }
    return {B_inv_t[0][0] * grad[0] + B_inv_t[0][1] * grad[1],
            B_inv_t[1][0] * grad[0] + B_inv_t[1][1] * grad[1]};
}
