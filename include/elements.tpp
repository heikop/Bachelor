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
        return {static_cast<datatype>(1),
                static_cast<datatype>(0)};
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
                                                 const datatype x,
                                                 const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(-1)
        && y >= static_cast<datatype>(-1)
        && x <= static_cast<datatype>( 1)
        && y <= static_cast<datatype>( 1));
    if (basis_function == 0)
        return (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) - y) / static_cast<datatype>(4);
    else if (basis_function == 1)
        return (static_cast<datatype>(1) + x) * (static_cast<datatype>(1) - y) / static_cast<datatype>(4);
    else if (basis_function == 2)
        return (static_cast<datatype>(1) + x) * (static_cast<datatype>(1) + y) / static_cast<datatype>(4);
    else //if (basis_function == 3)
        return (static_cast<datatype>(1) - x) * (static_cast<datatype>(1) + y) / static_cast<datatype>(4);
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
                                                                const datatype x,
                                                                const datatype y) const
{
    assert(basis_function >= 0 && basis_function <= this->num_basis_functions());
    assert(x >= static_cast<datatype>(-1)
        && y >= static_cast<datatype>(-1)
        && x <= static_cast<datatype>( 1)
        && y <= static_cast<datatype>( 1));
    if (basis_function == 0)
        return {(static_cast<datatype>(1) - y) / static_cast<datatype>(-4) ,
                (static_cast<datatype>(1) - x) / static_cast<datatype>(-4) };
    else if (basis_function == 1)
        return {(static_cast<datatype>(1) - y) / static_cast<datatype>( 4) ,
                (static_cast<datatype>(1) + x) / static_cast<datatype>(-4) };
    else if (basis_function == 2)
        return {(static_cast<datatype>(1) + y) / static_cast<datatype>( 4) ,
                (static_cast<datatype>(1) + x) / static_cast<datatype>( 4) };
    else //if (basis_function == 3)
        return {(static_cast<datatype>(1) + y) / static_cast<datatype>(-4) ,
                (static_cast<datatype>(1) - x) / static_cast<datatype>( 4) };
}

// ***** // ***** QuadrilateralQ2 ***** // ***** //
// TODO
