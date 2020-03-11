//
// Created by Niclas Laursen Brok on 2020-02-14.
//

#ifndef SWITCHINGTIMES_CPPAD_EIGEN_HPP
#define SWITCHINGTIMES_CPPAD_EIGEN_HPP

#include <pybind11/eigen.h>
#include <cppad/cppad.hpp>
#include <Eigen/Core>

// CppAD declarations
namespace CppAD {
    // AD<Base>
    template <class Base> class AD;
    // numeric_limits<Float>
    template <class Float>  class numeric_limits;
}

// Eigen NumTraits
namespace Eigen {
    template <class Base> struct NumTraits< CppAD::AD<Base> >
    {   // type that corresponds to the real part of an AD<Base> value
        typedef CppAD::AD<Base>   Real;
        // type for AD<Base> operations that result in non-integer values
        typedef CppAD::AD<Base>   NonInteger;
        //  type to use for numeric literals such as "2" or "0.5".
        typedef CppAD::AD<Base>   Literal;
        // type for nested value inside an AD<Base> expression tree
        typedef CppAD::AD<Base>   Nested;

        enum {
            // does not support complex Base types
                    IsComplex             = 0 ,
            // does not support integer Base types
                    IsInteger             = 0 ,
            // only support signed Base types
                    IsSigned              = 1 ,
            // must initialize an AD<Base> object
                    RequireInitialization = 1 ,
            // computational cost of the corresponding operations
                    ReadCost              = 1 ,
            AddCost               = 2 ,
            MulCost               = 2
        };

        // machine epsilon with type of real part of x
        // (use assumption that Base is not complex)
        static CppAD::AD<Base> epsilon()
        {   return CppAD::numeric_limits< CppAD::AD<Base> >::epsilon(); }

        // relaxed version of machine epsilon for comparison of different
        // operations that should result in the same value
        static CppAD::AD<Base> dummy_precision()
        {   return 100. *
                   CppAD::numeric_limits< CppAD::AD<Base> >::epsilon();
        }

        // minimum normalized positive value
        static CppAD::AD<Base> lowest()
        {   return CppAD::numeric_limits< CppAD::AD<Base> >::min(); }

        // maximum finite value
        static CppAD::AD<Base> highest()
        {   return CppAD::numeric_limits< CppAD::AD<Base> >::max(); }

        // number of decimal digits that can be represented without change.
        static int digits10()
        {   return CppAD::numeric_limits< CppAD::AD<Base> >::digits10; }
    };
}

// Eigen ScalarBinaryOpTraits
namespace Eigen {
    template<typename X, typename BinOp>
    struct ScalarBinaryOpTraits<CppAD::AD<X>,X,BinOp>
    {
        typedef CppAD::AD<X> ReturnType;
    };

    template<typename X, typename BinOp>
    struct ScalarBinaryOpTraits<X,CppAD::AD<X>,BinOp>
    {
        typedef CppAD::AD<X> ReturnType;
    };
}

// CppAD namespace
namespace CppAD {
    // functions that return references
    template <class Base> const AD<Base>& conj(const AD<Base>& x)
    {   return x; }
    template <class Base> const AD<Base>& real(const AD<Base>& x)
    {   return x; }

    // functions that return values (note abs is defined by cppad.hpp)
    template <class Base> AD<Base> imag(const AD<Base>& x)
    {   return CppAD::AD<Base>(0.); }
    template <class Base> AD<Base> abs2(const AD<Base>& x)
    {   return x * x; }
}


#endif //SWITCHINGTIMES_CPPAD_EIGEN_HPP
