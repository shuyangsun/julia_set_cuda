
#ifndef JULIA_SET_SRC_JULIA_COMPLEX_HPP_
#define JULIA_SET_SRC_JULIA_COMPLEX_HPP_

#include "helper.hpp"

template<typename T>
class cuComplex {
public:
    T r{};
    T l{};
    T l2_norm{};

    HOST_DEVICE cuComplex(T const real, T const lateral);

    HOST_DEVICE T L2Norm2() const;
    HOST_DEVICE cuComplex operator=(cuComplex& const rhs);
    HOST_DEVICE cuComplex operator+(cuComplex& const rhs) const;
    HOST_DEVICE cuComplex operator*(cuComplex& const rhs) const;
};

#include "julia_complex_impl.hpp"

#endif  // JULIA_SET_SRC_JULIA_COMPLEX_HPP_
