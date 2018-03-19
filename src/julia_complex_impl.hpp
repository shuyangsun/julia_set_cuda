
#ifndef JULIA_SET_SRC_JULIA_COMPLEX_IMPL_HPP_
#define JULIA_SET_SRC_JULIA_COMPLEX_IMPL_HPP_

#include "julia_complex.hpp"


template<typename T>
HOST_DEVICE cuComplex<T>::cuComplex(T const real, T const lateral):
    r{ real },
    l{ lateral },
    l2_norm{ real * real + lateral * lateral }
{ EMPTY_BLOCK }


template<typename T>
HOST_DEVICE T cuComplex<T>::L2Norm2() const {
    return l2_norm * l2_norm;
}


template<typename T>
HOST_DEVICE auto cuComplex<T>::operator=(cuComplex& const rhs) -> cuComplex {
    this->r = rhs.r;
    this->l = rhs.l;
    this->l2_norm = rhs.l2_norm;
    return *this;
}


template<typename T>
HOST_DEVICE auto cuComplex<T>::operator+(cuComplex& const rhs) const -> cuComplex {
    return cuComplex(r + rhs.r, l + rhs.l);
}


template<typename T>
HOST_DEVICE auto cuComplex<T>::operator*(cuComplex& const rhs) const -> cuComplex {
    return cuComplex(r * rhs.r - l * rhs.l, r * rhs.l + l * rhs.r);
}

#endif  // JULIA_SET_SRC_JULIA_COMPLEX_IMPL_HPP_
