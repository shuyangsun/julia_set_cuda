
#ifndef JULIA_SET_SRC_JULIA_HPP_
#define JULIA_SET_SRC_JULIA_HPP_

#include "helper.hpp"
#include "julia_complex.hpp"

template<typename T>
void CPUJulia(
    CPUBitmap& bitmap,
    int const width, int const height,
    T const x_scale, T const y_scale,
    cuComplex<T>& const julia_constant
);

template<typename T>
void GPUMandelbrot(
    CPUBitmap& bitmap,
    int const width, int const height,
    T const x_scale, T const y_scale,
    cuComplex<T>& const julia_constant,
    unsigned int const block_xy
);

#include "julia_impl.hpp"

#endif  // JULIA_SET_SRC_JULIA_HPP_
