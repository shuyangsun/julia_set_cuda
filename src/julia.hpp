
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
void GPUJulia(
    CPUBitmap& bitmap,
    int const width, int const height,
    T const x_scale, T const y_scale,
    cuComplex<T>& const julia_constant,
    unsigned int const block_xy
);

template<typename T>
__global__ void GPUJuliaKernel(
    unsigned char * const ptr,
    int const width, int const height,
    T const x_scale, T const y_scale,
    T const julia_constant_r, T const julia_constant_l
);

HOST_DEVICE void OffsetBitmap(unsigned char * const ptr, int const offset, unsigned char const is_julia);

template<typename T>
HOST_DEVICE unsigned char IsJulia(
    int const x, int const y,
    int const width, int const height,
    T const x_scale, T const y_scale,
    cuComplex<T>& const julia_constant
);


#include "julia_impl.hpp"

#endif  // JULIA_SET_SRC_JULIA_HPP_
