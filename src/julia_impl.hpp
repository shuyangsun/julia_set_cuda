
#ifndef JULIA_SET_SRC_JULIA_IMPL_HPP_
#define JULIA_SET_SRC_JULIA_IMPL_HPP_

#include "julia.hpp"

template<typename T>
__global__ void GPUJuliaKernel(
    unsigned char * const ptr,
    int const width, int const height,
    T const x_scale, T const y_scale,
    T const julia_constant_r, T const julia_constant_l
);


template<typename T>
__global__ void GPUMandelbrotKernel(
    unsigned char * const ptr,
    int const width, int const height,
    T const x_scale, T const y_scale
);


template<typename T>
HOST_DEVICE unsigned char IsJulia(
    int const x, int const y,
    int const width, int const height,
    T const x_scale, T const y_scale,
    cuComplex<T>& const julia_constant
);


template<typename T>
HOST_DEVICE unsigned char IsMandelbrot(
    int const x, int const y,
    int const width, int const height,
    T const x_scale, T const y_scale
);


HOST_DEVICE void OffsetBitmap(unsigned char * const ptr, int const offset, unsigned char const is_julia);


template<typename T>
void CPUJulia(
    CPUBitmap& bitmap,
    int const width, int const height,
    T const x_scale, T const y_scale,
    cuComplex<T>& const julia_constant
) {
    for (int y{ 0 }; y < height; ++y) {
        for (int x{ 0 }; x < width; ++x) {
            int offset = x + y * width;
            int is_julia = IsJulia(x, y, width, height, x_scale, y_scale, julia_constant) ? 1 : 0;

            OffsetBitmap(bitmap.get_ptr(), offset, is_julia);
        }
    }
}


template<typename T>
void GPUJulia(
    CPUBitmap& bitmap,
    int const width, int const height,
    T const x_scale, T const y_scale,
    cuComplex<T>& const julia_constant,
    unsigned int const block_xy
) {
    unsigned char *d_bitmap{};

    HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_bitmap), bitmap.image_size()));

    dim3 grid_dim((width + block_xy - 1) / block_xy, (height + block_xy - 1) / block_xy);
    dim3 block_dim{ block_xy, block_xy };
    GPUJuliaKernel<T> << <grid_dim, block_dim >> >(
        d_bitmap, width, height, x_scale, y_scale, julia_constant.r, julia_constant.l
        );

    cudaDeviceSynchronize();
    HANDLE_CUDA_ERROR(cudaGetLastError());

    HANDLE_CUDA_ERROR(cudaMemcpy(bitmap.get_ptr(), d_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaFree(d_bitmap));
    HANDLE_CUDA_ERROR(cudaDeviceReset());
}


template<typename T>
void GPUMandelbrot(
    CPUBitmap& bitmap,
    int const width, int const height,
    T const x_scale, T const y_scale,
    unsigned int const block_xy
) {
    unsigned char *d_bitmap{};

    HANDLE_CUDA_ERROR(cudaMalloc(reinterpret_cast<void **>(&d_bitmap), bitmap.image_size()));

    dim3 grid_dim((width + block_xy - 1) / block_xy, (height + block_xy - 1) / block_xy);
    dim3 block_dim{ block_xy, block_xy };
    GPUMandelbrotKernel<T> << <grid_dim, block_dim >> >(d_bitmap, width, height, x_scale, y_scale);

    cudaDeviceSynchronize();
    HANDLE_CUDA_ERROR(cudaGetLastError());

    HANDLE_CUDA_ERROR(cudaMemcpy(bitmap.get_ptr(), d_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaFree(d_bitmap));
    HANDLE_CUDA_ERROR(cudaDeviceReset());
}


template<typename T>
__global__ void GPUJuliaKernel(
    unsigned char * const ptr,
    int const width, int const height,
    T const x_scale, T const y_scale,
    T const julia_constant_r, T const julia_constant_l
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int offset = x + y * width;
    if (offset < width * height) {
        int is_julia = IsJulia(
            x, y, width, height, x_scale, y_scale, cuComplex<T>(julia_constant_r, julia_constant_l)
        ) ? 1 : 0;
        OffsetBitmap(ptr, offset, is_julia);
    }
}


template<typename T>
__global__ void GPUMandelbrotKernel(
    unsigned char * const ptr,
    int const width, int const height,
    T const x_scale, T const y_scale
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int offset = x + y * width;
    if (offset < width * height) {
        int is_mandelbrot = IsMandelbrot(x, y, width, height, x_scale, y_scale) ? 1 : 0;
        OffsetBitmap(ptr, offset, is_mandelbrot);
    }
}


HOST_DEVICE void OffsetBitmap(unsigned char * const ptr, int const offset, unsigned char const is_julia) {
    ptr[offset * 4] = 33 * is_julia;
    ptr[offset * 4 + 1] = 150 * is_julia;
    ptr[offset * 4 + 2] = 243 * is_julia;
    ptr[offset * 4 + 3] = 255;
}


template<typename T>
HOST_DEVICE unsigned char IsJulia(
    int const x, int const y,
    int const width, int const height,
    T const x_scale, T const y_scale,
    cuComplex<T>& const julia_constant
) {
    T const jx{ x_scale * (width / 2.0 - x) / (width / 2) };
    T const jy{ y_scale * (height / 2.0 - y) / (height / 2) };

    constexpr size_t julia_threshold{ 200 };
    cuComplex<T> cur = cuComplex<T>(jx, jy);
    for (size_t i{ 0 }; i < julia_threshold; ++i) {
        cur = cur * cur + julia_constant;
        if (cur.L2Norm2() >= 1000) {
            return 0;
        }
    }
    return 1;
}


template<typename T>
HOST_DEVICE unsigned char IsMandelbrot(
    int const x, int const y,
    int const width, int const height,
    T const x_scale, T const y_scale
) {
    T const jx{ x_scale * (width / 2.0 - x) / (width / 2) };
    T const jy{ y_scale * (height / 2.0 - y) / (height / 2) };

    constexpr size_t mandelbrot_threshold{ 1000 };
    cuComplex<T> cur = cuComplex<T>(0, 0);
    for (size_t i{ 0 }; i < mandelbrot_threshold; ++i) {
        cur = cur * cur + cuComplex<T>(jx, jy);
        if (cur.l2_norm > 2) {
            return 0;
        }
    }
    return 1;
}

#endif  // JULIA_SET_SRC_JULIA_IMPL_HPP_
