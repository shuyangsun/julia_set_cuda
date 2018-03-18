
#include "cuda_runtime.h"
#include "helper.hpp"
#include "common/cpu_bitmap.h"

#include <iostream>
#include <ctime>


template<typename T>
class cuComplex {
public:
    T r;
    T l;
    T l2_norm;

    HOST_DEVICE cuComplex(T const real, T const lateral) :
        r{ real },
        l{ lateral },
        l2_norm{ real * real + lateral * lateral }
    {}

    HOST_DEVICE T L2Norm2() const { return l2_norm * l2_norm; }

    HOST_DEVICE cuComplex operator=(cuComplex& const rhs) {
        this->r = rhs.r;
        this->l = rhs.l;
        this->l2_norm = rhs.l2_norm;
        return *this;
    }

    HOST_DEVICE cuComplex operator+(cuComplex& const rhs) const {
        return cuComplex(r + rhs.r, l + rhs.l);
    }

    HOST_DEVICE cuComplex operator*(cuComplex& const rhs) const {
        return cuComplex(
            r * rhs.r - l * rhs.l,
            r * rhs.l + l * rhs.r
        );
    }
};

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

int main(int const argc, char ** const argv) {
    
    using DType = typename double;

    constexpr int width{ 1920 };
    constexpr int height{ 1080 };
    constexpr DType zoom_scale{ 1.0 };
    constexpr DType x_scale{ 1.6 / zoom_scale };
    constexpr DType y_scale{ 0.9 / zoom_scale };
    auto julia_constant = cuComplex<DType>(-0.8, 0.156);
    constexpr unsigned int block_xy = 32;
    constexpr bool should_time_cpu{ false };
    
    CPUBitmap bitmap{ width, height };

    if (should_time_cpu) {
        std::cout << "Generating Julia Set visualization on CPU..." << std::endl;
        clock_t const cpu_start = clock();
        CPUJulia<DType>(
            bitmap,
            width, height,
            x_scale, y_scale,
            julia_constant
            );
        double const cpu_duration = double(clock() - cpu_start) / CLOCKS_PER_SEC;
        printf("CPU duration: %.2fs.\n\n", cpu_duration);
    }
    
    std::cout << "Generating Julia Set visualization on GPU..." << std::endl;
    clock_t const gpu_start = clock();
    GPUJulia<DType>(
        bitmap,
        width, height,
        x_scale, y_scale,
        julia_constant,
        block_xy
    );
    double const gpu_duration = double(clock() - gpu_start) / CLOCKS_PER_SEC;
    printf("GPU duration: %.2fs.\n\n", gpu_duration);

    bitmap.display_and_exit();

    return 0;
}

template<typename T>
void CPUJulia(
    CPUBitmap& bitmap,
    int const width, int const height,
    T const x_scale, T const y_scale,
    cuComplex<T>& const julia_constant
) {
    for (int y{0}; y < height; ++y) {
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
    GPUJuliaKernel<T><<<grid_dim, block_dim>>>(
        d_bitmap, width, height, x_scale, y_scale, julia_constant.r, julia_constant.l
    );
    
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


