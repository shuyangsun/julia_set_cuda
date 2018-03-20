
#include "common/cpu_bitmap.h"

#include "julia.hpp"
#include "julia_complex.hpp"

#include <cstdio>
#include <iostream>
#include <ctime>
#include <cstdlib>

int main(int const argc, char ** const argv) {
    
    using DType = typename double;

    constexpr int width{ 1920 };
    constexpr int height{ 1080 };
    constexpr DType zoom_scale{ 0.8 };
    constexpr DType x_scale{ 1.6 / zoom_scale };
    constexpr DType y_scale{ 0.9 / zoom_scale };
    constexpr DType julia_c_real{ -0.8 };
    constexpr DType julia_c_start{ 0.1900 };
    constexpr DType julia_c_stop{ 0.1400 };
    constexpr DType julia_c_step{ -0.0001 };
    auto julia_constant = cuComplex<DType>(julia_c_real, (julia_c_start + julia_c_stop) / 2);
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
    
    printf("Generating Mandelbrot Set visualization on GPU...\n");
    clock_t const gpu_start = clock();
    GPUMandelbrot<DType>(
        bitmap,
        width, height,
        x_scale, y_scale,
        block_xy
        );
    double const gpu_duration = double(clock() - gpu_start) / CLOCKS_PER_SEC;
    printf("GPU duration: %.2fs.\n", gpu_duration);

    bitmap.display_and_exit();
    
    return 0;
}
