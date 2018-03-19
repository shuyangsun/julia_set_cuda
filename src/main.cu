
#include "common/cpu_bitmap.h"

#include "julia.hpp"
#include "julia_complex.hpp"

#include <iostream>
#include <ctime>

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
    printf("GPU duration: %.2fs.\n", gpu_duration);

    bitmap.display_and_exit();

    return 0;
}
