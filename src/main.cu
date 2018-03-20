
#include "common/cpu_bitmap.h"

#include "julia.hpp"
#include "julia_complex.hpp"

#include <cstdio>
#include <iostream>
#include <ctime>
#include <cstdlib>

int main(int const argc, char ** const argv) {
    
    using DType = typename double;

    constexpr int width{ 1920 * 2 };
    constexpr int height{ 1080 * 2 };
    constexpr DType zoom_scale{ 1.0 };
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
    
    for (DType c{ julia_c_start }; c >= julia_c_stop; c += julia_c_step) {
        printf("Generating Julia Set visualization on GPU with C = %.2f + %.3fi...\n", julia_c_real, c);
        clock_t const gpu_start = clock();
        GPUJulia<DType>(
            bitmap,
            width, height,
            x_scale, y_scale,
            cuComplex<DType>(julia_c_real, c),
            block_xy
        );
        double const gpu_duration = double(clock() - gpu_start) / CLOCKS_PER_SEC;
        printf("GPU duration: %.2fs.\n", gpu_duration);

        char file_name[128];
        sprintf(
            file_name,
            "C:\\Users\\shuyangsun\\Desktop\\julia_set_sequence\\julia_set_%d_%04d.png",
            abs(int(julia_c_real * 10)),
            abs(int((julia_c_start - c) * 10000))
        );
        bitmap.save_to_file(file_name);
        HANDLE_CUDA_ERROR(cudaDeviceReset());
        // bitmap.display_and_exit();
    }
    
    return 0;
}
