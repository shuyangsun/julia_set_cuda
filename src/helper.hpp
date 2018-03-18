
#include <iostream>
#include <string>
#include "cuda_runtime.h"

static void HandleCUDAError(cudaError_t const err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::string err_msg{ cudaGetErrorString(err) };
        err_msg += " in ";
        err_msg += std::string{ file };
        err_msg += " at line ";
        err_msg += std::to_string(line);
        err_msg += "\n";
        std::cerr << err_msg << std::endl;
        throw std::runtime_error(err_msg);
    }
}

#define HANDLE_CUDA_ERROR( err ) (HandleCUDAError( err, __FILE__, __LINE__ ))

#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
  #define HOST_DEVICE
#endif
