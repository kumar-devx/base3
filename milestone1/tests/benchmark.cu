#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#include "batch_manager.cuh"

using namespace rocketsim_gpu;

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("Skipping benchmark: no CUDA device available\n");
        return 0;
    }

    const int num_envs = 1024 * 8;
    const int steps = 1000;
    BatchManager mgr(num_envs);

    std::vector<float> px(num_envs, 0.0f), py(num_envs, 0.0f), pz(num_envs, 500.0f);
    std::vector<float> vx(num_envs, 0.0f), vy(num_envs, 0.0f), vz(num_envs, 0.0f);
    mgr.upload(px, py, pz, vx, vy, vz);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < steps; ++i) {
        mgr.step(0.008f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    const double sps = static_cast<double>(num_envs) * steps / (ms / 1000.0);
    std::printf("Benchmark: %d envs, %d steps, %.2f ms, %.2f steps/sec\n", num_envs, steps, ms, sps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
