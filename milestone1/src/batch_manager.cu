#include "batch_manager.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace rocketsim_gpu {

namespace {

void ensure_cuda_device() {
    int count = 0;
    auto err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        throw std::runtime_error("No CUDA-capable device is detected");
    }
}

void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

void alloc_array(float** ptr, int count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(float) * count), "cudaMalloc");
}

void alloc_array_u8(uint8_t** ptr, int count) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(ptr), sizeof(uint8_t) * count), "cudaMalloc u8");
}

void free_array(float* ptr) {
    if (ptr) cudaFree(ptr);
}

void free_array_u8(uint8_t* ptr) {
    if (ptr) cudaFree(ptr);
}

void copy_h2d(float* dst, const std::vector<float>& src) {
    cuda_check(cudaMemcpy(dst, src.data(), sizeof(float) * src.size(), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
}

void copy_d2h(std::vector<float>& dst, const float* src) {
    cuda_check(cudaMemcpy(dst.data(), src, sizeof(float) * dst.size(), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}

void copy_d2h_u8(std::vector<uint8_t>& dst, const uint8_t* src) {
    cuda_check(cudaMemcpy(dst.data(), src, sizeof(uint8_t) * dst.size(), cudaMemcpyDeviceToHost), "cudaMemcpy D2H u8");
}

}  // namespace

BatchManager::BatchManager(int count) { resize(count); }
BatchManager::~BatchManager() { free_device(); }

BatchManager::BatchManager(BatchManager&& other) noexcept {
    buffers_ = other.buffers_;
    sim_ = other.sim_;
    arena_ = other.arena_;
    other.buffers_ = {};
}

BatchManager& BatchManager::operator=(BatchManager&& other) noexcept {
    if (this != &other) {
        free_device();
        buffers_ = other.buffers_;
        sim_ = other.sim_;
        arena_ = other.arena_;
        other.buffers_ = {};
    }
    return *this;
}

void BatchManager::resize(int num_envs) {
    free_device();
    buffers_.count = num_envs;
    if (num_envs <= 0) return;
    alloc_device();
}

void BatchManager::free_device() {
    free_array(buffers_.states.pos_x);
    free_array(buffers_.states.pos_y);
    free_array(buffers_.states.pos_z);
    free_array(buffers_.states.vel_x);
    free_array(buffers_.states.vel_y);
    free_array(buffers_.states.vel_z);
    free_array(buffers_.states.ang_vel_x);
    free_array(buffers_.states.ang_vel_y);
    free_array(buffers_.states.ang_vel_z);
    free_array_u8(buffers_.states.goal_mask);
    buffers_ = {};
}

void BatchManager::alloc_device() {
    ensure_cuda_device();
    const int n = buffers_.count;
    alloc_array(&buffers_.states.pos_x, n);
    alloc_array(&buffers_.states.pos_y, n);
    alloc_array(&buffers_.states.pos_z, n);
    alloc_array(&buffers_.states.vel_x, n);
    alloc_array(&buffers_.states.vel_y, n);
    alloc_array(&buffers_.states.vel_z, n);
    alloc_array(&buffers_.states.ang_vel_x, n);
    alloc_array(&buffers_.states.ang_vel_y, n);
    alloc_array(&buffers_.states.ang_vel_z, n);
    alloc_array_u8(&buffers_.states.goal_mask, n);
}

void BatchManager::upload(const std::vector<float>& px,
                          const std::vector<float>& py,
                          const std::vector<float>& pz,
                          const std::vector<float>& vx,
                          const std::vector<float>& vy,
                          const std::vector<float>& vz) {
    if (px.size() != static_cast<size_t>(buffers_.count)) throw std::invalid_argument("upload size mismatch");
    copy_h2d(buffers_.states.pos_x, px);
    copy_h2d(buffers_.states.pos_y, py);
    copy_h2d(buffers_.states.pos_z, pz);
    copy_h2d(buffers_.states.vel_x, vx);
    copy_h2d(buffers_.states.vel_y, vy);
    copy_h2d(buffers_.states.vel_z, vz);
}

void BatchManager::download(std::vector<float>& px,
                            std::vector<float>& py,
                            std::vector<float>& pz,
                            std::vector<float>& vx,
                            std::vector<float>& vy,
                            std::vector<float>& vz) const {
    if (px.size() != static_cast<size_t>(buffers_.count)) throw std::invalid_argument("download size mismatch");
    copy_d2h(px, buffers_.states.pos_x);
    copy_d2h(py, buffers_.states.pos_y);
    copy_d2h(pz, buffers_.states.pos_z);
    copy_d2h(vx, buffers_.states.vel_x);
    copy_d2h(vy, buffers_.states.vel_y);
    copy_d2h(vz, buffers_.states.vel_z);
}

void BatchManager::download_state(std::vector<HostState>& out) const {
    if (buffers_.count <= 0) {
        out.clear();
        return;
    }
    out.resize(buffers_.count);

    std::vector<float> px(buffers_.count), py(buffers_.count), pz(buffers_.count);
    std::vector<float> vx(buffers_.count), vy(buffers_.count), vz(buffers_.count);
    std::vector<float> wx(buffers_.count), wy(buffers_.count), wz(buffers_.count);
    std::vector<uint8_t> goal(buffers_.count, 0u);

    copy_d2h(px, buffers_.states.pos_x);
    copy_d2h(py, buffers_.states.pos_y);
    copy_d2h(pz, buffers_.states.pos_z);
    copy_d2h(vx, buffers_.states.vel_x);
    copy_d2h(vy, buffers_.states.vel_y);
    copy_d2h(vz, buffers_.states.vel_z);
    copy_d2h(wx, buffers_.states.ang_vel_x);
    copy_d2h(wy, buffers_.states.ang_vel_y);
    copy_d2h(wz, buffers_.states.ang_vel_z);
    if (buffers_.states.goal_mask) {
        copy_d2h_u8(goal, buffers_.states.goal_mask);
    }

    for (int i = 0; i < buffers_.count; ++i) {
        out[i] = {px[i], py[i], pz[i], vx[i], vy[i], vz[i], wx[i], wy[i], wz[i], goal[i]};
    }
}

void BatchManager::step(float dt) { step_ball_physics(buffers_, sim_, arena_, dt); }

}  // namespace rocketsim_gpu
