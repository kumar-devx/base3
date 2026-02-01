#include "batch_manager.cuh"
#include "types.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>

using rocketsim_gpu::BatchManager;

struct RSGPU_Batch {
    BatchManager manager;
};

extern "C" RSGPU_Batch* rsgpu_create(int num_envs) {
    try {
        auto* handle = new RSGPU_Batch();
        handle->manager.resize(num_envs);
        return handle;
    } catch (...) {
        return nullptr;
    }
}

extern "C" void rsgpu_destroy(RSGPU_Batch* batch) { delete batch; }

extern "C" void rsgpu_set_sim_params(RSGPU_Batch* batch, float gravity, float restitution, float friction, float radius, float max_speed) {
    if (!batch) return;
    auto& sim = batch->manager.sim_params();
    sim.gravity = gravity;
    sim.restitution = restitution;
    sim.friction = friction;
    sim.radius = radius;
    sim.max_speed = max_speed;
}

extern "C" void rsgpu_set_arena(RSGPU_Batch* batch, float half_length, float half_width, float ceiling) {
    if (!batch) return;
    auto& a = batch->manager.arena_params();
    a.half_length = half_length;
    a.half_width = half_width;
    a.ceiling = ceiling;
}

extern "C" void rsgpu_upload(RSGPU_Batch* batch,
                              const float* px,
                              const float* py,
                              const float* pz,
                              const float* vx,
                              const float* vy,
                              const float* vz) {
    if (!batch) return;
    const int n = batch->manager.size();
    std::vector<float> hpx(px, px + n);
    std::vector<float> hpy(py, py + n);
    std::vector<float> hpz(pz, pz + n);
    std::vector<float> hvx(vx, vx + n);
    std::vector<float> hvy(vy, vy + n);
    std::vector<float> hvz(vz, vz + n);
    batch->manager.upload(hpx, hpy, hpz, hvx, hvy, hvz);
}

extern "C" void rsgpu_download(RSGPU_Batch* batch,
                                float* px,
                                float* py,
                                float* pz,
                                float* vx,
                                float* vy,
                                float* vz) {
    if (!batch) return;
    const int n = batch->manager.size();
    std::vector<float> hpx(n), hpy(n), hpz(n), hvx(n), hvy(n), hvz(n);
    batch->manager.download(hpx, hpy, hpz, hvx, hvy, hvz);
    std::copy(hpx.begin(), hpx.end(), px);
    std::copy(hpy.begin(), hpy.end(), py);
    std::copy(hpz.begin(), hpz.end(), pz);
    std::copy(hvx.begin(), hvx.end(), vx);
    std::copy(hvy.begin(), hvy.end(), vy);
    std::copy(hvz.begin(), hvz.end(), vz);
}

extern "C" void rsgpu_download_goals(RSGPU_Batch* batch, uint8_t* goals) {
    if (!batch || !goals) return;
    const auto& buf = batch->manager.buffers();
    const int n = buf.count;
    if (n <= 0 || !buf.states.goal_mask) return;
    cudaMemcpy(goals, buf.states.goal_mask, sizeof(uint8_t) * n, cudaMemcpyDeviceToHost);
}

extern "C" void rsgpu_get_device_arrays(const RSGPU_Batch* batch,
                             const float** pos_x,
                             const float** pos_y,
                             const float** pos_z,
                             const float** vel_x,
                             const float** vel_y,
                             const float** vel_z,
                             const float** ang_vel_x,
                             const float** ang_vel_y,
                             const float** ang_vel_z,
                             const uint8_t** goal_mask) {
    if (!batch) return;
    const auto& s = batch->manager.buffers().states;
    if (pos_x) *pos_x = s.pos_x;
    if (pos_y) *pos_y = s.pos_y;
    if (pos_z) *pos_z = s.pos_z;
    if (vel_x) *vel_x = s.vel_x;
    if (vel_y) *vel_y = s.vel_y;
    if (vel_z) *vel_z = s.vel_z;
    if (ang_vel_x) *ang_vel_x = s.ang_vel_x;
    if (ang_vel_y) *ang_vel_y = s.ang_vel_y;
    if (ang_vel_z) *ang_vel_z = s.ang_vel_z;
    if (goal_mask) *goal_mask = s.goal_mask;
}

extern "C" void rsgpu_step(RSGPU_Batch* batch, float dt) {
    if (!batch) return;
    batch->manager.step(dt);
}
