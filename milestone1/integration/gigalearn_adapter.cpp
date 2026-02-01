#include "rocketsim_gpu.h"

#include <vector>

// Thin adapter to expose device views and host upload/download helpers for GigaLearnCPP.
namespace gigalearn_adapter {

struct DeviceViews {
    const float* pos_x = nullptr;
    const float* pos_y = nullptr;
    const float* pos_z = nullptr;
    const float* vel_x = nullptr;
    const float* vel_y = nullptr;
    const float* vel_z = nullptr;
    const float* ang_vel_x = nullptr;
    const float* ang_vel_y = nullptr;
    const float* ang_vel_z = nullptr;
    const uint8_t* goal_mask = nullptr;
};

struct AdapterState {
    RSGPU_Batch* batch = nullptr;
    int count = 0;
    DeviceViews views{};
};

static void refresh_views(AdapterState& s) {
    if (!s.batch) return;
    rsgpu_get_device_arrays(
        s.batch,
        &s.views.pos_x,
        &s.views.pos_y,
        &s.views.pos_z,
        &s.views.vel_x,
        &s.views.vel_y,
        &s.views.vel_z,
        &s.views.ang_vel_x,
        &s.views.ang_vel_y,
        &s.views.ang_vel_z,
        &s.views.goal_mask);
}

AdapterState create(int num_envs) {
    AdapterState s;
    s.batch = rsgpu_create(num_envs);
    s.count = num_envs;
    refresh_views(s);
    return s;
}

void destroy(AdapterState& s) {
    rsgpu_destroy(s.batch);
    s = {};
}

void upload(AdapterState& s,
            const std::vector<float>& px,
            const std::vector<float>& py,
            const std::vector<float>& pz,
            const std::vector<float>& vx,
            const std::vector<float>& vy,
            const std::vector<float>& vz) {
    if (!s.batch) return;
    rsgpu_upload(s.batch, px.data(), py.data(), pz.data(), vx.data(), vy.data(), vz.data());
}

void download(AdapterState& s,
              std::vector<float>& px,
              std::vector<float>& py,
              std::vector<float>& pz,
              std::vector<float>& vx,
              std::vector<float>& vy,
              std::vector<float>& vz) {
    if (!s.batch) return;
    if (static_cast<int>(px.size()) != s.count) {
        px.resize(s.count); py.resize(s.count); pz.resize(s.count);
        vx.resize(s.count); vy.resize(s.count); vz.resize(s.count);
    }
    rsgpu_download(s.batch, px.data(), py.data(), pz.data(), vx.data(), vy.data(), vz.data());
}

DeviceViews device_views(AdapterState& s) {
    refresh_views(s);
    return s.views;
}

void step(AdapterState& s, float dt) {
    if (!s.batch) return;
    rsgpu_step(s.batch, dt);
}

}  // namespace gigalearn_adapter
