#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "batch_manager.cuh"

using namespace rocketsim_gpu;

static void require(bool cond, const char* msg) {
    if (!cond) {
        std::printf("FAIL: %s\n", msg);
        std::fflush(stdout);
        std::abort();
    }
}

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("Skipping test_ball_physics: no CUDA device available\n");
        return 0;
    }

    constexpr int kN = 5;
    BatchManager mgr(kN);
    auto& sim = mgr.sim_params();
    auto& arena = mgr.arena_params();

    std::vector<float> px(kN, 0.0f), py(kN, 0.0f), pz(kN, 500.0f);
    std::vector<float> vx(kN, 0.0f), vy(kN, 0.0f), vz(kN, 0.0f);

    // Env 0: floor bounce.
    px[0] = 0.0f; py[0] = 0.0f; pz[0] = sim.radius * 0.5f;
    vx[0] = 0.0f; vy[0] = 0.0f; vz[0] = -500.0f;

    // Env 1: +X wall bounce.
    px[1] = arena.half_length - sim.radius * 0.5f;
    py[1] = 0.0f; pz[1] = 500.0f;
    vx[1] = 500.0f; vy[1] = 0.0f; vz[1] = 0.0f;

    // Env 2: rounded corner bounce (+X,+Y).
    const float inner_x = arena.half_length - arena.corner_radius;
    const float inner_y = arena.half_width - arena.corner_radius;
    px[2] = inner_x + 900.0f;  // inside the rounded region
    py[2] = inner_y + 900.0f;
    pz[2] = 500.0f;
    vx[2] = 400.0f; vy[2] = 400.0f; vz[2] = 0.0f;

    // Env 3: goal plane detection (+Y back wall).
    px[3] = 0.0f;
    py[3] = arena.half_width + arena.goal_depth + 10.0f;
    pz[3] = 200.0f;
    vx[3] = 0.0f; vy[3] = 0.0f; vz[3] = 0.0f;

    // Env 4: speed clamp.
    px[4] = 0.0f; py[4] = 0.0f; pz[4] = 500.0f;
    vx[4] = 10000.0f; vy[4] = 0.0f; vz[4] = 0.0f;

    mgr.upload(px, py, pz, vx, vy, vz);

    const float dt = 0.016f;
    mgr.step(dt);

    std::vector<BatchManager::HostState> state;
    mgr.download_state(state);

    // Floor bounce: height clamped to radius, upward velocity after bounce.
    require(state[0].pz >= sim.radius - 1e-3f, "floor clamp to radius");
    require(state[0].vz > 0.0f, "floor bounce velocity positive");

    // +X wall bounce: vx must flip sign.
    require(state[1].vx < 0.0f, "+X wall reflected");

    // Corner bounce: normal should push both components negative.
    require(state[2].vx < 0.0f && state[2].vy < 0.0f, "corner reflection on both axes");

    // Goal mask asserted when clamped to back plane in goal volume.
    require(state[3].goal == 1, "goal flag set on back-plane hit");

    // Speed clamp respected.
    const float speed = std::sqrt(state[4].vx * state[4].vx + state[4].vy * state[4].vy + state[4].vz * state[4].vz);
    require(speed <= sim.max_speed + 1e-3f, "speed clamp applied");

    std::printf("test_ball_physics OK\n");
    return 0;
}
