#include "ball_physics.cuh"

#include <algorithm>
#include <cmath>

namespace rocketsim_gpu {

__device__ float clamp_mag(float v, float max_v) {
    if (v > max_v) return max_v;
    if (v < -max_v) return -max_v;
    return v;
}

__global__ void step_ball_physics_kernel(
    BallStateSOA states,
    SimulationParams sim,
    ArenaParams arena,
    float dt,
    int num_envs) {
    const int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= num_envs) return;

    float px = states.pos_x[env_id];
    float py = states.pos_y[env_id];
    float pz = states.pos_z[env_id];
    float vx = states.vel_x[env_id];
    float vy = states.vel_y[env_id];
    float vz = states.vel_z[env_id];
    float wx = states.ang_vel_x[env_id];
    float wy = states.ang_vel_y[env_id];
    float wz = states.ang_vel_z[env_id];

    // Apply gravity
    vz += sim.gravity * dt;

    // Simple linear damping to mimic friction/air drag
    const float lin_damp = fmaxf(0.0f, 1.0f - sim.friction * dt);
    vx *= lin_damp;
    vy *= lin_damp;
    vz *= lin_damp;

    // Angular damping
    const float ang_damp = fmaxf(0.0f, 1.0f - sim.angular_damping * dt);
    wx *= ang_damp;
    wy *= ang_damp;
    wz *= ang_damp;

    // Integrate position
    px += vx * dt;
    py += vy * dt;
    pz += vz * dt;

    CollisionInfo info;
    collide_with_arena(sim.radius, arena, sim.restitution, px, py, pz, vx, vy, vz, info);

    if (info.hit) {
        // Tangential slip reduction and basic spin coupling
        const float n_dot_v = vx * info.nx + vy * info.ny + vz * info.nz;
        float tx = vx - n_dot_v * info.nx;
        float ty = vy - n_dot_v * info.ny;
        float tz = vz - n_dot_v * info.nz;
        const float t_speed = sqrtf(tx * tx + ty * ty + tz * tz) + 1e-6f;

        const float slip_scale = fmaxf(0.0f, 1.0f - sim.spin_friction * dt);
        tx *= slip_scale;
        ty *= slip_scale;
        tz *= slip_scale;

        // Update linear velocity with reduced tangential component
        vx = tx + n_dot_v * info.nx;
        vy = ty + n_dot_v * info.ny;
        vz = tz + n_dot_v * info.nz;

        // Spin coupling: apply a small torque proportional to tangential slip
        const float inv_r = (sim.radius > 1e-4f) ? (1.0f / sim.radius) : 0.0f;
        const float couple = sim.spin_friction * dt * inv_r * 0.25f;
        wx += couple * (info.ny * tz - info.nz * ty);
        wy += couple * (info.nz * tx - info.nx * tz);
        wz += couple * (info.nx * ty - info.ny * tx);
    }

    // Clamp translational speed magnitude to max_speed
    const float lin_speed = sqrtf(vx * vx + vy * vy + vz * vz);
    if (lin_speed > sim.max_speed && lin_speed > 0.0f) {
        const float scale = sim.max_speed / lin_speed;
        vx *= scale;
        vy *= scale;
        vz *= scale;
    }

    const float ang_mag = sqrtf(wx * wx + wy * wy + wz * wz);
    if (ang_mag > sim.max_ang_speed && ang_mag > 0.0f) {
        const float scale = sim.max_ang_speed / ang_mag;
        wx *= scale;
        wy *= scale;
        wz *= scale;
    }

    states.pos_x[env_id] = px;
    states.pos_y[env_id] = py;
    states.pos_z[env_id] = pz;
    states.vel_x[env_id] = vx;
    states.vel_y[env_id] = vy;
    states.vel_z[env_id] = vz;
    states.ang_vel_x[env_id] = wx;
    states.ang_vel_y[env_id] = wy;
    states.ang_vel_z[env_id] = wz;

    if (states.goal_mask) {
        states.goal_mask[env_id] = info.goal ? 1 : 0;
    }
}

void step_ball_physics(DeviceBuffers& buffers, const SimulationParams& sim, const ArenaParams& arena, float dt) {
    constexpr int kThreads = 256;
    const int blocks = (buffers.count + kThreads - 1) / kThreads;
    step_ball_physics_kernel<<<blocks, kThreads>>>(buffers.states, sim, arena, dt, buffers.count);
}

}  // namespace rocketsim_gpu
