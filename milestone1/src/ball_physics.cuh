#pragma once

#include "arena_collision.cuh"
#include "types.cuh"

namespace rocketsim_gpu {

// Kernel to integrate ball physics for all environments.
__global__ void step_ball_physics_kernel(
    BallStateSOA states,
    SimulationParams sim,
    ArenaParams arena,
    float dt,
    int num_envs);

// Convenience host wrapper to launch the kernel.
void step_ball_physics(DeviceBuffers& buffers, const SimulationParams& sim, const ArenaParams& arena, float dt);

}  // namespace rocketsim_gpu
