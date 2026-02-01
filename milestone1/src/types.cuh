#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace rocketsim_gpu {

// Device buffers laid out as Structure of Arrays for coalesced access.
struct BallStateSOA {
    float* pos_x{};
    float* pos_y{};
    float* pos_z{};
    float* vel_x{};
    float* vel_y{};
    float* vel_z{};
    float* ang_vel_x{};
    float* ang_vel_y{};
    float* ang_vel_z{};
    uint8_t* goal_mask{}; // 1 if ball crossed goal plane on last step
};

enum class GameMode : int {
    SOCCAR = 0,
    HEATSEEKER = 1,
    HOOPS = 2,
    DROPSHOT = 3,
    SNOWDAY = 4,
};

struct SimulationParams {
    float gravity = -650.0f;      // Rocket League units per second^2
    float restitution = 0.6f;     // Coefficient of restitution for walls/floor
    float friction = 0.035f;      // Linear drag/friction for translational motion
    float angular_damping = 0.03f; // Damping for angular velocity
    float spin_friction = 0.35f;   // Tangential slip reduction on bounce
    float radius = 92.75f;         // RL ball collision radius (uu)
    float max_speed = 6000.0f;     // Clamp translational speed
    float max_ang_speed = 6.0f;    // Clamp angular speed (rad/s)
    GameMode mode = GameMode::SOCCAR;
};

struct ArenaParams {
    // Axis-aligned bounds with analytic corner fillets and goal volumes.
    float half_length = 5120.0f;   // X extent
    float half_width = 4096.0f;    // Y extent
    float ceiling = 2044.0f;       // Z max
    float corner_radius = 1150.0f; // Radius for rounded corners (approx RL)
    float goal_depth = 880.0f;     // Depth beyond wall for goal volume
    float goal_width = 1800.0f;    // Goal mouth width (centered on X)
    float goal_height = 640.0f;    // Height below which goal counts
};

struct DeviceBuffers {
    BallStateSOA states{};
    int count = 0;
};

}  // namespace rocketsim_gpu
