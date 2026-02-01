#pragma once

#include <math.h>

#include "types.cuh"

namespace rocketsim_gpu {

struct CollisionInfo {
    bool hit = false;
    bool goal = false;
    float nx = 0.0f;
    float ny = 0.0f;
    float nz = 0.0f;
};

// Detect and resolve collisions against simplified box arena.
// Inline so device code can link without separate compilation units.
__host__ __device__ __forceinline__ void collide_with_arena(
    float radius,
    const ArenaParams& arena,
    float restitution,
    float& px,
    float& py,
    float& pz,
    float& vx,
    float& vy,
    float& vz,
    CollisionInfo& info) {
    info.hit = false;
    info.goal = false;

    auto reflect = [&](float nx, float ny, float nz) {
        const float vn = vx * nx + vy * ny + vz * nz;
        if (vn >= 0.0f) return;
        const float impulse = (1.0f + restitution) * vn;
        vx -= impulse * nx;
        vy -= impulse * ny;
        vz -= impulse * nz;
        info.hit = true;
        info.nx = nx;
        info.ny = ny;
        info.nz = nz;
    };

    // Floor
    const float min_z = radius;
    if (pz < min_z) {
        pz = min_z;
        reflect(0.0f, 0.0f, 1.0f);
    }

    // Ceiling
    const float max_z = arena.ceiling - radius;
    if (pz > max_z) {
        pz = max_z;
        reflect(0.0f, 0.0f, -1.0f);
    }

    // Side walls and curved corners (2D fillets in XY plane)
    const float inner_x = arena.half_length - arena.corner_radius;
    const float inner_y = arena.half_width - arena.corner_radius;
    const float corner_r = arena.corner_radius - radius;

    const bool past_corner_x = fabsf(px) > inner_x;
    const bool past_corner_y = fabsf(py) > inner_y;

    if (past_corner_x && past_corner_y) {
        const float cx = (px > 0.0f ? inner_x : -inner_x);
        const float cy = (py > 0.0f ? inner_y : -inner_y);
        float dx = px - cx;
        float dy = py - cy;
        const float dist = sqrtf(dx * dx + dy * dy);
        if (dist > corner_r && dist > 1e-6f) {
            const float inv = corner_r / dist;
            px = cx + dx * inv;
            py = cy + dy * inv;
            // Interior normal points back toward arena center so outward-moving balls reflect
            const float nx = -dx / dist;
            const float ny = -dy / dist;
            reflect(nx, ny, 0.0f);
        }
    } else {
        // Planar walls when not in the rounded region.
        const float max_x = arena.half_length - radius;
        if (px > max_x) {
            px = max_x;
            reflect(-1.0f, 0.0f, 0.0f);
        }
        if (px < -max_x) {
            px = -max_x;
            reflect(1.0f, 0.0f, 0.0f);
        }

        const float max_y = arena.half_width - radius;
        const bool in_goal_mouth = (fabsf(px) < arena.goal_width * 0.5f && pz < arena.goal_height);
        if (!in_goal_mouth) {
            if (py > max_y) {
                py = max_y;
                reflect(0.0f, -1.0f, 0.0f);
            }
            if (py < -max_y) {
                py = -max_y;
                reflect(0.0f, 1.0f, 0.0f);
            }
        }
    }

    // Goal volume back planes; only active when inside goal mouth.
    const bool in_goal_mouth = (fabsf(px) < arena.goal_width * 0.5f && pz < arena.goal_height);
    if (in_goal_mouth) {
        const float goal_plane = arena.half_width + arena.goal_depth;
        if (py > goal_plane) {
            py = goal_plane;
            reflect(0.0f, -1.0f, 0.0f);
            info.goal = true;
        } else if (py < -goal_plane) {
            py = -goal_plane;
            reflect(0.0f, 1.0f, 0.0f);
            info.goal = true;
        }
    }
}

}  // namespace rocketsim_gpu
