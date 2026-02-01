#include "arena_collision.cuh"

namespace rocketsim_gpu {

__host__ __device__ void collide_with_arena(
    float radius,
    const ArenaParams& arena,
    float restitution,
    float& px,
    float& py,
    float& pz,
    float& vx,
    float& vy,
    float& vz) {
    // Floor
    const float min_z = radius;
    if (pz < min_z) {
        pz = min_z;
        vz = -vz * restitution;
    }

    // Ceiling
    const float max_z = arena.ceiling - radius;
    if (pz > max_z) {
        pz = max_z;
        vz = -vz * restitution;
    }

    // Walls X
    const float max_x = arena.half_length - radius;
    if (px > max_x) {
        px = max_x;
        vx = -vx * restitution;
    }
    if (px < -max_x) {
        px = -max_x;
        vx = -vx * restitution;
    }

    // Walls Y
    const float max_y = arena.half_width - radius;
    if (py > max_y) {
        py = max_y;
        vy = -vy * restitution;
    }
    if (py < -max_y) {
        py = -max_y;
        vy = -vy * restitution;
    }

    // TODO: curved corners and goal volumes
}

}  // namespace rocketsim_gpu
