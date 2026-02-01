#pragma once

#include <cstddef>
#include <cstdint>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Minimal C API surface for embedding.

// Opaque handle type
struct RSGPU_Batch;

// Create batch with given environment count; returns nullptr on failure.
RSGPU_Batch* rsgpu_create(int num_envs);

// Destroy batch and release device memory.
void rsgpu_destroy(RSGPU_Batch* batch);

// Set gravity (negative Z), restitution, friction, radius, max_speed.
void rsgpu_set_sim_params(RSGPU_Batch* batch, float gravity, float restitution, float friction, float radius, float max_speed);

// Set arena bounds (half_length=X, half_width=Y, ceiling=Z).
void rsgpu_set_arena(RSGPU_Batch* batch, float half_length, float half_width, float ceiling);

// Upload SoA buffers; each array length must equal num_envs.
void rsgpu_upload(RSGPU_Batch* batch,
                  const float* px,
                  const float* py,
                  const float* pz,
                  const float* vx,
                  const float* vy,
                  const float* vz);

// Download SoA buffers back to host.
void rsgpu_download(RSGPU_Batch* batch,
                    float* px,
                    float* py,
                    float* pz,
                    float* vx,
                    float* vy,
                    float* vz);

// Optional: download goal flags (1 if ball crossed goal plane) into provided buffer of length num_envs.
void rsgpu_download_goals(RSGPU_Batch* batch, uint8_t* goals);

// Expose raw device pointers for zero-copy tensor ingestion; pointers stay valid until resize/destroy.
void rsgpu_get_device_arrays(const RSGPU_Batch* batch,
                             const float** pos_x,
                             const float** pos_y,
                             const float** pos_z,
                             const float** vel_x,
                             const float** vel_y,
                             const float** vel_z,
                             const float** ang_vel_x,
                             const float** ang_vel_y,
                             const float** ang_vel_z,
                             const uint8_t** goal_mask);

// Advance physics by dt seconds.
void rsgpu_step(RSGPU_Batch* batch, float dt);

#ifdef __cplusplus
}  // extern "C"
#endif
