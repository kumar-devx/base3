#pragma once

#include <cstdint>
#include <vector>

#include "ball_physics.cuh"
#include "types.cuh"

namespace rocketsim_gpu {

// Manages device buffers for a batch of ball states and drives stepping.
class BatchManager {
  public:
    BatchManager() = default;
    explicit BatchManager(int count);
    ~BatchManager();

    BatchManager(const BatchManager&) = delete;
    BatchManager& operator=(const BatchManager&) = delete;

    BatchManager(BatchManager&&) noexcept;
    BatchManager& operator=(BatchManager&&) noexcept;

    // Allocate buffers for num_envs environments; existing buffers are freed.
    void resize(int num_envs);

    // Copy host positions/velocities into device buffers; expects length == count.
    void upload(const std::vector<float>& px,
                const std::vector<float>& py,
                const std::vector<float>& pz,
                const std::vector<float>& vx,
                const std::vector<float>& vy,
                const std::vector<float>& vz);

    // Copy device data back to host buffers.
    void download(std::vector<float>& px,
                  std::vector<float>& py,
                  std::vector<float>& pz,
                  std::vector<float>& vx,
                  std::vector<float>& vy,
                  std::vector<float>& vz) const;

    struct HostState {
      float px, py, pz;
      float vx, vy, vz;
      float wx, wy, wz;
      uint8_t goal;
    };

    // Copy full state into host vector; resizes out to match batch size.
    void download_state(std::vector<HostState>& out) const;

    // Run a single physics step.
    void step(float dt);

    int size() const { return buffers_.count; }
    DeviceBuffers& buffers() { return buffers_; }
    const DeviceBuffers& buffers() const { return buffers_; }

    SimulationParams& sim_params() { return sim_; }
    const SimulationParams& sim_params() const { return sim_; }

    ArenaParams& arena_params() { return arena_; }
    const ArenaParams& arena_params() const { return arena_; }

  private:
    void free_device();
    void alloc_device();

    DeviceBuffers buffers_{};
    SimulationParams sim_{};
    ArenaParams arena_{};
};

}  // namespace rocketsim_gpu
