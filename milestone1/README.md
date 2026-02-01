# Milestone 1: Ball Physics GPU PoC

**Budget:** $800  
**Timeline:** 10 Days  
**Status:** In Progress

---

## Objective

Implement ball physics (gravity, spin, bounces, arena collisions) on GPU with batched environment stepping compatible with GigaLearnCPP. Validate throughput (~2M SPS) and basic accuracy.

---

## Reference Repositories

- **RocketSim**: https://github.com/ZealanL/RocketSim
- **GigaLearnCPP**: https://github.com/ZealanL/GigaLearnCPP-Leak

---

## Deliverables

### 1. Ball Physics Implementation (CUDA)

- [ ] Gravity simulation
- [ ] Ball spin/angular velocity
- [ ] Linear velocity and movement
- [ ] Bounce physics with restitution

### 2. Arena Collision System

- [ ] Floor collision and bounce
- [ ] Wall collisions (all arena walls)
- [ ] Ceiling collision
- [ ] Corner/curve handling (Rocket League arena shape)
- [ ] Goal area detection

### 3. Batched Environment Architecture

- [ ] Structure of Arrays (SoA) data layout for GPU efficiency
- [ ] Parallel stepping of thousands of environments
- [ ] CUDA kernel for batch physics updates
- [ ] Memory-efficient state representation

### 4. GigaLearnCPP Integration

- [ ] Study GigaLearnCPP's current RocketSim interface
- [ ] Design compatible batched state output format
- [ ] Implement tensor-ready state returns
- [ ] Ensure training loop compatibility

### 5. Performance Validation

- [ ] Benchmark SPS (Steps Per Second)
- [ ] Target: ~2M SPS (matching ZealanL's 3070 results)
- [ ] Profile and optimize bottlenecks
- [ ] Test on RTX 5070 and RTX 5090

### 6. Accuracy Validation

- [ ] Compare ball trajectories against original RocketSim
- [ ] Validate bounce angles and velocities
- [ ] Test edge cases (corners, high-speed impacts)

---

## Technical Approach

### Data Layout (SoA)

```
// Instead of Array of Structures:
// Ball balls[N];

// Use Structure of Arrays for GPU efficiency:
struct BallStatesBatched {
    float* pos_x;      // [N]
    float* pos_y;      // [N]
    float* pos_z;      // [N]
    float* vel_x;      // [N]
    float* vel_y;      // [N]
    float* vel_z;      // [N]
    float* ang_vel_x;  // [N]
    float* ang_vel_y;  // [N]
    float* ang_vel_z;  // [N]
};
```

### CUDA Kernel Structure

```
__global__ void stepBallPhysics(
    BallStatesBatched* states,
    float dt,
    int num_envs
) {
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= num_envs) return;

    // 1. Apply gravity
    // 2. Update position
    // 3. Check arena collisions
    // 4. Apply bounce with spin
}
```

### Arena Geometry

- Rocket League arena: ~4096 x 5120 x 2044 units
- Curved corners and walls
- Goal areas at each end
- Implement as analytical shapes or mesh collision

### Integration Points with GigaLearnCPP

1. Study `RocketSim::Step()` interface
2. Match state vector format
3. Batch action inputs
4. Return batched observations

---

## Files to Create

```
milestone1/
├── README.md                 # This file
├── src/
│   ├── ball_physics.cu       # CUDA ball physics kernel
│   ├── ball_physics.cuh      # Header file
│   ├── arena_collision.cu    # Arena collision detection
│   ├── arena_collision.cuh   # Header file
│   ├── batch_manager.cu      # Batched environment manager
│   ├── batch_manager.cuh     # Header file
│   └── types.cuh             # Data structures
├── include/
│   └── rocketsim_gpu.h       # Public API header
├── tests/
│   ├── test_ball_physics.cu  # Unit tests
│   └── benchmark.cu          # Performance benchmarks
├── CMakeLists.txt            # Build configuration
└── integration/
    └── gigalearn_adapter.cpp # GigaLearnCPP integration
```

---

## Success Criteria

1. ✅ Ball physics running on GPU
2. ✅ Batched stepping of 1000+ environments
3. ✅ ~2M SPS throughput achieved
4. ✅ Ball behavior matches original RocketSim
5. ✅ Compatible interface with GigaLearnCPP

---

## RocketSim Code Analysis Required

- `src/Sim/Ball/Ball.cpp` - Ball physics implementation
- `src/Sim/Arena/Arena.cpp` - Arena collision handling
- `src/Sim/BulletPhysics/` - Underlying physics engine
- GigaLearnCPP's `RocketSim` usage patterns
