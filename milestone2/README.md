# Milestone 2: Car Kinematics + Boost

**Budget:** $800  
**Timeline:** 2 Weeks  
**Status:** Not Started  
**Dependencies:** Milestone 1 Complete

---

## Objective

Implement car movement, jumping, boosting, and flipping on GPU. Integrate batched states with the existing ball physics system. No car-car collisions in this milestone.

---

## Reference Repositories

- **RocketSim**: https://github.com/ZealanL/RocketSim
- **GigaLearnCPP**: https://github.com/ZealanL/GigaLearnCPP-Leak

---

## Deliverables

### 1. Car Driving Physics (CUDA)

- [ ] Throttle/acceleration
- [ ] Steering and turning
- [ ] Braking and reverse
- [ ] Ground friction model
- [ ] Handbrake/powerslide

### 2. Jump Mechanics

- [ ] Single jump
- [ ] Double jump
- [ ] Jump timing and cooldowns
- [ ] Air control after jump

### 3. Flip/Dodge System

- [ ] Front flip
- [ ] Back flip
- [ ] Side flips (left/right)
- [ ] Diagonal flips
- [ ] Flip cancels
- [ ] Dodge timing windows

### 4. Boost System

- [ ] Boost acceleration
- [ ] Boost amount tracking (0-100)
- [ ] Boost consumption rate
- [ ] Supersonic speed threshold

### 5. Aerial Control

- [ ] Pitch control
- [ ] Yaw control
- [ ] Roll control
- [ ] Air roll mechanics

### 6. Batched State Integration

- [ ] Combine car states with ball states
- [ ] Unified stepping kernel
- [ ] Action input batching
- [ ] State output batching

---

## Technical Approach

### Car State Data Layout (SoA)

```cpp
struct CarStatesBatched {
    // Position
    float* pos_x;           // [N]
    float* pos_y;           // [N]
    float* pos_z;           // [N]

    // Orientation (quaternion)
    float* quat_w;          // [N]
    float* quat_x;          // [N]
    float* quat_y;          // [N]
    float* quat_z;          // [N]

    // Linear velocity
    float* vel_x;           // [N]
    float* vel_y;           // [N]
    float* vel_z;           // [N]

    // Angular velocity
    float* ang_vel_x;       // [N]
    float* ang_vel_y;       // [N]
    float* ang_vel_z;       // [N]

    // Car-specific state
    float* boost_amount;    // [N] 0-100
    bool* on_ground;        // [N]
    bool* has_jumped;       // [N]
    bool* has_flipped;      // [N]
    float* jump_time;       // [N]
    float* flip_time;       // [N]
    bool* is_supersonic;    // [N]
};
```

### Action Input Structure

```cpp
struct CarActionsBatched {
    float* throttle;        // [N] -1 to 1
    float* steer;           // [N] -1 to 1
    float* pitch;           // [N] -1 to 1
    float* yaw;             // [N] -1 to 1
    float* roll;            // [N] -1 to 1
    bool* jump;             // [N]
    bool* boost;            // [N]
    bool* handbrake;        // [N]
};
```

### CUDA Kernel Structure

```cpp
__global__ void stepCarPhysics(
    CarStatesBatched* car_states,
    CarActionsBatched* actions,
    float dt,
    int num_envs
) {
    int env_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (env_id >= num_envs) return;

    // 1. Process ground/air state
    // 2. Apply throttle/steering (if grounded)
    // 3. Apply aerial controls (if airborne)
    // 4. Process jump/flip input
    // 5. Apply boost
    // 6. Update position and rotation
    // 7. Check ground contact
}
```

### Rocket League Car Physics Constants

```cpp
// These need to be extracted from RocketSim
#define MAX_CAR_SPEED 2300.0f
#define SUPERSONIC_THRESHOLD 2200.0f
#define BOOST_CONSUMPTION_RATE 33.3f  // per second
#define BOOST_ACCELERATION 991.667f
#define JUMP_FORCE 292.0f
#define FLIP_TORQUE 78.0f
#define GRAVITY 650.0f
```

---

## Files to Create

```
milestone2/
├── README.md                    # This file
├── src/
│   ├── car_physics.cu           # Car movement CUDA kernel
│   ├── car_physics.cuh          # Header file
│   ├── car_controls.cu          # Jump, flip, boost logic
│   ├── car_controls.cuh         # Header file
│   ├── car_aerial.cu            # Aerial control physics
│   ├── car_aerial.cuh           # Header file
│   └── unified_step.cu          # Combined ball + car stepping
├── include/
│   └── car_types.cuh            # Car data structures
├── tests/
│   ├── test_car_driving.cu      # Driving physics tests
│   ├── test_car_jump.cu         # Jump/flip tests
│   ├── test_car_boost.cu        # Boost system tests
│   └── test_aerial.cu           # Aerial control tests
└── validation/
    └── compare_rocketsim.py     # Accuracy comparison script
```

---

## RocketSim Code Analysis Required

- `src/Sim/Car/Car.cpp` - Car physics implementation
- `src/Sim/Car/CarControls.cpp` - Input handling
- `src/Sim/CarConfig/` - Car hitbox configurations
- `src/Sim/Car/CarState.h` - State definitions

---

## Key Challenges

### 1. Quaternion Math on GPU

- Efficient quaternion rotation
- Avoiding gimbal lock
- Batch quaternion operations

### 2. Ground Detection

- Ray casting from car to arena surface
- Handling slopes and curved surfaces
- Efficient batch ground checks

### 3. Physics Accuracy

- Matching RocketSim's exact values
- Frame-rate independent physics
- Reproducing edge cases

---

## Success Criteria

1. ✅ Car can drive, accelerate, and turn
2. ✅ Jump and double jump working
3. ✅ All flip types functional
4. ✅ Boost system working correctly
5. ✅ Aerial controls responsive
6. ✅ Car physics matches RocketSim behavior
7. ✅ Batched with ball physics from Milestone 1
8. ✅ Maintains ~2M SPS throughput

---

## Notes

- Car-ball collisions are **NOT** included in this milestone
- Car-car collisions are **NOT** included in this milestone
- Focus is purely on single-car kinematics
- Arena/wall collisions for cars will be in Milestone 3
