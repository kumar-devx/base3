# Milestone 3: Full Collision System

**Budget:** $700  
**Timeline:** 2-3 Weeks  
**Status:** Not Started  
**Dependencies:** Milestone 1 & 2 Complete

---

## Objective

Implement complete collision detection and game mechanics including car-ball, car-car, wall/arena collisions, demolitions, goal detection, and boost pads. Performance tune for parallel environments.

---

## Reference Repositories

- **RocketSim**: https://github.com/ZealanL/RocketSim
- **GigaLearnCPP**: https://github.com/ZealanL/GigaLearnCPP-Leak

---

## Deliverables

### 1. Car-Ball Collisions

- [ ] Hitbox detection (Octane, Dominus, etc.)
- [ ] Ball contact physics
- [ ] Ball spin transfer on hit
- [ ] Dribbling mechanics
- [ ] Power shots / clearing

### 2. Car-Car Collisions

- [ ] Car hitbox collision detection
- [ ] Bump physics
- [ ] Momentum transfer
- [ ] Demolition detection

### 3. Demolition System

- [ ] Supersonic demo conditions
- [ ] Demo state management
- [ ] Respawn logic
- [ ] Invulnerability period

### 4. Car-Arena Collisions

- [ ] Wall collision detection
- [ ] Wall driving physics
- [ ] Ceiling collision
- [ ] Corner navigation
- [ ] Goal post collisions

### 5. Goal Detection

- [ ] Ball crossing goal plane
- [ ] Goal scoring logic
- [ ] Goal reset mechanics
- [ ] Score tracking per environment

### 6. Boost Pad System

- [ ] Small boost pad locations
- [ ] Large boost pad locations
- [ ] Pickup detection
- [ ] Respawn timers (4s small, 10s large)
- [ ] Boost amount granted (12 small, 100 large)

### 7. Performance Optimization

- [ ] Broad-phase collision culling
- [ ] Spatial partitioning for efficiency
- [ ] Minimize thread divergence
- [ ] Optimize memory access patterns

---

## Technical Approach

### Car Hitbox Data

```cpp
// Standard Rocket League hitboxes
enum HitboxType {
    OCTANE,
    DOMINUS,
    PLANK,
    BREAKOUT,
    HYBRID,
    MERC
};

struct Hitbox {
    float length;  // X
    float width;   // Y
    float height;  // Z
    float3 offset; // Center offset
};

// Octane example
const Hitbox OCTANE_HITBOX = {
    .length = 118.01f,
    .width = 84.20f,
    .height = 36.16f,
    .offset = {13.88f, 0.0f, 20.75f}
};
```

### Collision Detection Strategy

```cpp
// Broad phase: AABB overlap test
__device__ bool checkAABBOverlap(
    float3 pos1, float3 size1,
    float3 pos2, float3 size2
);

// Narrow phase: OBB vs Sphere (car vs ball)
__device__ bool checkOBBSphereCollision(
    float3 car_pos, float4 car_quat, Hitbox hitbox,
    float3 ball_pos, float ball_radius,
    float3* contact_point, float3* contact_normal
);

// Narrow phase: OBB vs OBB (car vs car)
__device__ bool checkOBBOBBCollision(
    float3 pos1, float4 quat1, Hitbox hitbox1,
    float3 pos2, float4 quat2, Hitbox hitbox2,
    float3* contact_point, float3* contact_normal
);
```

### Demolition Logic

```cpp
__device__ void checkDemolition(
    CarState* car1, CarState* car2,
    float3 relative_velocity,
    bool collision_occurred
) {
    float rel_speed = length(relative_velocity);

    // Demo threshold ~2200 uu/s
    if (rel_speed >= DEMO_THRESHOLD) {
        // Car moving faster gets the demo
        if (car1->is_supersonic && !car2->is_supersonic) {
            car2->is_demoed = true;
            car2->respawn_timer = RESPAWN_TIME;
        }
        // ... handle other cases
    }
}
```

### Boost Pad Positions

```cpp
// Rocket League has 34 boost pads total
// 6 large (100 boost), 28 small (12 boost)
struct BoostPad {
    float3 position;
    bool is_large;
    float respawn_time;  // 4s small, 10s large
    float boost_amount;  // 12 or 100
};

struct BoostPadStatesBatched {
    float* respawn_timers;  // [N * 34]
    bool* is_active;        // [N * 34]
};
```

### Goal Detection

```cpp
// Goal positions in Rocket League
#define GOAL_Y_POSITIVE 5120.0f
#define GOAL_Y_NEGATIVE -5120.0f
#define GOAL_WIDTH 892.755f
#define GOAL_HEIGHT 642.775f

__device__ int checkGoal(float3 ball_pos) {
    if (ball_pos.y > GOAL_Y_POSITIVE) {
        if (abs(ball_pos.x) < GOAL_WIDTH/2 &&
            ball_pos.z < GOAL_HEIGHT) {
            return 1;  // Blue team scored
        }
    }
    if (ball_pos.y < GOAL_Y_NEGATIVE) {
        // ... Orange team scored
        return -1;
    }
    return 0;  // No goal
}
```

---

## Files to Create

```
milestone3/
├── README.md                        # This file
├── src/
│   ├── collision/
│   │   ├── car_ball_collision.cu    # Car-ball collision
│   │   ├── car_car_collision.cu     # Car-car collision
│   │   ├── car_arena_collision.cu   # Wall/arena collision
│   │   ├── collision_utils.cu       # Shared collision math
│   │   └── collision_types.cuh      # Collision data structures
│   ├── mechanics/
│   │   ├── demolition.cu            # Demo system
│   │   ├── goal_detection.cu        # Goal scoring
│   │   ├── boost_pads.cu            # Boost pad system
│   │   └── respawn.cu               # Respawn logic
│   └── spatial/
│       ├── broad_phase.cu           # Broad phase culling
│       └── spatial_hash.cu          # Spatial partitioning
├── include/
│   ├── hitboxes.cuh                 # Car hitbox definitions
│   └── arena_geometry.cuh           # Arena collision mesh
├── tests/
│   ├── test_car_ball.cu             # Car-ball collision tests
│   ├── test_car_car.cu              # Car-car collision tests
│   ├── test_demos.cu                # Demolition tests
│   ├── test_goals.cu                # Goal detection tests
│   └── test_boost_pads.cu           # Boost pad tests
└── data/
    ├── boost_pad_locations.json     # Boost pad coordinates
    └── arena_mesh.bin               # Arena collision mesh
```

---

## RocketSim Code Analysis Required

- `src/Sim/Car/Car.cpp` - Car collision handling
- `src/Sim/Ball/Ball.cpp` - Ball collision response
- `src/Sim/Arena/Arena.cpp` - Arena collision mesh
- `src/Sim/BulletPhysics/` - Physics collision system
- `src/Sim/BoostPad/` - Boost pad implementation

---

## Key Challenges

### 1. Efficient Parallel Collision Detection

- N cars \* M cars = O(N²) per environment
- Need broad-phase culling to reduce checks
- Spatial hashing or grid-based approach

### 2. Arena Collision Mesh

- Rocket League arena has curved surfaces
- Need efficient GPU-friendly representation
- Consider signed distance fields (SDF)

### 3. Physics Accuracy

- Contact point calculation
- Impulse response matching RocketSim
- Spin transfer on car-ball contact

### 4. Multiple Cars per Environment

- Support 1v1, 2v2, 3v3 configurations
- Variable number of cars per environment
- Efficient memory layout for variable counts

---

## Performance Considerations

### Thread Organization

```cpp
// For car-car collisions with N cars per env
// Use N*(N-1)/2 threads per environment for pair checks
// Or use spatial partitioning to reduce pairs

dim3 blocks((num_envs + 255) / 256);
dim3 threads(256);
```

### Memory Coalescing

- Align collision results for coalesced writes
- Use shared memory for frequently accessed data
- Minimize global memory transactions

---

## Success Criteria

1. ✅ Car-ball collisions accurate
2. ✅ Car-car bumps working
3. ✅ Demolitions triggering correctly
4. ✅ Cars can drive on walls/ceiling
5. ✅ Goals detected and scored
6. ✅ Boost pads functional
7. ✅ Maintains target SPS with collisions enabled
8. ✅ Multi-car environments supported (up to 6 cars)
