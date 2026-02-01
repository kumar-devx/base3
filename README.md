# RocketSim GPU Acceleration Project

## Project Overview

GPU-accelerated port of [RocketSim](https://github.com/ZealanL/RocketSim) for high-throughput Rocket League AI bot training, integrated with [GigaLearnCPP](https://github.com/ZealanL/GigaLearnCPP-Leak).

## Client Requirements

- Current bottleneck: 77k steps/second (target: 2M+ SPS like ZealanL achieved on a 3070)
- Hardware: Dual RTX 5090s (cloud server), RTX 5070 (personal PC), 64GB RAM
- Goal: GPU-parallel simulation with batched tensor outputs for RL training

## Budget: $2,800 Total

---

## Milestones

### Milestone 1: Ball Physics GPU PoC - $800 (10 Days)

Implement ball physics on GPU with batched environment stepping compatible with GigaLearnCPP.

**Deliverables:**

- Ball gravity, spin, and bounce physics on CUDA
- Arena collision detection (walls, floor, ceiling)
- Batched environment stepping (thousands of parallel sims)
- GigaLearnCPP-compatible interface
- Throughput validation (~2M SPS target)
- Basic accuracy validation against original RocketSim

### Milestone 2: Car Kinematics + Boost - $800 (2 Weeks)

Implement car movement mechanics on GPU.

**Deliverables:**

- Car driving physics (acceleration, turning, braking)
- Jumping mechanics
- Boosting system
- Flipping/dodging mechanics
- Batched state integration
- _No car-car collisions yet_

### Milestone 3: Full Collision System - $700 (2-3 Weeks)

Complete collision detection and game mechanics.

**Deliverables:**

- Car-ball collisions
- Car-car collisions
- Wall/arena collisions for cars
- Demolition (demo) system
- Goal detection
- Boost pad pickup
- Performance tuning for parallel environments

### Milestone 4: Integration + Optimization - $500 (1-2 Weeks)

Full integration and performance maximization.

**Deliverables:**

- Full GigaLearnCPP integration
- Multi-GPU support (dual RTX 5090s)
- Maximum SPS optimization
- Optional mixed-precision optimizations
- Memory optimization
- Final testing and validation

---

## Reference Repositories

- **RocketSim**: https://github.com/ZealanL/RocketSim
- **GigaLearnCPP**: https://github.com/ZealanL/GigaLearnCPP-Leak

## Technical Approach

- Data-oriented design (Structure of Arrays - SoA)
- CUDA pipeline for parallel environment stepping
- Batched tensor outputs for neural network training
- GPU-friendly collision system (replacing sequential CPU collisions)
