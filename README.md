# SO-ARM100 Sim2Real RL Pipeline

A high-performance, numerically stabilized Reinforcement Learning pipeline for training continuous control policies on the SO-ARM100/101 6-DOF robotic manipulators. 

This repository implements advanced **Sim2Real** transfer techniques, engineered to handle the chaotic complexities of physical hardware through strict behavioral gating and numerical "hardening."

---

## 🚀 Technical Architecture

This pipeline is built on **NVIDIA Isaac Lab** and uses **Proximal Policy Optimization (PPO)** via the `rsl_rl` library. 

### Core Specifications
- **Algorithm**: PPO (On-Policy)
- **Parallelism**: 4,096 Environments (Massively Parallel GPU Simulation)
- **Simulation Frequency**: 60Hz Control Loop / 120Hz Physics
- **Observation Space (28D)**: Joint Positions (6), Joint Velocities (6), Relative Object Position (3), Command/Goal Pose (7), Last Actions (6).
- **Action Space (6D)**: 5-DOF Arm Joints + 1-DOF Gripper position delta.

```mermaid
graph TD
    A[Isaac Sim Physics] -->|Raw Obs| B(Iron Shield: ObservationShield)
    B -->|Clamped [-10, 10]| C[PPO Policy]
    C -->|Actions| D[Safe Action Rate Limiter]
    D -->|Clamped Gradients| A
    A -->|Raw Reward| E(Iron Shield: RewardShield)
    E -->|Clamped [-100, 100]| F[RL Optimizer]
```

---

## 🛡️ The "Iron Shield" (Numerical Hardening)

Robotic simulations often suffer from "Singularity Explosions"—rare collisions that generate infinite forces, crashing the learning process. We protect our networks with a three-layer defense:

1.  **`ObservationShield`**: Clamps all incoming simulator data to `[-10.0, 10.0]`. Sensor glitches can no longer corrupt the brain.
2.  **`RewardShield`**: Clamps total step rewards to `[-100.0, 100.0]`. This acts as a circuit breaker against runaway gradient spikes.
3.  **`Safe Action Rate`**: A custom `safe_action_rate_l2` term that bounds the derivative of joint movements, ensuring physical smoothness and preventing the arm from "flailing" away from its base.

---

## 🎓 Behavioral Logic: Phase 2.5 "Logic Master"

Moving beyond simple grasping, this pipeline implements **Sequential Logic Gating** to ensure the robot respects the correct order of operations: **Locate → Reach → Open → Grip → Lift.**

### 🌍 Exploded Spatial Randomization
To prevent the agent from memorizing a single "favorite" spot, the cube's spawn area was expanded by **300%** (covering a 23cm x 50cm region). The agent MUST use its sensory input to find the cube on every reset.

### ⚖️ Sequence Proximity Gate
We enforce a strict **Behavioral Gate** on the grasping reward:
- **Loose Mode (Old)**: Gripper could close anywhere within 7cm of the cube.
- **Strict Mode (New)**: The `squeeze_object` reward is locked behind a **3.5cm threshold**.
- **Impact**: The agent can no longer "cheat" by air-squeezing while moving. It must arrive precisely at the target before it receives points for grasping.

### 🎯 Pinpoint Fingertip Calibration
The reaching reward tracks the **`gripper_frame_link`** directly.
- **Previous**: Manually calculated offsets often resulted in "Ghost Reaching" (targeting the wrist).
- **Current**: The target focus is synced to the URDF's native fingertip center, ensuring the fingers arrive exactly on the cube faces.

---

## 💻 CLI Operations

### 1. Initial Training (Grasp Mastery)
```bash
uv run train --task Isaac-SO-ARM100-Lift-Cube-Sim2Real-v0 --num_envs 4096 --headless --run_name logic_master_v1
```

### 2. Phase 2 Resume (Pick-and-Place)
To transition from a "Grasper" to a "Lifter," lower the transport threshold to 1.0cm:
```bash
uv run train --task Isaac-SO-ARM100-Lift-Cube-Sim2Real-v0 --resume --load_run YYYY-MM-DD_HH-MM-SS --run_name transport_master_v1
```

### 3. Visual Evaluation
Watch the latest "Logic Master" in action:
```bash
uv run play --task Isaac-SO-ARM100-Lift-Cube-Sim2Real-v0 --num_envs 32 --checkpoint "logs/rsl_rl/lift/2026-04-20_19-34-29/model_65508.pt"
```

---
> [!IMPORTANT]
> **Sim2Real Transfer**: Ensure that Empirical Normalization remains **DISABLED** in `rsl_rl_ppo_cfg.py` for all hardware-ready runs.
