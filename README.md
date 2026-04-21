# SO-ARM100 Sim2Real RL Pipeline

A highly robust, numerically stabilized Reinforcement Learning pipeline for training continuous control policies on the SO-ARM100 6-DOF robotic manipulator. Built on top of **NVIDIA Isaac Lab** and **RSL-RL**, this project is specifically engineered for strict **Sim2Real** transfer, focusing on object reaching, grasping, and lifting.

---

## 🚀 Environment Overview

*   **Task ID**: `Isaac-SO-ARM100-Lift-Cube-Sim2Real-v0`
*   **Robot**: SO-ARM100 (6-DOF manipulator + Surface Gripper)
*   **Objective**: Reach a randomized target block, firmly enclose it with the gripper, lift it off the table, and transport it to a specific 3D target coordinates (a drop box).
*   **Simulation Engine**: NVIDIA Omniverse / Isaac Sim Physics (with Fabric).

## 🧠 Reinforcement Learning Architecture

The pipeline utilizes **Proximal Policy Optimization (PPO)** implemented via the `rsl_rl` library. 

### Why PPO?
PPO is chosen for continuous control due to its sample efficiency and stable policy updates via clipped surrogate objectives. The `rsl_rl` implementation is highly optimized for massively parallel GPU environments, allowing us to train on 4,096 parallel universes simultaneously.

### Policy Specifications
*   **Algorithm**: PPO (On-Policy)
*   **Network Architecture**: Multi-Layer Perceptron (MLP) Actor-Critic
*   **State Space**: 28 dimensions (Joint Positions (6), Joint Velocities (6), Object Position (3), Target Object Position (7), Previous Actions (6)).
*   **Action Space**: 6 dimensions (5-DOF Arm Joints + 1-DOF Gripper actuation).
*   **Normalization**: Empirical normalization explicitly **disabled** to maintain absolute coordinate grounding for physical hardware transfer.

---

## 🛡️ The "Iron Shield" (Numerical Hardening)

A major hurdle in GPU-accelerated continuous physics is "Singularity Explosions"—rare edge cases where the physics solver outputs massive overlapping forces (NaNs or Infs) resulting in Trillion-scale loss spikes that instantly corrupt the neural network weights.

This project implements the **Iron Shield** at the lowest level of the Gymnasium ecosystem to prevent this:

1.  **`ObservationShield` Wrapper**: Intercepts the raw dictionary observations coming from Isaac Sim and rigorously clamps all tensor values between `[-10.0, 10.0]`.
2.  **`RewardShield` Wrapper**: Clamps the accumulated step rewards to a firm `[-100.0, 100.0]`, guaranteeing that no extreme penalty or bonus can skew the gradient descent.
3.  **Circuit Breaker Rewards**: Replaced standard Isaac Lab L2 penalties with custom bounds. For instance, `safe_action_rate_l2` traps and bounds squared action differentials, preventing degenerate joint states from ever bleeding into the reward calculation.

---

## 🎓 Sim2Real Training Curriculum

Training a complex 6-DOF arm to grasp and place requires a multi-stage behavioral curriculum. We split the training to prevent the network from finding "lazy local optima" (like flailing or hovering safely away from the target).

### Phase 1: Reaching & Grasping Expert
*   **Goal**: Lock onto the block and secure a grip without dropping it.
*   **Reward Balance**:
    *   `squeeze_object`: Massive weight (40.0)
    *   `reaching_object`: High weight (15.0)
    *   `joint_deviation`: Medium penalty (-3.0) to serve as a strict "stability anchor" preventing random flailing.
*   **Result**: The network converges into a **Grasping Expert**, grabbing the cube rapidly but freezing in place to avoid movement penalties.

### Phase 2: Lifting & Transporting (Pick-and-Place)
*   **Goal**: Pull the object up and move it to the spatial target zone.
*   **Reward Balance**:
    *   `lifting_object`: Skyrocketed to 50.0.
    *   `object_goal_tracking`: Coarse tracking (+100.0) combined with fine-grained tracking (+25.0).
    *   *Threshold Tuning*: Goal tracking activation heights lowered to **1.0cm** above the table, immediately "tugging" the arm toward the target the moment it is safely lifted.
    *   *Anchor Release*: `joint_deviation` penalty relaxed to -1.0 to give the arm permission to stretch across the physical workspace.

---

## 💻 CLI Operations

### Train the Network
To launch a training run natively:
```bash
uv run train --task Isaac-SO-ARM100-Lift-Cube-Sim2Real-v0 --num_envs 4096 --headless
```

### Resume / Fine-Tune (Phase 2)
To load a certified expert from a specific run and continue training:
```bash
uv run train --task Isaac-SO-ARM100-Lift-Cube-Sim2Real-v0 --num_envs 4096 --headless --resume --load_run YYYY-MM-DD_HH-MM-SS --checkpoint model_XXXX.pt
```

### Evaluate & Play
Watch the trained policy perform tasks in the simulator GUI: # Note: Must use absolute or relative file path for checkpoint!
```bash
uv run play --task Isaac-SO-ARM100-Lift-Cube-Sim2Real-v0 --num_envs 32 --checkpoint "logs/rsl_rl/lift/YYYY-MM-DD_HH-MM-SS/READY_Grasping_Expert.pt"
```

---
*Maintained as the core ML infrastructure for the SO-ARM100 continuous control initiative.*
