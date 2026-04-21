# SO-ARM101 Deployment Manifest (Sim2Real Ready)

This manifest tracks models that have successfully mastered specific behavioral milestones in simulation and are certified for real-world testing.

## Certified Models

### 🥇 Expert Reaching & Grasping
- **Model Name**: `READY_Grasping_Expert_model_41898.pt`
- **Source Run**: `2026-04-18_22-05-27`
- **Milestone Achieved**: Unified Reaching + Firm Object Enclosure (`squeeze_object` > 30.0).
- **Sim2Real Notes**: Stabilized by the **Iron Shield**. Provides consistent "reach-and-grab" behavior with near-zero resets.
- **Play Command**:
```powershell
uv run play --task Isaac-SO-ARM100-Lift-Cube-Sim2Real-v0 --num_envs 32 --checkpoint "logs/rsl_rl/lift/2026-04-18_22-05-27/READY_Grasping_Expert_model_41898.pt"
```

### 🥈 Phase 2: Grasp-Corrected Master
- **Model Name**: `model_65508.pt`
- **Source Run**: `2026-04-20_19-34-29`
- **Milestone Achieved**: Secure grasp restored. Fingers close reliably before transport.
- **Current Objective**: Increasing lift height to 3cm to trigger goal tracking.
- **Play Command**:
```powershell
uv run play --task Isaac-SO-ARM100-Lift-Cube-Sim2Real-v0 --num_envs 4 --checkpoint "logs/rsl_rl/lift/2026-04-20_19-34-29/model_65508.pt"
```

---

## Verified Capabilities Registry
| Milestone | Accomplished? | Best Model | Notes |
|-----------|---------------|------------|-------|
| **Base Stability** | ✅ | `model_31899+` | Resets dropped from 74% to <1%. |
| **Object Reaching** | ✅ | `model_41898` | Dense reaching reward climb successful. |
| **Solid Grasping** | ✅ | `model_41898` | Firm enclosure verified in sim. |
| **Lifting & Place** | 🚧 | N/A | Next training objective. |
