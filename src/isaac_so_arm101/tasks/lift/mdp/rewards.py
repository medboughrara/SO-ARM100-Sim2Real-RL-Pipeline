# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ── Gripper constants (SO-ARM100) ─────────────────────────────────────────────
# Joint index 5 = gripper revolute joint
_GRIPPER_IDX = 5
_GRIPPER_OPEN = 1.1    # rad — wide open (safely below the 1.2 hard limit)
_GRIPPER_CLOSE = 0.15  # rad — fully closed (no self-intersection)

# Distance thresholds that define the 4-stage behavioral sequence
_DIST_START_OPEN = 0.20   # At 20 cm: start opening the gripper
_DIST_ENCLOSURE  = 0.035  # At 3.5 cm: strictly over the cube — now close fingers

# ── STAGE 1: Approach (always active) ────────────────────────────────────────

def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward moving EE toward the cube (always active, tanh-kernel)."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    return 1.0 - torch.tanh(dist / std)


def reward_open_gripper_while_approaching(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """STAGE 1->2 boundary: when within 20 cm, reward keeping the gripper OPEN.
    Teaches: approach the cube with a wide-open hand before enclosing.
    Returns values in [0, 1] — higher = more open.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    robot: Articulation = env.scene["robot"]
    gpos = robot.data.joint_pos[:, _GRIPPER_IDX]
    in_approach = (dist < _DIST_START_OPEN).float()
    openness = torch.clamp(gpos / _GRIPPER_OPEN, 0.0, 1.0)
    return in_approach * openness


# ── STAGE 2: Enclose (active inside 7 cm) ────────────────────────────────────

def gripper_is_closed_near_object(
    env: ManagerBasedRLEnv,
    std: float = _DIST_ENCLOSURE,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """STAGE 2: When EE is directly over the cube (<7 cm), reward CLOSING fingers.
    Only activates inside the enclosure zone — away from cube this returns 0.
    Returns values in [0, 1] — higher = more closed.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    robot: Articulation = env.scene["robot"]
    gpos = robot.data.joint_pos[:, _GRIPPER_IDX]
    in_enclosure = (dist < std).float()
    closed = torch.clamp((_GRIPPER_OPEN - gpos) / _GRIPPER_OPEN, 0.0, 1.0)
    return in_enclosure * closed


# ── STAGE 3: Lift ────────────────────────────────────────────────────────────

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """STAGE 3: Binary reward — 1.0 if cube is above the table, else 0."""
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


# ── STAGE 4: Goal Tracking (only while lifted) ───────────────────────────────

def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """STAGE 4: Move the lifted cube toward its commanded goal pose.
    Zero reward if cube is not yet lifted.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )
    dist = torch.norm(des_pos_w - obj.data.root_pos_w[:, :3], dim=1)
    lifted = (obj.data.root_pos_w[:, 2] > minimal_height).float()
    return lifted * (1.0 - torch.tanh(dist / std))


# ── PICK-AND-PLACE: Release Logic ─────────────────────────────────────────────

def reward_open_gripper_at_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "object_pose",
) -> torch.Tensor:
    """STAGE 5: Reward opening the gripper when the object is at the goal.
    This version is more strict: it requires the object to be within a small 
    horizontal threshold and the gripper to be relatively aligned vertically.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Calculate goal position in world frame
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )
    
    # Horizontal distance is more important for the drop zone
    dist_xy = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    dist_z = torch.abs(des_pos_w[:, 2] - obj.data.root_pos_w[:, 2])
    
    # Check if gripper is open
    robot_art: Articulation = env.scene[robot_cfg.name]
    gpos = robot_art.data.joint_pos[:, _GRIPPER_IDX]
    openness = torch.clamp((gpos - _GRIPPER_CLOSE) / (_GRIPPER_OPEN - _GRIPPER_CLOSE), 0.0, 1.0)

    # Only reward opening if very close to the center of the box
    at_goal = (dist_xy < threshold).float() * (dist_z < 0.1).float()
    return at_goal * openness


def gripper_closed_at_goal_penalty(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Penalize keeping the gripper closed when at the goal threshold.
    Matches the user's request to treat 'holding at goal' as an unwanted action.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Calculate goal position in world frame
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )
    dist = torch.norm(des_pos_w - obj.data.root_pos_w[:, :3], dim=1)

    # Check if gripper is closed
    robot_art: Articulation = env.scene[robot_cfg.name]
    gpos = robot_art.data.joint_pos[:, _GRIPPER_IDX]
    closedness = torch.clamp((_GRIPPER_OPEN - gpos) / _GRIPPER_OPEN, 0.0, 1.0)

    # Penalize closedness only when near goal
    near_goal = (dist < threshold).float()
    return near_goal * closedness

def object_in_target_box(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "object_pose",
) -> torch.Tensor:
    """FINAL STAGE: Significant reward for the object actually resting inside the target box area.
    This is the ultimate 'success' signal.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: RigidObject = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Calculate goal position in world frame
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )
    
    # Check if object is inside the box dimensions (8cm box -> 0.04 radius)
    dist_xy = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    
    # Object must be on the table (Z low) and not moving fast
    on_table = (obj.data.root_pos_w[:, 2] < 0.05).float()
    stationary = (torch.norm(obj.data.root_lin_vel_w, dim=1) < 0.1).float()
    inside = (dist_xy < threshold).float()
    
    return inside * on_table * stationary
