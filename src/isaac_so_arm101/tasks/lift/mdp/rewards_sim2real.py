import torch
import torch.nn.functional as F
from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils

from isaaclab.envs import ManagerBasedRLEnv

_PREV_ACTIONS: dict[int, torch.Tensor] = {}

def safe_action_rate_l2(
    env: ManagerBasedRLEnv,
    max_sq_diff: float = 4.0,  # clamp per-joint squared diff; 2.0 rad change -> 4.0 sq
) -> torch.Tensor:
    """A NaN-safe, bounded replacement for mdp_isaac.action_rate_l2.
    
    The standard action_rate_l2 can explode to infinity if the physics engine
    returns NaN/Inf joint states. This version:
    1. Replaces any NaN/Inf actions with 0.0 before computing
    2. Clamps each joint's squared difference to [0, max_sq_diff]
    3. Returns a bounded total — impossible to exceed 6 * max_sq_diff per step
    """
    env_id = id(env)
    curr_actions = env.action_manager.action.clone()
    
    # Replace NaN/Inf with 0 to prevent propagation
    curr_actions = torch.nan_to_num(curr_actions, nan=0.0, posinf=0.0, neginf=0.0)
    
    if env_id not in _PREV_ACTIONS or _PREV_ACTIONS[env_id].shape != curr_actions.shape:
        _PREV_ACTIONS[env_id] = curr_actions.clone()
        return torch.zeros(curr_actions.shape[0], device=curr_actions.device)
    
    prev_actions = _PREV_ACTIONS[env_id]
    prev_actions = torch.nan_to_num(prev_actions, nan=0.0, posinf=0.0, neginf=0.0)
    
    sq_diff = (curr_actions - prev_actions) ** 2
    sq_diff = torch.clamp(sq_diff, 0.0, max_sq_diff)  # circuit breaker per joint
    
    _PREV_ACTIONS[env_id] = curr_actions.clone()
    return torch.sum(sq_diff, dim=-1)


def object_ee_distance_exp(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward moving EE toward the cube using a dense exponential kernel."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    dist = torch.norm(obj.data.root_pos_w - ee.data.target_pos_w[..., 0, :], dim=1)
    return torch.exp(-dist / std)

def object_lifted_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Continuous reward tracking the z-height of the object relative to table."""
    obj: RigidObject = env.scene[object_cfg.name]
    
    # Measure how close the object's Z is to the target height
    # Assumes table surface is roughly at Z=0.0 based on ObjectTableSceneCfg
    z_height = torch.clamp(obj.data.root_pos_w[:, 2], min=0.0, max=target_height)
    
    # Returns values smoothly going from 0.0 to 1.0
    return z_height / target_height

def gripper_alignment(
    env: ManagerBasedRLEnv,
    forward_axis: list[float] = [0.0, 1.0, 0.0], # Typically Y or Z for EE approach
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward aligning the gripper's forward axis with the world Up axis (-Z direction to point down at top face)."""
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Get EE orientation quaternion
    ee_quat = ee.data.target_quat_w[..., 0, :]
    
    # Calculate the current forward direction of the EE in world frame
    fwd_tensor = torch.tensor(forward_axis, device=env.device).repeat(ee_quat.shape[0], 1)
    ee_fwd_w = math_utils.quat_apply(ee_quat, fwd_tensor)
    
    # Desired vector is pointing straight DOWN at the top face of the cube ( वर्ल्ड [0, 0, -1] )
    desired_up_w = torch.zeros_like(ee_fwd_w)
    desired_up_w[:, 2] = -1.0
    
    # Cosine similarity: dot product of normalized vectors
    # Will be +1 if perfectly aligned going down, -1 if going up
    cos_sim = torch.sum(ee_fwd_w * desired_up_w, dim=-1)
    
    return torch.clamp(cos_sim, min=0.0, max=1.0)

def exponential_joint_limit_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Safe quadratic penalty for joints approaching within 10% of their physical limits.
    
    NOTE: The previous exponential formula (exp(violation*10)) caused catastrophic reward
    explosion when joints exceeded limits even slightly. This version is bounded and safe.
    It uses a clamped quadratic that smoothly rises to a max of 1.0 per joint.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    
    q = robot.data.joint_pos
    q_lower = robot.data.soft_joint_pos_limits[..., 0]
    q_upper = robot.data.soft_joint_pos_limits[..., 1]
    
    # Normalize position to [-1, 1] range based on limits
    middle = (q_upper + q_lower) / 2.0
    half_range = (q_upper - q_lower) / 2.0
    # Avoid zero division for fixed joints
    half_range = torch.where(half_range == 0, torch.ones_like(half_range), half_range)
    
    q_norm = (q - middle) / half_range  # [-1, 1]
    
    # Penalty only active if within 10% of limits (|q_norm| > 0.9)
    # Quadratic ramp from 0 to 1.0 — bounded, can never blow up
    violation = torch.relu(torch.abs(q_norm) - 0.9) / 0.1  # 0 to 1 range
    penalty_per_joint = torch.clamp(violation ** 2, 0.0, 1.0)
    return torch.sum(penalty_per_joint, dim=-1)

def vertical_place_alignment(
    env: ManagerBasedRLEnv,
    forward_axis: list[float] = [0.0, 1.0, 0.0],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """Phase 2 Place Reward: Reward downward alignment ONLY when grasping the object."""
    obj: RigidObject = env.scene[object_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # Check if grasped (cube is lifted)
    is_grasped = (obj.data.root_pos_w[:, 2] > minimal_height).float()
    
    ee_quat = ee.data.target_quat_w[..., 0, :]
    fwd_tensor = torch.tensor(forward_axis, device=env.device).repeat(ee_quat.shape[0], 1)
    ee_fwd_w = math_utils.quat_apply(ee_quat, fwd_tensor)
    
    desired_up_w = torch.zeros_like(ee_fwd_w)
    desired_up_w[:, 2] = -1.0 # Point down
    
    cos_sim = torch.sum(ee_fwd_w * desired_up_w, dim=-1)
    alignment = torch.clamp(cos_sim, min=0.0, max=1.0)
    
    return is_grasped * alignment

def release_success_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "object_pose",
) -> torch.Tensor:
    """The Release Trigger: Reward triggered if object is over box, gripper opens, and cube drops."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Goal position
    des_pos_w, _ = math_utils.combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )
    
    # In XY bounds?
    dist_xy = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    in_xy_bounds = (dist_xy < threshold).float()
    
    # Gripper open?
    _GRIPPER_IDX = 5
    _GRIPPER_OPEN = 1.1
    _GRIPPER_CLOSE = 0.15
    gpos = robot.data.joint_pos[:, _GRIPPER_IDX]
    openness = torch.clamp((gpos - _GRIPPER_CLOSE) / (_GRIPPER_OPEN - _GRIPPER_CLOSE), 0.0, 1.0)
    is_open = (openness > 0.5).float()
    
    # Cube height dropping? (Z velocity is negative)
    is_dropping = (obj.data.root_lin_vel_w[:, 2] < -0.05).float()
    
    return in_xy_bounds * is_open * is_dropping

def post_drop_retreat(
    env: ManagerBasedRLEnv,
    threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Post-Release Bonus: Reward moving hand away from the box after successful drop."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee: FrameTransformer = env.scene[ee_frame_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    des_pos_w, _ = math_utils.combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], command[:, :3]
    )
    
    dist_xy = torch.norm(des_pos_w[:, :2] - obj.data.root_pos_w[:, :2], dim=1)
    in_box = (dist_xy < threshold).float()
    on_table = (obj.data.root_pos_w[:, 2] < 0.05).float()
    
    # Maximize distance between EE and object up to 30cm
    ee_pos = ee.data.target_pos_w[..., 0, :]
    dist_ee_obj = torch.norm(ee_pos - obj.data.root_pos_w, dim=1)
    
    retreat_reward = torch.clamp(dist_ee_obj / 0.3, 0.0, 1.0)
    
    return in_box * on_table * retreat_reward
