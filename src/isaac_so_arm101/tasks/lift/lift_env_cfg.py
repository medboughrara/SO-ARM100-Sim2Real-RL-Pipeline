# Copyright (c) 2024-2025, Muammer Bay (LycheeAI), Louis Le Lay
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

# from . import mdp
import isaac_so_arm101.tasks.lift.mdp as mdp
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    DeformableObjectCfg,
    RigidObjectCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from isaaclab.utils.offset import OffsetCfg
# from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from isaaclab.utils.visualizer import FRAME_MARKER_CFG
# from isaaclab.utils.assets import RigidBodyPropertiesCfg


##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # contact sensor for the arm (to penalize table contact)
    # We target specific bodies to avoid penalizing the gripper
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        update_period=0.0, 
        history_length=6, 
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table", "/World/GroundPlane"]
    )

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Target drop-zone box — bright red visual marker on the table surface
    target_box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetBox",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.2, 0.015]),
        spawn=CuboidCfg(
            size=(0.08, 0.08, 0.005),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.1, 0.1),
                opacity=0.85,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            # Goal is strictly IN FRONT of the arm (+X direction) to avoid ghosting the base
            pos_x=(0.10, 0.25),   
            pos_y=(-0.20, 0.20),
            pos_z=(0.05, 0.10),  # just above the surface
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Strictly in front of the robot arm
            "pose_range": {"x": (0.1, 0.2), "y": (-0.15, 0.15), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )

    # Randomize the red drop-zone box position securely in front of the robot arm
    reset_target_box_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Target box spawns anywhere in front of the base but away from the edge
            "pose_range": {
                "x": (0.10, 0.25),   
                "y": (-0.20, 0.20),
                "z": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("target_box"),
        },
    )


@configclass
class RewardsCfg:
    """4-Stage behavioral reward.

    Stage 1: Approach — move EE toward cube (always active)
    Stage 1->2: Open approach — keep gripper OPEN while within 20 cm
    Stage 2: Enclose — close fingers when within 7 cm of cube
    Stage 3: Lift — binary bonus when cube leaves the table
    Stage 4: Goal — move lifted cube to target pose
    """

    # Stage 1: Approach — drive EE to cube
    reaching_object = RewTerm(
        func=mdp.object_ee_distance,
        params={"std": 0.1},
        weight=10.0,
    )

    # Stage 1->2: Keep hand OPEN while approaching
    open_gripper_approach = RewTerm(
        func=mdp.reward_open_gripper_while_approaching,
        params={},
        weight=3.0,
    )

    # Stage 2: Close fingers once over the cube
    squeeze_object = RewTerm(
        func=mdp.gripper_is_closed_near_object,
        params={},  # uses default _DIST_ENCLOSURE = 7 cm
        weight=15.0,
    )

    # Stage 3: Lift — binary bonus
    lifting_object = RewTerm(
        func=mdp.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=20.0,
    )

    # Stage 4: Goal tracking — coarse
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    # Stage 4: Goal tracking — fine-grained bonus
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # Regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Joint limit penalty — penalise joints within 5 % of their hard limit
    # This stops the arm from grinding against its own mechanical stops
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping_early = DoneTerm(
        func=mdp.root_height_below_minimum, 
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    # HARD CAGE: Terminate if the robot turns more than 90 degrees (1.57 rad) away from the table.
    base_rotation_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="shoulder_pan"),
            "bounds": (-1.6, 1.6),
        }
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
