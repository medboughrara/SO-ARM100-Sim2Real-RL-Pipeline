from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp_isaac
import isaac_so_arm101.tasks.lift.mdp as mdp

# local imports
import isaac_so_arm101.tasks.lift.mdp.rewards_sim2real as custom_rewards
from isaac_so_arm101.tasks.lift.joint_pos_env_cfg import SoArm100LiftCubeEnvCfg, SoArm101LiftCubeEnvCfg
from isaac_so_arm101.tasks.lift.lift_env_cfg import EventCfg, RewardsCfg

##
# Updated Event Config (with Domain Randomization)
##

@configclass
class Sim2RealEventCfg(EventCfg):
    """Event terms for the MDP adding domain randomizations for Sim2Real robustness."""
    
    # 1. Randomize the mass of the cube by +/- 20%
    randomize_cube_mass = EventTerm(
        func=mdp_isaac.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2), # scaling factors
            "operation": "scale"
        },
    )



##
# Updated Reward Config
##

@configclass
class Sim2RealRewardsCfg(RewardsCfg):
    """Updated 4-Stage reward structure fine-tuned for Sim2Real transfer.
    
    Key fix: joint_vel and action_rate penalties are 15x stronger than the base
    config. This prevents the arm from learning that slow continuous spinning
    toward the goal is worth the small spin cost.
    """

    # 1. Dense Approaching with EXP kernel
    reaching_object = RewTerm(
        func=custom_rewards.object_ee_distance_exp,
        params={"std": 0.1},
        weight=15.0,
    )

    # 2. Continuous Lifting Reward (Max 0.05 height cap)
    lifting_object = RewTerm(
        func=custom_rewards.object_lifted_height,
        params={"target_height": 0.05},
        weight=50.0, # Increased from 20.0 to 50.0 to prioritize lifting
    )

    # 2b. Grasping Incentive: Massive boost to ensure fingers close near object.
    # We boost this to 150.0 to make the grasp the 'key' to all further rewards.
    squeeze_object = RewTerm(
        func=mdp.gripper_is_closed_near_object,
        params={},
        weight=150.0,
    )

    # 3. Anti-spin: strong velocity and smoothness penalties
    # Previously -0.01, increased 15x to break the spinning local optimum.
    # The arm was spinning because reward(goal) >> cost(spin). Now cost(spin)
    # is balanced against gain from goal tracking.
    # 3. Anti-spin: Safe bounded version — cannot exceed 6 * 4.0 = 24.0 per step.
    # Replaces mdp_isaac.action_rate_l2 which has no NaN/Inf guard and can blow up
    # when the physics engine produces degenerate joint states.
    action_rate = RewTerm(func=custom_rewards.safe_action_rate_l2, weight=-0.05)
    joint_vel = RewTerm(
        func=mdp_isaac.joint_vel_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 4. Joint limit penalty - SAFE QUADRATIC (bounded, replaces the exploding exponential)
    exponential_joint_limit_penalty = RewTerm(
        func=custom_rewards.exponential_joint_limit_penalty,
        weight=-3.0,  # Max penalty per step: -3.0 * 6 joints * 1.0 = -18.0 (safe)
    )

    # 4b. Joint deviation penalty
    # Set to 'Light' (-1.0) to give the arm permission to move up and transport the cube.
    joint_deviation = RewTerm(
        func=mdp_isaac.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 5. GOAL TRACKING (THE PRIORITY)
    # Massively boosted to ensure progress to the box is the #1 goal once lifted.
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.2, "minimal_height": 0.03, "command_name": "object_pose"},
        weight=100.0,
    )

    # 5b. Fine-grained goal tracking for the 'Place' phase
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.03, "command_name": "object_pose"},
        weight=25.0,
    )

    # 6. Alignment Reward (Place Component)
    vertical_place_alignment = RewTerm(
        func=custom_rewards.vertical_place_alignment,
        params={"forward_axis": [0.0, 1.0, 0.0], "minimal_height": 0.01},
        weight=20.0,
    )

    # 7. Dropping penalty
    dropping_penalty = RewTerm(
        func=mdp_isaac.is_terminated_term,
        params={"term_keys": ["object_dropping_early"]},
        weight=-200.0,
    )

    # 8. Arm-Table Contact Penalty
    # Penalize if any part of the arm (NOT the gripper) touches the table or ground
    arm_table_contact_penalty = RewTerm(
        func=mdp_isaac.undesired_contacts,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["shoulder", "upper_arm", "lower_arm", "wrist"]),
            "threshold": 1.0,
        },
        weight=-2.0,
    )

    # 9. Release Logic (Pick-and-Place continuation)
    # Encourages the agent to open the gripper once reached the target
    reward_open_gripper_at_goal = RewTerm(
        func=mdp.reward_open_gripper_at_goal,
        params={"threshold": 0.05},
        weight=20.0,
    )

    # 10. Unwanted action: holding the cube at the goal
    # Directly penalizes keeping the fingers closed when at the target coordinates
    gripper_closed_at_goal_penalty = RewTerm(
        func=mdp.gripper_closed_at_goal_penalty,
        params={"threshold": 0.05},
        weight=-15.0,
    )

    # 11. FINAL SUCCESS: Object is inside the box and stationary
    object_in_target_box = RewTerm(
        func=mdp.object_in_target_box,
        params={"threshold": 0.04}, # Within the 8cm box radius
        weight=50.0,
    )

    # 12. The Release Trigger (Success Bonus)
    release_success_bonus = RewTerm(
        func=custom_rewards.release_success_bonus,
        params={"threshold": 0.04},
        weight=50.0,
    )

    # 13. Post-Release Bonus (Retreat)
    post_drop_retreat = RewTerm(
        func=custom_rewards.post_drop_retreat,
        params={"threshold": 0.06},
        weight=20.0,
    )


##
# New Environment Configs
##

@configclass
class SoArm100LiftCubeSim2RealEnvCfg(SoArm100LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        # Override events and rewards with our new Sim2Real versions
        self.events = Sim2RealEventCfg()
        self.rewards = Sim2RealRewardsCfg()

@configclass
class SoArm100LiftCubeSim2RealEnvCfg_PLAY(SoArm100LiftCubeSim2RealEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # Disable DR on reset/startup during Evaluation
        self.events.randomize_cube_mass = None
        self.events.randomize_table_friction = None
