# Standard Library
from typing import Dict

# Third Party
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
import torch


class MotionGeneration:

    def __init__(self,
                
                 robot_prim_path,
                 collisions_prim_path,
                 word,
                 frame_skip,
                 robot_path="ur16e.yml") -> None:

        n_obstacle_cuboids = 30
        n_obstacle_mesh = 10
      
        tensor_args = TensorDeviceType()

        robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                        robot_path))["robot_cfg"]

        j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

        world_cfg_table = WorldConfig.from_dict(
            load_yaml(
                join_path(get_world_configs_path(), "collision_table.yml")))
        world_cfg_table.cuboid[0].pose[2] -= 0.04
        world_cfg1 = WorldConfig.from_dict(
            load_yaml(
                join_path(get_world_configs_path(),
                          "collision_table.yml"))).get_mesh_world()
        world_cfg1.mesh[0].name += "_mesh"
        world_cfg1.mesh[0].pose[2] = -10.5

        world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid,
                                mesh=world_cfg1.mesh)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.03,
            collision_cache={
                "obb": n_obstacle_cuboids,
                "mesh": n_obstacle_mesh
            },
            collision_activation_distance=0.025,
            acceleration_scale=1.0,
            self_collision_check=True,
            maximum_trajectory_dt=0.25,
            fixed_iters_trajopt=True,
            finetune_dt_scale=1.05,
            velocity_scale=[0.25, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0],
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up...")
        self.motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False)

        self.plan_config = MotionGenPlanConfig(enable_graph=False,
                                               enable_graph_attempt=4,
                                               max_attempts=2,
                                               enable_finetune_trajopt=True)

        usd_help = UsdHelper()

        # obstacles = usd_help.get_obstacles_from_stage(
        #     # only_paths=[obstacles_path],
        #     reference_prim_path=robot_prim_path,
        #     ignore_substring=[
        #         robot_prim_path,
        #         "/curobo",
        #     ],
        # ).get_collision_check_world()

        # self.motion_gen.update_world(obstacles)
        self.tensor_args = TensorDeviceType()

        self.target_pose = None
        self.past_pose = None

        self._world = word
        self.frame_skip = frame_skip

    def step_path(self, target_ee_pos, target_ee_orientation,sim_js,sim_js_names):
        from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion

      
        cmd_plan = None

        # sim_js = self.robot.get_joints_state()
        # sim_js_names = self.robot.dof_names
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities),
            acceleration=self.tensor_args.to_device(sim_js.velocities),
            jerk=self.tensor_args.to_device(sim_js.velocities),
            joint_names=sim_js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(
            self.motion_gen.kinematics.joint_names)

        ee_translation_goal = target_ee_pos
        ee_orientation_teleop_goal = target_ee_orientation

        # compute curobo solution:
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_teleop_goal),
        )

        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal,
                                             self.plan_config)

        succ = result.success.item()  # ik_result.success.item()
      
        if succ:
            
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = self.motion_gen.get_full_js(cmd_plan)
            # get only joint names that are in both:
            idx_list = []
            common_js_names = []
            for x in sim_js_names:
                if x in cmd_plan.joint_names:
                    idx_list.append(self.robot.get_dof_index(x))
                    common_js_names.append(x)
            # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

            cmd_plan = cmd_plan.get_ordered_joint_state(common_js_names)

            cmd_state = cmd_plan[-1]

            return cmd_state.position

        return self.tensor_args.to_device(sim_js.positions)
