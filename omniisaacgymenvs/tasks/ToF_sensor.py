# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

############################################################
#################  isaac sim   ##############################
############################################################
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
# from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omniisaacgymenvs.robots.articulations.ur_robot import UR

from omni.isaac.core.objects import DynamicSphere

from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import DynamicCylinder
from omni.isaac.core.objects import FixedCuboid
from omniisaacgymenvs.robots.controller.ocs2 import Controller_ocs2
from omniisaacgymenvs.robots.controller.osc import Controller_osc

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, is_prim_path_valid
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.simulation_context import SimulationContext
import numpy as np
import torch
import math
import trimesh
from trimesh import creation, transformations

# from omni.isaac.universal_robots import UR10
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage, clear_stage
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper
from pxr import Usd, UsdGeom
from .raycast import Raycast, geom_to_trimesh, warp_from_trimesh
import pdb

import os
import omni.isaac.core.utils.prims as prim_utils
import omni
from pxr import UsdPhysics
from omni.physx.scripts import utils
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf

from omni.physx import acquire_physx_interface

import carb

import gym

import warp as wp
import omni.isaac.core.utils.nucleus as nucleus_utils

# ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

from omniisaacgymenvs.utils.tools.rotation_conversions import *

from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
import omniisaacgymenvs.utils.tools.transform_utils as tf
from cprint import *
import xml.etree.ElementTree as ET
# 3D transformations functions
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, matrix_to_quaternion, quaternion_multiply
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper_Properties
from omniisaacgymenvs.robots.articulations.surface_gripper import SurfaceGripper

import time

############################################################
#################  curobo  ##############################
############################################################

from typing import Dict
import warp as wp
# Third Party
import carb
import numpy as np

from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere


class TofSensorTask(RLTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._device = self._cfg["rl_device"]

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["num_actions"]

        RLTask.__init__(self, name, env)

        self._robot_positions = self._task_cfg['sim']["URRobot"]["position"]
        self._robot_rotations = self._task_cfg['sim']["URRobot"]["quaternion"]
        self._robot_dof_target = torch.tensor(
            self._task_cfg['sim']["URRobot"]["dof_target"],
            device=self._device)
        self._robot_dof_targets = self._robot_dof_target.repeat(
            self._num_envs, 1)

        self.init_table_position = torch.tensor(
            self._task_cfg['sim']["Table"]["position"],
            device=self._device).repeat(self._num_envs, 1)

        self.object_category = self._task_cfg['sim']["Object"]["category"]
        self._manipulated_object_positions = [
            torch.tensor([-0.6, 0.0, 1.9]),
            torch.tensor([-0.6, -0.25, 1.9])
        ]
        self.debug_draw = _debug_draw.acquire_debug_draw_interface()

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._end_effector_link = "ee_link"
        self.current_directory = os.getcwd()

        self._step = 0
        self.angle_dev = torch.ones(self.num_envs)
        self.current_euler_angles = torch.zeros((self.num_envs, 3))

        # control parameter

        self.frame_skip = 1

        self.object_prim_path = []

        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.pre_action = torch.zeros((self.num_envs, 6), device=self.device)
        velocity_limit = torch.as_tensor([1.0] * 3 + [3.0] * 3,
                                         device=self.device)  # slow down

        self.velocity_limit = torch.as_tensor(torch.stack(
            [-velocity_limit, velocity_limit], dim=1),
                                              device=self.device)

        # self.start_time = time.time()

        self.load_bin_yaml('omniisaacgymenvs/cfg/bin.xml')

        return

    def init_curobo(self):

        # # CuRobo
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

        # from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.rollout.rollout_base import Goal
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import JointState, RobotConfig
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
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
        from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

        self.tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                        "ur16e.yml"))["robot_cfg"]
        j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

        world_cfg_table = WorldConfig.from_dict(
            load_yaml(
                join_path(get_world_configs_path(), "collision_table.yml")))
        world_cfg_table.cuboid[0].pose[2] -= 0.002
        world_cfg1 = WorldConfig.from_dict(
            load_yaml(
                join_path(get_world_configs_path(),
                          "collision_table.yml"))).get_mesh_world()
        world_cfg1.mesh[0].name += "_mesh"
        world_cfg1.mesh[0].pose[2] = -10.5

        n_obstacle_cuboids = 30
        n_obstacle_mesh = 10

        world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid,
                                mesh=world_cfg1.mesh)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                "obb": n_obstacle_cuboids,
                "mesh": n_obstacle_mesh
            },
            # use_fixed_samples=True,
        )
        ik_solver = IKSolver(ik_config)

        self.position_grid_offset = self.tensor_args.to_device(
            self.get_pose_grid(10, 10, 5, 0.5, 0.5, 0.5))

        fk_state = ik_solver.fk(ik_solver.get_retract_config().view(1, -1))
        self.goal_pose = fk_state.ee_pose
        self.goal_pose = self.goal_pose.repeat(
            self.position_grid_offset.shape[0])
        self.goal_pose.position += self.position_grid_offset

        return ik_solver

    def get_pose_grid(self, n_x, n_y, n_z, max_x, max_y, max_z):
        x = np.linspace(-max_x, max_x, n_x)
        y = np.linspace(-max_y, max_y, n_y)
        z = np.linspace(0, max_z, n_z)
        x, y, z = np.meshgrid(x, y, z, indexing="ij")

        position_arr = np.zeros((n_x * n_y * n_z, 3))
        position_arr[:, 0] = x.flatten()
        position_arr[:, 1] = y.flatten()
        position_arr[:, 2] = z.flatten()

        return position_arr

    def load_bin_yaml(self, file):
        tree = ET.parse(file)
        root = tree.getroot()

        self.init_robot_joints_name = {}
        self.init_robot_joints = []

        # Now you can iterate over the elements and extract the data
        for group_state in root.findall('group_state'):
            state_name = group_state.get('name')
            group_name = group_state.get('group')
            self.init_robot_joints_name[state_name] = []
            joint_angle = []
            for joint in group_state.findall('joint'):
                joint_name = joint.get('name')
                joint_value = float(joint.get('value'))
                joint_angle.append(joint_value)

            self.init_robot_joints_name[group_name] = torch.as_tensor(
                np.array(joint_angle)[[2, 1, 0, 3, 4, 5]]).to(self.device)
            self.init_robot_joints.append(
                np.array(joint_angle)[[2, 1, 0, 3, 4, 5]])
        self.init_robot_joints = torch.as_tensor(self.init_robot_joints)

    def init_data(self) -> None:

        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz],
                                device=device,
                                dtype=torch.float)

        stage = get_current_stage()

        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(
                stage.GetPrimAtPath("/World/envs/env_0/robot/ee_link")),
            self._device)
        self._end_effector.get_world_poses()

        hand_pose_inv_rot, hand_pose_inv_pos = (tf_inverse(
            hand_pose[3:7], hand_pose[0:3]))
        self.franka_local_grasp_pos = hand_pose[3:7].repeat(
            (self._num_envs, 1))
        self.franka_local_grasp_rot = hand_pose[0:3].repeat(
            (self._num_envs, 1))

        self.init_mesh()

        self.raytracer = Raycast(self._cfg["raycast_width"],
                                 self._cfg["raycast_height"],
                                 [self.mesh_vertices[0]], [self.mesh_faces[0]])

    def init_mesh(self):

        cube = UsdGeom.Cube(
            get_prim_at_path(self._manipulated_object.prim_paths[0]))

        if self.object_category in ['cube']:
            size = cube.GetSizeAttr().Get()
            cube = trimesh.creation.box(extents=(1, 1, 1))

            self.mesh_faces = torch.as_tensor(cube.faces[None, :, :],
                                              dtype=torch.int32).repeat(
                                                  (self.num_envs, 1,
                                                   1)).to(self.device)
            self.mesh_vertices = torch.as_tensor(cube.vertices[None, :, :],
                                                 dtype=torch.float32).repeat(
                                                     (self.num_envs, 1,
                                                      1)).to(self.device)

    def set_up_scene(self, scene) -> None:

        self.load_robot()
        self.add_gripper()

        # self.load_sphere()
        # self.load_manipulated_object()
        if self.object_category in ['cube']:
            self.load_cube()

        # self.load_pod()
        # Table
        self.load_table()

        super().set_up_scene(scene)

        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/robot",
                                        name="robot_view",
                                        reset_xform_properties=False)
        scene.add(self._robots)

        # end-effectors view
        self._end_effector = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/ee_link",
            name="end_effector_view",
            reset_xform_properties=False)
        scene.add(self._end_effector)

        self.wrist_2_link = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/wrist_2_link",
            name="wrist_2_link_view",
            reset_xform_properties=False)
        scene.add(self.wrist_2_link)

        # manipulated object
        self._manipulated_object = RigidPrimView(
            prim_paths_expr="/World/envs/.*/manipulated_object_1",
            name="manipulated_object_view",
            reset_xform_properties=False)
        scene.add(self._manipulated_object)

        # table
        self._table = RigidPrimView(prim_paths_expr="/World/envs/.*/table",
                                    name="table_view",
                                    reset_xform_properties=False)
        scene.add(self._table)

        # Raytracing

        self.init_data()
        # self.curo_ik_solver = self.init_curobo()

        return

    def compute_ik(self, target_position, target_orientation):

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=
            "/omniisaacgymenvs/cfg/robot/robot_descriptor.yaml",
            urdf_path="/omniisaacgymenvs/assests/robots/ur16e/ur16e.urdf")
        from omni.isaac.core.articulations import Articulation

        # robot_prim_path = "/panda"
        # path_to_robot_usd = self.current_directory + self._task_cfg['sim']["URRobot"]['robot_path']

        # add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        # articulation = Articulation(self.default_zero_env_path + "/robot")
        # self._robots = ArticulationView(
        #     prim_paths_expr="/World/envs/.*/robot", name="robot_view", reset_xform_properties=False)
        from omni.isaac.core.robots.robot import Robot
        _robot = Robot(self.default_zero_env_path + "/robot")
        _robot.initialize()

        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            _robot, self._kinematics_solver, "ee_link")

        self._kinematics_solver.set_robot_base_pose(
            robot_position=np.array(self._robot_positions),
            robot_orientation=np.array(self._robot_rotations))

        joint_positions, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation)
        return joint_positions, success

    def add_gripper(self):
        assets_root_path = get_assets_root_path()

        gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"

        self.grippers = []

        for i in range(self.num_envs):

            add_reference_to_stage(
                usd_path=gripper_usd,
                prim_path=f"/World/envs/env_{i}/robot/ee_link")

            surface_gripper = SurfaceGripper(
                end_effector_prim_path=f"/World/envs/env_{i}/robot/ee_link",
                translate=0.1611,
                direction="y")
            surface_gripper.set_force_limit(value=8.0e1)
            surface_gripper.set_torque_limit(value=10.0e0)
            # surface_gripper.initialize(physics_sim_view=None, articulation_num_dofs=self.robot.num_dof)
            self.grippers.append(surface_gripper)

    def load_robot(self):

        from omniisaacgymenvs.robots.articulations.ur10 import UR10

        self.robot = UR10(prim_path=self.default_zero_env_path + "/robot",
                          name="robot",
                          position=self._robot_positions,
                          orientation=self._robot_rotations,
                          attach_gripper=False,
                          usd_path=self.current_directory +
                          self._task_cfg['sim']["URRobot"]['robot_path'])

        self.robot.set_joint_positions(self._robot_dof_target)
        self.robot.set_joints_default_state(self._robot_dof_target)

        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(self.robot.prim_path),
            self._sim_config.parse_actor_config("robot"))

    def load_sphere(self):

        target = DynamicSphere(prim_path=self.default_zero_env_path +
                               "/target",
                               name="target",
                               radius=0.025,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings(
            "target", get_prim_at_path(target.prim_path),
            self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)

    def load_object(self,
                    usd_path,
                    env_index,
                    object_index,
                    translaton=[-0.69, 0.1, 1.3],
                    orientation=[0, 0, 0.707, 0.707],
                    scale=[0.4, 0.4, 0.4]):

        # ================================= load object ========================================
        prim_utils.create_prim(f"/World/envs/env_{env_index}" +
                               f"/manipulated_object_{object_index}",
                               usd_path=usd_path,
                               translation=translaton,
                               orientation=orientation,
                               scale=scale)

        stage = omni.usd.get_context().get_stage()
        object_prim = stage.GetPrimAtPath(
            f"/World/envs/env_{env_index}" +
            f"/manipulated_object_{object_index}")

        # ================================= set property ========================================
        # Make it a rigid body
        # utils.setRigidBody(object_prim, "convexHull", True)
        # # mass_api = UsdPhysics.MassAPI.Apply(object_prim)
        # # mass_api.CreateMassAttr(10)
        # # # Alternatively set the density
        # # mass_api.CreateDensityAttr(1000)
        # UsdPhysics.CollisionAPI.Apply(object_prim)

        # self._sim_config.apply_rigid_body_settings("Object", object_prim.GetPrim(),self._sim_config.parse_actor_config("Object"),is_articulation=False)
        # Make it a rigid body with kinematic
        # utils.setRigidBody(object_prim, "convexMeshSimplification", True)

        # mass_api = UsdPhysics.MassAPI.Apply(object_prim)
        # mass_api.CreateMassAttr(10)
        # # Alternatively set the density
        # mass_api.CreateDensityAttr(1000)
        # UsdPhysics.CollisionAPI.Apply(object_prim)

        # ================================= set property ========================================
        # Make it a rigid body with kinematic
        # utils.setRigidBody(object_prim, "convexMeshSimplification", True)

        # mass_api = UsdPhysics.MassAPI.Apply(object_prim)
        # mass_api.CreateMassAttr(10)
        # # Alternatively set the density
        # mass_api.CreateDensityAttr(1000)
        UsdPhysics.CollisionAPI.Apply(object_prim)
        self._sim_config.apply_rigid_body_settings(
            "Object",
            object_prim.GetPrim(),
            self._sim_config.parse_actor_config("Object"),
            is_articulation=False)

        # ================================= add texture ========================================
        # Change the server to your Nucleus install, default is set to localhost in omni.isaac.sim.base.kit
        default_server = carb.settings.get_settings().get(
            "/persistent/isaac/asset_root/default")
        mtl_created_list = []
        # Create a new material using OmniPBR.mdl
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=mtl_created_list,
        )
        stage = omni.usd.get_context().get_stage()
        mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])
        # Set material inputs, these can be determined by looking at the .mdl file
        # or by selecting the Shader attached to the Material in the stage window and looking at the details panel
        omni.usd.create_material_input(
            mtl_prim,
            "diffuse_texture",
            default_server +
            "/Isaac/Samples/DR/Materials/Textures/marble_tile.png",
            Sdf.ValueTypeNames.Asset,
        )

        # Bind the material to the prim
        cube_mat_shade = UsdShade.Material(mtl_prim)
        UsdShade.MaterialBindingAPI(object_prim).Bind(
            cube_mat_shade, UsdShade.Tokens.strongerThanDescendants)

    def load_pod(self):
        for i in range(self.num_envs):

            # add_reference_to_stage(
            #         prim_path=f"/World/envs/env_{i}" + "/pod",
            #         usd_path = "/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/pod/pod.usd",
            # )

            # initila toe orientation
            from scipy.spatial.transform import Rotation
            # Define the Euler angles in the format (roll, pitch, yaw)
            euler_angles = [np.pi / 2, np.pi / 2, np.pi / 2]

            # Create a Rotation object from the Euler angles
            r = Rotation.from_euler('xyz', euler_angles, degrees=False)

            # Get the corresponding quaternion
            quaternion = r.as_quat()

            prim_utils.create_prim(
                prim_path=f"/World/envs/env_{i}" + "/pod",
                usd_path=self.current_directory +
                "/omniisaacgymenvs/assests/robots/pod/pod.usd",
                translation=[-0.463, 0.95, 0],
                # orientation=quaternion,
            )
            stage = omni.usd.get_context().get_stage()
            cube_prim = stage.GetPrimAtPath(f"/World/envs/env_{i}" + "/pod")

            # ================================= set property ========================================
            # Make it a rigid body with kinematic
            utils.setRigidBody(cube_prim, "convexMeshSimplification", True)

            mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
            mass_api.CreateMassAttr(10)
            # Alternatively set the density
            mass_api.CreateDensityAttr(1000)
            UsdPhysics.CollisionAPI.Apply(cube_prim)

    def load_table(self):
        table_translation = np.array(
            self._task_cfg['sim']["Table"]["position"])
        table_orientation = np.array(
            self._task_cfg['sim']["Table"]["quaternion"])

        table = FixedCuboid(
            prim_path=self.default_zero_env_path + "/table",
            name="table",
            translation=table_translation,
            orientation=table_orientation,
            scale=np.array(self._task_cfg['sim']["Table"]["scale"]),
            size=1.0,
            color=np.array([1, 197 / 255, 197 / 255]),
        )
        table_usd_path = f"{nucleus_utils.get_assets_root_path()}/NVIDIA/Assets/ArchVis/Residential/Furniture/Tables/Whittershins.usd"
        # fix table base
        # table = prim_utils.create_prim(self.default_zero_env_path + "/table",
        #                                usd_path=table_usd_path,
        #                                translation=table_translation,
        #                                scale=(0.005, 0.005, 0.0202))
        table_prim = get_prim_at_path(self.default_zero_env_path + "/table")

        # utils.setRigidBody(table_prim, "convexHull", True)
        # UsdPhysics.CollisionAPI.Apply(table_prim)

        # x = UsdShade.MaterialBindingAPI.Apply(table_prim)
        # from omni.isaac.core.materials.physics_material import PhysicsMaterial
        # material = PhysicsMaterial(
        #     prim_path="/World/PhysicsMaterials/FrankaFingerMaterial",
        #     name="franka_finger_material_physics",
        #     static_friction=0.7,
        #     dynamic_friction=0.0,
        # )
        # x.Bind(
        #     material.material,
        #     bindingStrength="weakerThanDescendants",
        #     materialPurpose="physics",
        # )

        self._sim_config.apply_rigid_body_settings(
            "table",
            table_prim,
            self._sim_config.parse_actor_config("table"),
            is_articulation=False)

    def load_cube(self):
        self.scale_size = torch.as_tensor(
            self._task_cfg["sim"]["Object"]["scale"]).repeat(self.num_envs,
                                                             1).to(self.device)
        for i in range(self.num_envs):

            target_object_1 = DynamicCuboid(
                prim_path=f"/World/envs/env_{i}/manipulated_object_1",
                name="manipulated_object_1",
                position=[0, 0, 2.02],
                # size=0.2,
                scale=np.array(self._task_cfg["sim"]["Object"]["scale"]),
                color=torch.tensor([0, 169 / 255, 1]))

            self._sim_config.apply_articulation_settings(
                "table", get_prim_at_path(target_object_1.prim_path),
                self._sim_config.parse_actor_config("manipulated_object_1"))

    def load_manipulated_object(self):

        object_dir = self.current_directory + "/omniisaacgymenvs/assests/objects/shapenet_nomat/" + self._task_cfg[
            'sim']["Object"]["category"]
        object_list = os.listdir(object_dir)

        for i in range(self.num_envs):
            object_name = object_list[i]  # np.random.choice(object_list)

            object_path = object_dir + "/" + object_name + "/model_normalized_nomat.usd"
            self.load_object(usd_path=object_path, env_index=i, object_index=1)
            self.object_prim_path.append(object_path)

            # object_path = object_dir + "/" + np.random.choice(object_list) + "/model_normalized_nomat.usd"
            # self.load_object(usd_path=object_path,env_index=i,object_index=2)

    def lock_motion(self, stage, joint_path, prim_path, ee_link, i, lock=True):
        from pxr import UsdPhysics, Gf
        # D6 fixed joint
        d6FixedJoint = UsdPhysics.Joint.Define(stage, joint_path)
        d6FixedJoint.CreateBody0Rel().SetTargets(
            [f"/World/envs/env_{i}/manipulated_object_1"])
        d6FixedJoint.CreateBody1Rel().SetTargets([prim_path])

        d6FixedJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.26, 0.0, 0))
        d6FixedJoint.CreateLocalRot0Attr().Set(Gf.Quatf(
            1.0, Gf.Vec3f(0, 0, 0)))

        d6FixedJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))

        d6FixedJoint.CreateLocalRot1Attr().Set(Gf.Quatf(
            1.0, Gf.Vec3f(0, 0, 0)))
        # lock all DOF (lock - low is greater than high)
        d6Prim = stage.GetPrimAtPath(joint_path)

        if lock:
            for name in ["transX", "transY", "transZ"]:
                limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, name)
                limitAPI.CreateLowAttr(1.0)
                limitAPI.CreateHighAttr(-1.0)

            # for limit_name in ["rotX", "rotY", "rotZ"]:
            #     limit_api = UsdPhysics.LimitAPI.Apply(d6Prim, limit_name)

            #     limit_api.CreateLowAttr(-45.0)
            #     limit_api.CreateHighAttr(45.0)

    def unlock_motion(self, fixed_joint_path):
        for i in range(self._num_envs):
            is_valid = is_prim_path_valid(fixed_joint_path)
            if is_valid:
                delete_prim(fixed_joint_path)

    def get_observations(self) -> dict:

        _, current_orientation = self._end_effector.get_world_poses()

        # current_euler_angles = quaternion_to_axis_angle(current_orientation)
        # quaternion = quaternion_multiply(
        #     quaternion_invert(self.init_ee_link_orientation),
        #     current_orientation)
        # self.current_euler_angles = quaternion_to_axis_angle(quaternion)

        _wrist2_local_pos, _ = self.wrist_2_link.get_local_poses()
        _ee_local_pos, _ = self._end_effector.get_local_poses()

        current_euler_angles_x = torch.atan2(
            _ee_local_pos[:, 1] - _wrist2_local_pos[:, 1],
            _ee_local_pos[:, 0] - _wrist2_local_pos[:, 0])

        self.angle_dev_target = torch.atan2(
            _ee_local_pos[:, 2] - _wrist2_local_pos[:, 2],
            torch.linalg.norm(_ee_local_pos[:, :2] - _wrist2_local_pos[:, :2],
                              dim=1))

        self.angle_dev = (current_euler_angles_x -
                          torch.pi / 2) - self.target_angle
        cur_position, _ = self._end_effector.get_local_poses()

        cur_position[:, 0] = -cur_position[:, 0]

        self.dist_dev = torch.linalg.norm(self.target_position - cur_position,
                                          dim=1)

        # start = time.time()
        if self._cfg["raycast"]:
            self.raytrace_step()
        # print("time",time.time()-start)

        # self.render_curobo()
        joint_angle = self._robots.get_joint_positions()

        if self._task_cfg['Training']["use_oracle"]:
            self.obs_buf = torch.cat([
                current_euler_angles_x[:, None], self.target_angle[:, None],
                self.angle_dev[:, None], cur_position, self.target_position,
                self.target_position - cur_position, joint_angle
            ],
                                     dim=1)

        elif self._cfg["raycast"]:
            self.obs_buf = torch.cat([
                self.raytrace_dist,
                cur_position,
                self.target_position,joint_angle
            ],
                                     dim=1)

        return self.obs_buf

    def update_cache_state(self):
        self._q = self._robots.get_joint_positions()
        self._qd = self._robots.get_joint_velocities()
        self._mm = self._robots.get_mass_matrices()
        self._robots.set_gains(kds=torch.zeros((self.num_envs, 6)),
                               kps=torch.zeros((self.num_envs, 6)))
        self._j_eef = self._robots.get_jacobians(clone=False)[:, 6, :, :]
        self.kp = torch.tensor([150., 150., 150., 100., 100., 100.],
                               device=self.device) / 100
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = torch.tensor([10.] * 6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        self.num_ur16e_dofs = 6
        self.ur16e_default_dof_pos = self.default_dof

        self._ee_vel = torch.concat([
            self._robots.get_linear_velocities(),
            self._robots.get_angular_velocities()
        ],
                                    dim=1)
        self._ur16e_effort_limits = self._robots.get_max_efforts()

    def recover_action(self, action, limit):

        # delta pose
        action = torch.clip(action, -1, 1)
        # self.pre_action[:, 5] = action.reshape(-1) * 0
        self.pre_action[:, [0, 1, 2, 3, 4, 5]] = action

        # action[:,[0,1,2,3,4]] = 0 # rotate along z axis to rotation

        delta_pose = (self.pre_action + 1) / 2 * (limit[:, 1] -
                                                  limit[:, 0]) + limit[:, 0]
        self.control_time = self._env._world.get_physics_dt()
        delta_pose = delta_pose * self.control_time

        self.jacobians = self._robots.get_jacobians(clone=False)
        delta_dof_pos = self.ik(jacobian_end_effector=self.jacobians[:,
                                                                     6, :, :],
                                delta_pose=delta_pose)
        delta_dof_pos = torch.clip(delta_dof_pos, -torch.pi, torch.pi)

        return delta_dof_pos, delta_pose

    def ik(self, jacobian_end_effector, delta_pose, damping_factor=0.05):
        """
            Damped Least Squares method: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
            """

        # compute position and orientation error
        delta_pose = delta_pose[:, :, None]

        # solve damped least squares (dO = J.T * V)
        transpose = torch.transpose(jacobian_end_effector, 1, 2)
        lmbda = torch.eye(6).to(jacobian_end_effector.device) * (damping_factor
                                                                 **2)
        return (transpose @ torch.inverse(jacobian_end_effector @ transpose +
                                          lmbda) @ delta_pose).squeeze(dim=2)

    def raytrace_step(self) -> None:

        self.debug_draw.clear_lines()
        self.debug_draw.clear_points()

        gripper_pose, gripper_rot = self._end_effector.get_world_poses()

        self.target_object_pose, self.target_object_rot = self._manipulated_object.get_world_poses(
        )
        # transform = Transform3d(device=self.device).scale(
        #     self.scale_size).rotate(
        #         quaternion_to_matrix(quaternion_invert(
        #             self.target_object_rot))).translate(
        #                 self.target_object_pose)

        # transformed_vertices = transform.transform_points(
        #     self.mesh_vertices.clone().to(self.device))
        _, _, transformed_vertices = self.transform_mesh()

        # PT Multiple sensors TODO need to move this to utils file
        normals = find_plane_normal(self.num_envs, gripper_rot)
        circle = circle_points(
            self._task_cfg['sim']["URRobot"]['sensor_radius'], gripper_pose,
            normals, self._task_cfg['sim']["URRobot"]['num_sensors'])

        # for draw point
        debug_sensor_ray_pos_list = []
        debug_ray_hit_points_list = []
        debug_ray_colors = []
        debug_ray_sizes = []
        debug_point_sizes = []
        debug_end_point_colors = []
        debug_start_point_colors = []
        debug_circle = []

        self.raytrace_dist = torch.zeros((self.num_envs, 2)).to(self.device)

        for i, env in zip(
                torch.arange(
                    self._task_cfg['sim']["URRobot"]['num_sensors']).repeat(
                        self.num_envs),
                torch.arange(self.num_envs).repeat_interleave(
                    self._task_cfg['sim']["URRobot"]['num_sensors'])):

            # trimesh_1 = geom_to_trimesh(
            #     UsdGeom.Cube(
            #         get_prim_at_path(
            #             self._manipulated_object.prim_paths[env])),
            #     self.target_object_pose[env].cpu(),
            #     transformations.euler_from_quaternion(
            #         self.target_object_rot[env].cpu()))  #TODO Why to CPU?
            # warp_mesh = warp_from_trimesh(trimesh_1, self._device)

            self.raytracer.set_geom(wp.from_torch(transformed_vertices[env]),
                                    mesh_index=0)
            ray_t, ray_dir, normal = self.raytracer.render(
                circle[env][i], gripper_rot[env])

            ray_t = wp.torch.to_torch(ray_t)

            if len(torch.where(ray_t > 0)[0]) > 0:
                average_distance = torch.mean(ray_t[torch.where(ray_t > 0)])
            else:
                average_distance = -0.01
            self.raytrace_dist[env][i] = average_distance
            # standard_deviation = math.sqrt(
            #     max(average_distance * 100 * 0.4795 - 3.2018, 0))
            # noise_distance = np.random.normal(average_distance * 1000,
            #                                     standard_deviation)

            if self._cfg["debug"]:
                sensor_ray_pos_np = circle[env][i].cpu().numpy()
                sensor_ray_pos_tuple = (sensor_ray_pos_np[0],
                                        sensor_ray_pos_np[1],
                                        sensor_ray_pos_np[2])
                ray_t = ray_t.cpu().numpy()
                ray_dir = ray_dir.numpy()

                line_vec = np.transpose(
                    np.multiply(np.transpose(ray_dir), ray_t))

                # print(
                #     f'distance with noise sensor {i}: , {average_distance*100}'
                # )

                # Get rid of ray misses (0 values)
                line_vec = line_vec[np.any(line_vec, axis=1)]
                ray_hit_points_list = line_vec + np.array(sensor_ray_pos_tuple)
                hits_len = len(ray_hit_points_list)

                if hits_len > 0:
                    sensor_ray_pos_list = [
                        sensor_ray_pos_tuple for _ in range(hits_len)
                    ]
                    ray_colors = [(1, i, 0, 1) for _ in range(hits_len)]
                    ray_sizes = [2 for _ in range(hits_len)]
                    point_sizes = [7 for _ in range(hits_len)]
                    start_point_colors = [
                        (0, 0.75, 0, 1) for _ in range(hits_len)
                    ]  # start (camera) points: green
                    end_point_colors = [(1, i, 1, 1) for _ in range(hits_len)]

                    debug_sensor_ray_pos_list.append(sensor_ray_pos_list)
                    debug_ray_hit_points_list.append(ray_hit_points_list)
                    debug_ray_colors.append(ray_colors)
                    debug_ray_sizes.append(ray_sizes)

                    debug_end_point_colors.append(end_point_colors)
                    debug_point_sizes.append(point_sizes)
                    debug_start_point_colors.append(start_point_colors)

                    debug_circle.append([circle[env][i].cpu().numpy()])

        if self._cfg["debug"]:

            if len(debug_sensor_ray_pos_list) > 0:

                debug_sensor_ray_pos_list = np.concatenate(
                    debug_sensor_ray_pos_list, axis=0)
                debug_ray_hit_points_list = np.concatenate(
                    debug_ray_hit_points_list, axis=0)
                debug_ray_colors = np.concatenate(debug_ray_colors, axis=0)
                debug_ray_sizes = np.concatenate(debug_ray_sizes, axis=0)
                debug_end_point_colors = np.concatenate(debug_end_point_colors,
                                                        axis=0)
                debug_point_sizes = np.concatenate(debug_point_sizes, axis=0)
                debug_start_point_colors = np.concatenate(
                    debug_start_point_colors, axis=0)
                debug_circle = np.concatenate(debug_circle, axis=0)

                self.debug_draw.draw_lines(debug_sensor_ray_pos_list,
                                           debug_ray_hit_points_list,
                                           debug_ray_colors, debug_ray_sizes)
                self.debug_draw.draw_points(debug_ray_hit_points_list,
                                            debug_end_point_colors,
                                            debug_point_sizes)
                self.debug_draw.draw_points(debug_sensor_ray_pos_list,
                                            debug_start_point_colors,
                                            debug_point_sizes)
                # Debug draw the gripper pose
                self.debug_draw.draw_points(debug_circle, [(1, 0, 0, 1)], [10])

        # if self._cfg[
        #         "debug_with_trimesh"]:  #TODO Update with sensor circle
        #     ## Visualization for trimesh
        #     # Create axis for visualization
        #     axis_origins = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        #     axis_directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        #     # stack axis rays into line segments for visualization as Path3D
        #     axis_visualize = trimesh.load_path(np.hstack(
        #         (axis_origins,
        #          axis_origins + axis_directions)).reshape(-1, 2, 3),
        #                                        colors=np.array(
        #                                            [[0, 0, 255, 255],
        #                                             [0, 255, 0, 255],
        #                                             [255, 0, 0, 255]]))
        #     # stack rays into line segments for visualization as Path3D
        #     ray_origins = np.repeat(np.expand_dims(gripper_pose.cpu()[0],
        #                                            axis=0),
        #                             repeats=256,
        #                             axis=0)
        #     ray_visualize = trimesh.load_path(
        #         np.hstack(
        #             (ray_origins,
        #              ray_origins + ray_dir.numpy())).reshape(-1, 2, 3))
        #     # trimesh_1.apply_transform(matrix)
        #     self.j = self.j + 1
        #     if self.j % 25 == 0:
        #         scene = trimesh.Scene(
        #             [trimesh_1, ray_visualize, axis_visualize])

        #         # display the scene
        #         scene.show()

    def get_target_pose(self):

        target_x = 0.3 * torch.sin(torch.as_tensor(self.target_angle)).to(
            self.device) + self.init_ee_local_pos[:, 0]

        target_y = 0.3 * (1 - torch.cos(torch.as_tensor(
            self.target_angle))).to(self.device) + self.init_ee_local_pos[:, 1]

        self.target_position = torch.cat([
            -target_x[:, None], target_y[:, None],
            self.init_ee_local_pos[:, 2][:, None]
        ],
                                         dim=1)

        self.init_dist = torch.linalg.norm(self.target_position[:, :2] -
                                           self.init_ee_local_pos[:, :2],
                                           dim=1)

    def recover_rule_based_action(self):

        delta_pose = torch.zeros((self.num_envs, 6)).to(self.device)

        delta_pose[:, 5] = torch.as_tensor(torch.pi / 200)

        # target_x = 0.4 * torch.sin(torch.as_tensor(self.target_angle)).to(
        #     self.device) + self.init_ee_local_pos[:, 0]

        # target_y = 0.4 * (1 - torch.cos(torch.as_tensor(
        #     self.target_angle))).to(self.device) + self.init_ee_local_pos[:, 1]

        cur_pos, _ = self._end_effector.get_local_poses()
        cur_pos[:, 0] = -cur_pos[:, 0]

        delta_pose[:, 0] = self.target_position[:, 0] - cur_pos[:, 0]
        delta_pose[:, 1] = self.target_position[:, 1] - cur_pos[:, 1]

        satified_index = torch.where(abs(self.angle_dev) < 0.02)[0]

        if torch.numel(satified_index) != 0:
            delta_pose[satified_index, 5] = 0

        self.jacobians = self._robots.get_jacobians(clone=False)
        delta_dof_pos = self.ik(jacobian_end_effector=self.jacobians[:,
                                                                     8, :, :],
                                delta_pose=delta_pose)

        return delta_dof_pos, delta_pose

    def pre_physics_step(self, actions) -> None:

        dof_limits = self._robots.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :,
                                                 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :,
                                                 1].to(device=self._device)

        self._step += 1
        if not self._env._world.is_playing():
            return

        if not self._task_cfg["sim"]["Control"]["rule-base"]:

            actions = actions.to(self._device)
            delta_dof_pos, delta_pose = self.recover_action(
                actions, self.velocity_limit)
        else:
            delta_dof_pos, delta_pose = self.recover_rule_based_action()

        # current dof and current joint velocity
        current_dof = self._robots.get_joint_positions()
        targets_dof = current_dof + delta_dof_pos[:, :6]

        # targets_dof = torch.clamp(targets_dof, self.robot_dof_lower_limits,
        #                           self.robot_dof_upper_limits)

        # targets_dof[:, -2] = torch.clamp(targets_dof[:, -2], -torch.pi / 2,
        #                                  torch.pi / 2)

        targets_dof[:, -1] = 0

        self._robots.set_joint_position_targets(targets_dof)

        pre_position, pre_orientation = self._end_effector.get_local_poses()

        for i in range(1):
            self._env._world.step(render=False)

    def render_curobo(self):

        # frame skip
        ee_translation_goal, ee_orientation_teleop_goal = self._manipulated_object.get_local_poses(
        )

        ee_world_pose, _ = self._manipulated_object.get_world_poses()

        self.goal_pose.position[:] = torch.as_tensor([-0.4, 0.8, 1.4]).to(
            self.device) + self.position_grid_offset
        self.goal_pose.quaternion[:] = torch.as_tensor([1, 0, 0, 0]).to(
            self.device)  # ik_goal.quaternion[:]
        # tranformed
        pre_transoform_goal_pose = self.goal_pose.clone()

        pre_transoform_goal_pose.position[:, 0] += ee_world_pose[
            0, 0] - ee_translation_goal[0, 0]

        transform = Transform3d(device=self.device).rotate(
            quaternion_to_matrix((torch.as_tensor(
                self._robot_rotations).to(device=self.device))))

        # rotation_matrix = quaternion_to_matrix(
        #     (torch.as_tensor(orientation).to(device)))

        self.goal_pose.position[:] -= torch.as_tensor(
            [self._robot_positions]).to(device=self.device)

        self.goal_pose.position[:] = transform.transform_points(
            self.goal_pose.position[:].clone().to(device=self.device))

        self.goal_pose.quaternion[:] = quaternion_multiply(
            quaternion_invert(
                torch.as_tensor(self._robot_rotations).to(device=self.device)),
            self.goal_pose.quaternion[:])

        result = self.curo_ik_solver.solve_batch(self.goal_pose)

        # succ = torch.any(result.success)

        # get spheres and flags:

        # transform back

        self.draw_points(pre_transoform_goal_pose, result.success)

    def draw_points(self, pose, success):
        # Third Party
        from omni.isaac.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
        N = 100
        # if draw.get_num_points() > 0:
        draw.clear_points()
        cpu_pos = pose.position.cpu().numpy()
        b, _ = cpu_pos.shape
        point_list = []
        colors = []
        for i in range(b):
            # get list of points:

            if success[i].item():
                colors += [(0, 1, 0, 0.25)]
                point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
            # else:
            #     colors += [(1, 0, 0, 0.25)]
            #     point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        sizes = [60.0 for _ in range(len(colors))]

        draw.draw_points(point_list, colors, sizes)

    def transform_mesh(self):
        self.target_object_pose, self.target_object_rot = self._manipulated_object.get_world_poses(
        )
        transform = Transform3d(device=self.device).scale(
            self.scale_size).rotate(
                quaternion_to_matrix(quaternion_invert(
                    self.target_object_rot))).translate(
                        self.target_object_pose)

        transformed_vertices = transform.transform_points(
            self.mesh_vertices.clone().to(self.device))

        max_xyz = torch.max(transformed_vertices, dim=1).values
        min_xyz = torch.min(transformed_vertices, dim=1).values
        bboxes = torch.hstack([min_xyz, max_xyz])
        center_points = (max_xyz + min_xyz) / 2

        return bboxes, center_points, transformed_vertices

    def post_reset(self):

        self.robot.initialize()
        self.robot.disable_gravity()

        for i in range(self.num_envs):
            self.grippers[i].initialize(
                articulation_num_dofs=self._robots.num_dof)
        self.reset()
      

    def calculate_angledev_reward(self) -> None:

        dev_percentage = self.angle_dev / self.init_angle_dev

        # exceed the target
        negative_index = torch.where(dev_percentage < 0)[0]
        if not negative_index.size()[0] == 0:
            dev_percentage[negative_index] = abs(
                dev_percentage[negative_index]) + 1

        action_penalty = torch.sum(torch.clamp(
            self._robots.get_joint_velocities() - 1, 1),
                                   dim=1) * -0.0

        dev = torch.clamp(dev_percentage, 0, 1.8)

        angle_reward = abs((1 - dev)**2) * 3

        negative_index = torch.where(dev > 1)[0]

        angle_reward[negative_index] = -abs((1 - dev[negative_index])**2) * 3
        return angle_reward

    def calculate_targetangledev_reward(self) -> None:

        angle_reward = -abs(self.angle_dev_target) * 7

        return angle_reward

    def calculate_dist_reward(self) -> None:

        dev_percentage = self.dist_dev / self.init_dist

        # exceed the target
        negative_index = torch.where(dev_percentage < 0)[0]
        if not negative_index.size()[0] == 0:
            dev_percentage[negative_index] = abs(
                dev_percentage[negative_index]) + 1

        action_penalty = torch.sum(torch.clamp(
            self._robots.get_joint_velocities() - 1, 1),
                                   dim=1) * -0.0

        dev = torch.clamp(dev_percentage, 0, 1.8)

        dist_reward = abs((1 - dev)**2) * 3

        negative_index = torch.where(dev > 1)[0]

        dist_reward[negative_index] = -abs((1 - dev[negative_index])**2) * 3

        return dist_reward

    def calculate_metrics(self) -> None:

        self.rew_buf = self.calculate_dist_reward()
        self.rew_buf += self.calculate_angledev_reward()
        self.rew_buf += self.calculate_targetangledev_reward()

        return self.rew_buf

    def is_done(self) -> None:

        # return torch.full((self.num_envs,), 0, dtype=torch.int)
        

        if (self._step + 1) % 201 == 0:
            self._step = 0
            self.post_reset()
            return [True for i in range(self.num_envs)]

        return [False for i in range(self.num_envs)]

    def reset_internal(self):

        self.scene.remove_object("robot_view")

        self.load_robot()
        # self.load_sphere()
        # self.load_manipulated_object()

        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/robot",
                                        name="robot_view",
                                        reset_xform_properties=False)
        self.scene.add(self._robots)

    def reset(self):
        # # initial robot
        target_joint_positions = torch.zeros(6, device=self.device)
        target_joint_positions[0] = 0
        target_joint_positions[1] = -1.57
        target_joint_positions[2] = 1.57 / 2 * 2
        target_joint_positions[3] = -1.57 * 2
        target_joint_positions[4] = 0
        random_values = torch.randint(low=0,
                                      high=len(self.init_robot_joints),
                                      size=(self.num_envs, ))

        self.target_joint_positions = torch.tensor(
            self.init_robot_joints[random_values], dtype=torch.float)

        self._robots.set_joint_positions(
            torch.tensor(target_joint_positions,
                         dtype=torch.float).repeat(self.num_envs, 1))

        for i in range(1):
            self._env._world.step(render=False)
        self.init_ee_link_position, self.init_ee_link_orientation = self._end_effector.get_world_poses(
        )

        # init object location
        # random orientation
        target_obj_position, _ = self._end_effector.get_world_poses()  # wxyz
        rand_ori_z = torch.rand(self.num_envs).to(self.device) / 2 + 0.2
        self.rand_orientation = torch.zeros((self.num_envs, 3)).to(self.device)
        self.rand_orientation[:, 2] = rand_ori_z * torch.pi / 2 / 0.7 * 0.5
        object_target_quaternion = tf.axis_angle_to_quaternion(
            self.rand_orientation)

        # init position
        object_target_position = target_obj_position.clone()
        object_target_position[:, 1] += 0.3
        # object_target_position[:, 0] += 0.1

        self._manipulated_object.set_world_poses(object_target_position,
                                                 object_target_quaternion)
        self.default_dof = torch.tensor(target_joint_positions,
                                        dtype=torch.float).repeat(
                                            self.num_envs, 1).clone()

        # init table position
        table_position, _ = self._table.get_world_poses()
        table_position[:, 0] = self.init_ee_link_position[:, 0]
        self._table.set_world_poses(table_position)

        for i in range(10):
            self._env._world.step(render=False)

        self.init_ee_local_pos, _ = self._end_effector.get_local_poses()

        top_ee_local_pose = self.init_ee_local_pos.clone()
        top_ee_local_pose[:, 1] += 0.16
        self._end_effector_points = torch.cat(
            [self.init_ee_local_pos[None], top_ee_local_pose[None]], dim=1)

        # reset goal orientation

        self.target_angle = torch.zeros(self.num_envs, device=self.device)
        self.target_angle = -self.rand_orientation[:, 2].clone()
        self.init_angle_dev = -self.target_angle.clone()
        self.get_target_pose()
        self._step = 0

        # self.unlock_motion(f"/World/envs/env_{0}/robot/ee_link_cube")

    def reset_raytracer(self):
        self.target_object_pose, self.target_object_rot = self._manipulated_object.get_world_poses(
            clone=True)
        trimesh_1 = geom_to_trimesh(
            UsdGeom.Cube(
                get_prim_at_path(self._manipulated_object.prim_paths[0])),
            self.target_object_pose[0].cpu(),
            self.target_object_rot[0].cpu())  # TODO Why to CPU?
        warp_mesh = warp_from_trimesh(trimesh_1, self._device)

        if self._cfg["debug_with_trimesh"]:
            self.j = 0


def circle_points(radius, centers, normals, num_points):
    """
    Generate points on a batch of circles in 3D space.

    Args:
    radius (float): The radius of the circles.
    centers (torch.Tensor): a tensor of shape (batch_size, 3) representing the centers of the circles.
    normals (torch.Tensor): a tensor of shape (batch_size, 3) representing the normals to the planes of the circles.
    num_points (int): The number of points to generate on each circle.

    Returns:
    torch.Tensor: a tensor of shape (batch_size, num_points, 3) representing the points on the circles.
    """
    batch_size = centers.shape[0]

    # Normalize the normal vectors
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)

    # Generate random vectors not in the same direction as the normals
    not_normals = torch.rand(batch_size, 3, device='cuda:0')
    while (normals * not_normals).sum(
            dim=-1).max() > 0.99:  # Ensure they're not too similar
        not_normals = torch.rand(batch_size, 3, device='cuda:0')

    # Compute the basis of the planes
    basis1 = torch.cross(normals, not_normals)
    basis1 /= torch.norm(basis1, dim=-1, keepdim=True)
    basis2 = torch.cross(normals, basis1)
    basis2 /= torch.norm(basis2, dim=-1, keepdim=True)

    # Generate points on the circles
    t = torch.arange(0,
                     2 * torch.pi,
                     step=2 * torch.pi / num_points,
                     device='cuda:0')
    circles = centers[:, None, :] + radius * (
        basis1[:, None, :] * torch.cos(t)[None, :, None] +
        basis2[:, None, :] * torch.sin(t)[None, :, None])
    return circles


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a batch of quaternions to rotation matrices.

    Args:
    quaternion (torch.Tensor): a tensor of shape (batch_size, 4) representing the quaternions.

    Returns:
    torch.Tensor: a tensor of shape (batch_size, 3, 3) representing the rotation matrices.
    """
    w, x, y, z = quaternion.unbind(dim=-1)

    batch_size = quaternion.shape[0]

    rotation_matrix = torch.empty((batch_size, 3, 3), device='cuda:0')

    rotation_matrix[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    rotation_matrix[:, 0, 1] = 2 * x * y - 2 * z * w
    rotation_matrix[:, 0, 2] = 2 * x * z + 2 * y * w
    rotation_matrix[:, 1, 0] = 2 * x * y + 2 * z * w
    rotation_matrix[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    rotation_matrix[:, 1, 2] = 2 * y * z - 2 * x * w
    rotation_matrix[:, 2, 0] = 2 * x * z - 2 * y * w
    rotation_matrix[:, 2, 1] = 2 * y * z + 2 * x * w
    rotation_matrix[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2

    return rotation_matrix


def find_plane_normal(num_env, quaternions):
    """
    Find the normal to a plane defined by a batch of points and rotations.

    Args:
    num_env: 
    quaternions (torch.Tensor): a tensor of shape (batch_size, 4) representing the rotations.

    Returns:
    torch.Tensor: a tensor of shape (batch_size, 3) representing the normals to the planes.
    """
    # Convert the quaternions to rotation matrices
    rotation_matrices = quaternion_to_rotation_matrix(quaternions)
    normals = torch.tensor([1.0, 0.0, 0.0],
                           device='cuda:0').expand(num_env, -1)
    normals = normals.view(num_env, 3, 1)
    rotated_normals = torch.bmm(rotation_matrices, normals)
    return rotated_normals.view(num_env, 3)
