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

# init extension
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.debug_draw")
enable_extension("omni.isaac.manipulators")
enable_extension("omni.isaac.motion_generation")

# import env setting
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.ur_robot import UR
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import FixedCuboid
from omniisaacgymenvs.robots.articulations.surface_gripper import SurfaceGripper

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, is_prim_path_valid

from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni.isaac.core.utils.prims as prim_utils

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf
from pxr import UsdPhysics
from omni.physx.scripts import utils
from omni.physx import acquire_physx_interface
from omni.isaac.debug_draw import _debug_draw

import omni
import carb

import omni.isaac.core.utils.nucleus as nucleus_utils
from omniisaacgymenvs.utils.tools.rotation_conversions import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
import omniisaacgymenvs.utils.tools.transform_utils as tf

from .raycast import Raycast

from omniisaacgymenvs.controller.ik import recover_action, recover_rule_based_action, diffik
from omniisaacgymenvs.controller.curobo import MotionGeneration

# import util package
import numpy as np
import torch
import math
import trimesh
import os
import gym
import warp as wp
from cprint import *
import time
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion


class TofSensorTask(RLTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:

        # config info
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._device = self._cfg["rl_device"]

        # env info
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = self._task_cfg["env"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["num_actions"]

        RLTask.__init__(self, name, env)

        # robot info
        self._robot_positions = self._task_cfg['sim']["URRobot"]["position"]
        self._robot_rotations = self._task_cfg['sim']["URRobot"]["quaternion"]
        self._robot_dof_target = torch.tensor(
            self._task_cfg['sim']["URRobot"]["dof_target"],
            device=self._device)
        self._robot_dof_targets = self._robot_dof_target.repeat(
            self._num_envs, 1)

        # table/object info
        self.init_table_position = torch.tensor(
            self._task_cfg['sim']["Table"]["position"],
            device=self._device).repeat(self._num_envs, 1)
        self.object_category = self._task_cfg['sim']["Object"]["category"]
        self._manipulated_object_positions = [
            torch.tensor([-0.6, 0.0, 1.9]),
            torch.tensor([-0.6, -0.25, 1.9])
        ]

        # draw info
        self.debug_draw = _debug_draw.acquire_debug_draw_interface()

        # control parameter
        self._step = 0
        self.frame_skip = 5
        velocity_limit = torch.as_tensor([1.0] * 3 + [3.0] * 3,
                                         device=self.device)  # slow down

        self.velocity_limit = torch.as_tensor(torch.stack(
            [-velocity_limit, velocity_limit], dim=1),
                                              device=self.device)

        self.robot_joints_buffer = []

        return

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

        self.init_mesh()

        self.raytracer = Raycast(self._cfg["raycast_width"],
                                 self._cfg["raycast_height"],
                                 [self.mesh_vertices[0]], [self.mesh_faces[0]])
        self.raytracer.init_setting(self._task_cfg, self._cfg, self.num_envs,
                                    self.debug_draw, self.device)

        if self._task_cfg["sim"]["Control"] == "MotionGeneration":
            self.motion_generation = MotionGeneration(self._robots,
                self._env._world)

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

        self.robot = UR10(
            prim_path=self.default_zero_env_path + "/robot",
            name="robot",
            position=self._robot_positions,
            orientation=self._robot_rotations,
            attach_gripper=False,
            usd_path=self._task_cfg['sim']["URRobot"]['robot_path'])

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

    def update_cache_state(self):

        self.robot_joints = self._robots.get_joint_positions()
        self._wrist2_local_pos, _ = self.wrist_2_link.get_local_poses()
        self._ee_local_pos, _ = self._end_effector.get_local_poses()

    def get_observations(self) -> dict:

        self.update_cache_state()

        current_euler_angles_x = torch.atan2(
            self._ee_local_pos[:, 1] - self._wrist2_local_pos[:, 1],
            self._ee_local_pos[:, 0] - self._wrist2_local_pos[:, 0])

        self.angle_x_dev = torch.atan2(
            self._ee_local_pos[:, 2] - self._wrist2_local_pos[:, 2],
            torch.linalg.norm(self._ee_local_pos[:, :2] -
                              self._wrist2_local_pos[:, :2],
                              dim=1))

        self.angle_z_dev = (current_euler_angles_x -
                            torch.pi / 2) - self.target_angle

        cur_position = self._ee_local_pos.clone()
        cur_position[:, 0] = -cur_position[:, 0]
        self.ee_object_dist = torch.linalg.norm(self.target_ee_position -
                                                cur_position,
                                                dim=1)

        if self._cfg["raycast"]:
            gripper_pose, gripper_rot = self._end_effector.get_world_poses()

            _, _, transformed_vertices = self.transform_mesh()
            self.raycast_reading, self.raytrace_cover_range, self.raytrace_dev = self.raytracer.raytrace_step(
                gripper_pose, gripper_rot, transformed_vertices)

            self.obs_buf = torch.cat([self.robot_joints, self.raycast_reading],
                                     dim=1)
        # if isinstance(self._num_observations, dict):
        #     self.obs_buf = {}
        #     self.obs_buf["state"] = self.robot_joints
        #     self.obs_buf["image"] = self.raycast_reading * 255
        #     return self.obs_buf

        # if self._task_cfg['Training']["use_oracle"]:
        #     self.obs_buf = torch.cat([
        #         current_euler_angles_x[:, None], self.target_angle[:, None],
        #         self.angle_z_dev[:, None], cur_position, self.target_ee_position,
        #         self.target_ee_position - cur_position, self.robot_joints
        #     ],
        #                              dim=1)

        return self.obs_buf

    def get_target_pose(self):

        target_x = 0.3 * torch.sin(torch.as_tensor(self.target_angle)).to(
            self.device) + self.init_ee_dev_local_pos[:, 0]

        target_y = 0.3 * (1 - torch.cos(torch.as_tensor(self.target_angle))
                          ).to(self.device) + self.init_ee_dev_local_pos[:, 1]

        self.target_ee_position = torch.cat([
            -target_x[:, None], target_y[:, None],
            self.init_ee_dev_local_pos[:, 2][:, None]
        ],
                                            dim=1)

        self.init_ee_object_dist = torch.linalg.norm(
            self.target_ee_position[:, :2] - self.init_ee_dev_local_pos[:, :2],
            dim=1)

    def pre_physics_step(self, actions) -> None:

        self.actions = actions

        self._step += 1
        if not self._env._world.is_playing():
            return

        actions = actions.to(self._device)
        actions[:, [2, 3, 4]] = 0
        control_time = self._env._world.get_physics_dt()
        delta_pose = recover_action(actions, self.velocity_limit, control_time)

        if self._task_cfg["sim"]["Control"] == "diffik":

            jacobians = self._robots.get_jacobians(clone=False)
            delta_dof_pos = diffik(jacobian_end_effector=jacobians[:, 6, :, :],
                                   delta_pose=delta_pose)
            delta_dof_pos = torch.clip(delta_dof_pos, -torch.pi, torch.pi)
            current_dof = self._robots.get_joint_positions()
            targets_dof = torch.zeros((self.num_envs, 6)).to(self.device)
            targets_dof = current_dof + delta_dof_pos[:6]

        elif self._task_cfg["sim"]["Control"] == "MotionGeneration":
            cur_ee_pos, cur_ee_orientation = self._end_effector.get_local_poses(
            )
            from omni.isaac.core.utils.types import ArticulationActions, JointsState, XFormPrimViewState

            sim_js = JointsState(
                positions=self._robots.get_joint_positions()[0],
                velocities=self._robots.get_joint_velocities()[0],
                efforts=None)
            sim_js_names = self.robot.dof_names

            target_ee_orientation = quaternion_multiply(
                quaternion_invert(axis_angle_to_quaternion(delta_pose[:, 3:])),
                cur_ee_orientation)
            target_ee_pos = cur_ee_pos + delta_pose[:, :3]
            print(target_ee_pos[0], target_ee_orientation[0])

            targets_dof = self.motion_generation.step_path(
                target_ee_pos, target_ee_orientation)
            targets_dof = targets_dof[None]

        # else:
        #     delta_dof_pos, delta_pose = recover_rule_based_action(
        #         self.num_envs, self.device, self._end_effector,
        #         self.target_ee_position, self.angle_z_dev, self._robots)
        #     current_dof = self._robots.get_joint_positions()
        #     targets_dof = torch.zeros((self.num_envs, 6)).to(self.device)
        #     targets_dof = current_dof + delta_dof_pos[:6]

        targets_dof[:, -1] = 0

        self._robots.set_joint_position_targets(targets_dof)

        pre_position, pre_orientation = self._end_effector.get_local_poses()
        target_position = pre_position + delta_pose[:, :3]

        for i in range(self.frame_skip):
            self._env._world.step(render=False)
        curr_position, curr_orientation = self._end_effector.get_local_poses()
        self.cartesian_error = torch.linalg.norm(curr_position -
                                                 target_position,
                                                 dim=1)
        self.robot_joints_buffer.append([
            self._robots.get_joint_positions()[0].cpu().numpy(),
            curr_position[0].cpu().numpy(), curr_orientation[0].cpu().numpy()
        ])

        # print(curr_position[0],self._robots.get_local_poses())
        np.save("joint.npy", self.robot_joints_buffer)

        # print(
        #     quaternion_multiply(
        #         quaternion_invert(axis_angle_to_quaternion(delta_pose[:, 3:])),
        #         pre_orientation), curr_orientation)

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

        index = torch.where(abs(self.angle_z_dev) < 1.0 / 180 * torch.pi)
        self.angle_z_dev[index] = 0

        dev_percentage = self.angle_z_dev / self.init_angle_z_dev

        # exceed the target
        negative_index = torch.where(dev_percentage < 0)[0]
        if not negative_index.size()[0] == 0:
            dev_percentage[negative_index] = abs(
                dev_percentage[negative_index]) + 1

        dev = torch.clamp(dev_percentage, 0, 1.8)

        angle_reward = abs((1 - dev)**3) * 5

        negative_index = torch.where(dev > 1)[0]

        angle_reward[negative_index] = -abs((1 - dev[negative_index])**3) * 5
        return angle_reward

    def calculate_targetangledev_reward(self) -> None:

        angle_reward = -abs(self.angle_x_dev) * 3

        return angle_reward

    def calculate_raytrace_reward(self) -> None:

        dev_percentage = torch.sum(self.raytrace_cover_range / 0.50, dim=1)

        positive_reward = torch.where(dev_percentage > 1)[0]
        raytrace_range_reward = -(1 - dev_percentage) * 1

        if torch.numel(positive_reward) != 0:
            raytrace_range_reward[positive_reward] = (
                dev_percentage[positive_reward] - 1) * 1

        return raytrace_range_reward

    def calculate_dist_reward(self) -> None:

        dev_percentage = self.ee_object_dist / self.init_ee_object_dist

        # exceed the target
        negative_index = torch.where(dev_percentage < 0)[0]
        if not negative_index.size()[0] == 0:
            dev_percentage[negative_index] = abs(
                dev_percentage[negative_index]) + 1

        dev = torch.clamp(dev_percentage, -1, 1.8)

        dist_reward = abs((1 - dev)**2) * 1

        negative_index = torch.where(dev > 1)[0]

        dist_reward[negative_index] = -abs((1 - dev[negative_index])**2) * 1

        return dist_reward

    def calculate_raytrace_dev_reward(self):
        dev = torch.mean(self.raytrace_dev / 0.04, axis=1)

        dev_reward = torch.clip(1 - dev, -0.5, 1)**3 * 5
        return dev_reward

    def calculate_metrics(self) -> None:

        self.rew_buf = self.calculate_dist_reward()

        self.rew_buf += self.calculate_angledev_reward()
        self.rew_buf += self.calculate_targetangledev_reward()
        self.rew_buf += self.calculate_raytrace_reward()
        self.rew_buf += self.calculate_raytrace_dev_reward()
        self.rew_buf /= 1.2

        controller_penalty = (self.cartesian_error**2) * -1e3
        self.rew_buf += controller_penalty

        action_penalty = torch.sum(
            torch.clip(self._robots.get_joint_velocities(), -1, 1)**2, dim=1
        ) * -1 + torch.sum(torch.clip(self.actions, -1, 1)**2, dim=1) * -0.5

        self.rew_buf += action_penalty

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

        self._robots.set_joint_positions(
            torch.tensor([0, -1.57, 1.57 / 2 * 2, -1.57 * 2, 0, 0],
                         dtype=torch.float).repeat(self.num_envs,
                                                   1).clone().detach())

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
        object_target_position[:, 1] += 0.4
        random_x = torch.rand(self.num_envs).to(self.device) * 0.2
        object_target_position[:, 0] -= random_x
        self._manipulated_object.set_world_poses(object_target_position,
                                                 object_target_quaternion)

        # init table position
        table_position, _ = self._table.get_world_poses()
        table_position[:, 0] = self.init_ee_link_position[:, 0]
        self._table.set_world_poses(table_position)

        for i in range(10):
            self._env._world.step(render=False)

        self.init_ee_dev_local_pos, _ = self._end_effector.get_local_poses()
        self.init_ee_dev_local_pos[:, 0] += random_x

        # reset goal orientation
        self.target_angle = -self.rand_orientation[:, 2].clone()
        self.init_angle_z_dev = -self.target_angle.clone()
        self.get_target_pose()
        self._step = 0
