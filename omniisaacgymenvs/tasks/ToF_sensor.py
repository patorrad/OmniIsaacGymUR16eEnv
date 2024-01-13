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

from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.debug_draw")
enable_extension("omni.isaac.manipulators")
enable_extension("omni.isaac.motion_generation")

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
# from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omniisaacgymenvs.robots.articulations.ur_robot import UR

from omni.isaac.core.objects import DynamicSphere

from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import DynamicCylinder
from omni.isaac.core.objects import FixedCuboid

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
from .raycast import Raycast, find_plane_normal, circle_points, draw_raytrace
import pdb
from omniisaacgymenvs.controller.ik import recover_action, recover_rule_based_action

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
        self.frame_skip = 1
        velocity_limit = torch.as_tensor([0.5] * 3 + [1.5] * 3,
                                         device=self.device)  # slow down

        self.velocity_limit = torch.as_tensor(torch.stack(
            [-velocity_limit, velocity_limit], dim=1),
                                              device=self.device)

    
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
        if self._task_cfg["Curobo"]:
            self.curo_ik_solver = self.init_curobo()


       

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

        self.dist_dev = torch.linalg.norm(self.target_ee_position - cur_position,
                                          dim=1)

        # start = time.time()
        if self._cfg["raycast"]:
            self.raytrace_step()
      
        if self._task_cfg["Curobo"]:
            self.render_curobo()

        joint_angle = self._robots.get_joint_positions()

        if isinstance(self._num_observations, dict):
            self.obs_buf = {}
            self.obs_buf["state"] = joint_angle
            self.obs_buf["image"] = self.raycast_reading * 255
            return self.obs_buf

        if self._task_cfg['Training']["use_oracle"]:
            self.obs_buf = torch.cat([
                current_euler_angles_x[:, None], self.target_angle[:, None],
                self.angle_dev[:, None], cur_position, self.target_ee_position,
                self.target_ee_position - cur_position, joint_angle
            ],
                                     dim=1)

        elif self._cfg["raycast"]:

            self.obs_buf = torch.cat([joint_angle, self.raycast_reading],
                                     dim=1)

        return self.obs_buf

    def update_cache_state(self):
      
        self._robots.set_gains(kds=torch.zeros((self.num_envs, 6)),
                               kps=torch.zeros((self.num_envs, 6)))
     

    def raytrace_step(self) -> None:

        gripper_pose, gripper_rot = self._end_effector.get_world_poses()

        self.target_object_pose, self.target_object_rot = self._manipulated_object.get_world_poses(
        )

        _, _, transformed_vertices = self.transform_mesh()

        gripper_pose, gripper_rot = self._end_effector.get_world_poses()
        normals = find_plane_normal(self.num_envs, gripper_rot)
        self.raycast_circle = circle_points(
            self._task_cfg['sim']["URRobot"]['sensor_radius'], gripper_pose,
            normals, self._task_cfg['sim']["URRobot"]['num_sensors'])

        # PT Multiple sensors TODO need to move this to utils file

        # for draw point
        debug_sensor_ray_pos_list = []
        debug_ray_hit_points_list = []
        debug_ray_colors = []
        debug_ray_sizes = []
        debug_point_sizes = []
        debug_end_point_colors = []
        debug_start_point_colors = []
        debug_circle = []

        self.raycast_reading = torch.zeros(
            (self._num_envs,
             self._cfg["raycast_width"] * self._cfg["raycast_height"] *
             self._task_cfg['sim']["URRobot"]['num_sensors'])).to(
                 self.device) - 1

        num_pixel = self._cfg["raycast_width"] * self._cfg["raycast_height"]
        # ray average distance
        self.raytrace_dist = torch.zeros((self.num_envs, 2)).to(self.device)
        # ray tracing reading
        self.raytrace_reading = torch.zeros(
            (self.num_envs,
             self._cfg["raycast_width"] * self._cfg["raycast_height"],
             2)).to(self.device)
        # ray trace coverage
        self.raytrace_cover_range = torch.zeros(
            (self.num_envs, 2)).to(self.device)
        # ray trace max min dist
        self.raytrace_dev = torch.zeros((self.num_envs, 2)).to(self.device)

        for i, env in zip(
                torch.arange(
                    self._task_cfg['sim']["URRobot"]['num_sensors']).repeat(
                        self.num_envs),
                torch.arange(self.num_envs).repeat_interleave(
                    self._task_cfg['sim']["URRobot"]['num_sensors'])):

            self.raytracer.set_geom(wp.from_torch(transformed_vertices[env]),
                                    mesh_index=0)
            ray_t, ray_dir, normal = self.raytracer.render(
                self.raycast_circle[env][i], gripper_rot[env])

            ray_t = wp.torch.to_torch(ray_t)

            if len(torch.where(ray_t > 0)[0]) > 0:

                # normalize tof reading
                reading = ray_t[torch.where(ray_t > 0)]

                noise_distance = torch.rand(len(torch.where(ray_t > 0)[0]),
                                            device=self.rl_device) / 1000 * 0
                reading += noise_distance
                reading = (reading - torch.min(reading)) / (
                    torch.max(reading) - torch.min(reading) + 1e-5)

                self.raycast_reading[env][i * num_pixel +
                                          torch.where(ray_t > 0)[0]] = reading

                average_distance = torch.mean(ray_t[torch.where(ray_t > 0)])
                cover_percentage = len(torch.where(ray_t > 0)[0]) / 64
            else:
                reading = ray_t
                average_distance = -0.01
                cover_percentage = 0

            self.raytrace_dist[env][i] = average_distance
            self.raytrace_cover_range[env][i] = cover_percentage
            self.raytrace_reading[env, :, i] = ray_t

            # replace the zero value

            ray_t_copy = ray_t.clone()
            if len(torch.where(ray_t <= 0)[0]) > 0:
                index = torch.where(ray_t <= 0)[0]
                ray_t[index] = torch.max(ray_t)

            if torch.max(ray_t) < 1e-2:
                self.raytrace_dev[env][i] = 10
            else:
                self.raytrace_dev[env][i] = torch.max(ray_t) - torch.min(ray_t)

            if self._cfg["debug"]:

                sensor_ray_pos_np = self.raycast_circle[env][i].cpu().numpy()
                sensor_ray_pos_tuple = (sensor_ray_pos_np[0],
                                        sensor_ray_pos_np[1],
                                        sensor_ray_pos_np[2])

                ray_t = ray_t_copy.cpu().numpy()
                ray_dir = ray_dir.numpy()

                line_vec = np.transpose(
                    np.multiply(np.transpose(ray_dir), ray_t))

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

                    debug_circle.append(
                        [self.raycast_circle[env][i].cpu().numpy()])

        if self._cfg["debug"]:

            if len(debug_sensor_ray_pos_list) > 0:

                draw_raytrace(self.debug_draw, debug_sensor_ray_pos_list,
                              debug_ray_hit_points_list, debug_ray_colors,
                              debug_ray_sizes, debug_end_point_colors,
                              debug_point_sizes, debug_start_point_colors,
                              debug_circle)

    def get_target_pose(self):

        target_x = 0.3 * torch.sin(torch.as_tensor(self.target_angle)).to(
            self.device) + self.init_ee_dev_local_pos[:, 0]

        target_y = 0.3 * (1 - torch.cos(torch.as_tensor(
            self.target_angle))).to(self.device) + self.init_ee_dev_local_pos[:, 1]

        self.target_ee_position = torch.cat([
            -target_x[:, None], target_y[:, None],
            self.init_ee_dev_local_pos[:, 2][:, None]
        ],
                                         dim=1)

        self.init_ee_object_dist = torch.linalg.norm(self.target_ee_position[:, :2] -
                                           self.init_ee_dev_local_pos[:, :2],
                                           dim=1)

    

    def pre_physics_step(self, actions) -> None:

        self.actions = actions


        self._step += 1
        if not self._env._world.is_playing():
            return

        if not self._task_cfg["sim"]["Control"]["rule-base"]:

            actions = actions.to(self._device)
            
            delta_dof_pos, delta_pose = recover_action(actions,
                                                       self.velocity_limit,
                                                       self._env, self._robots)
        else:
            delta_dof_pos, delta_pose = recover_rule_based_action(
                self.num_envs, self.device, self._end_effector,
                self.target_ee_position, self.angle_dev, self._robots)

        # current dof and current joint velocity
        current_dof = self._robots.get_joint_positions()
        targets_dof = current_dof + delta_dof_pos[:, :6]

        targets_dof[:, -1] = 0

        self._robots.set_joint_position_targets(targets_dof)

        pre_position, pre_orientation = self._end_effector.get_local_poses()
        target_position = pre_position + delta_pose[:, :3]

        for i in range(1):
            self._env._world.step(render=False)
        curr_position, _ = self._end_effector.get_local_poses()
        self.cartesian_error = torch.linalg.norm(curr_position -
                                                 target_position,
                                                 dim=1)

   

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

        index = torch.where(abs(self.angle_dev) < 1.5 / 180 * torch.pi)
        self.angle_dev[index] = 0

        dev_percentage = self.angle_dev / self.init_angle_dev

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

        angle_reward = -abs(self.angle_dev_target) * 3

        return angle_reward

    def calculate_raytrace_reward(self) -> None:

        dev_percentage = torch.sum(self.raytrace_cover_range / 0.4, dim=1)

        positive_reward = torch.where(dev_percentage > 1)[0]
        raytrace_range_reward = -(1 - dev_percentage) * 1

        if torch.numel(positive_reward) != 0:
            raytrace_range_reward[positive_reward] = (
                dev_percentage[positive_reward] - 1) * 1

        return raytrace_range_reward

    def calculate_dist_reward(self) -> None:

        dev_percentage = self.dist_dev / self.init_ee_object_dist

        # exceed the target
        negative_index = torch.where(dev_percentage < 0)[0]
        if not negative_index.size()[0] == 0:
            dev_percentage[negative_index] = abs(
                dev_percentage[negative_index]) + 1

        dev = torch.clamp(dev_percentage, 0, 1.8)

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
        object_target_position[:, 1] += 0.3
        random_x = torch.rand(self.num_envs).to(self.device) * 0.1
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
        self.init_angle_dev = -self.target_angle.clone()
        self.get_target_pose()
        self._step = 0
