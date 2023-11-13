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
from omni.isaac.core.utils.prims import get_prim_at_path
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
# from pxr import Sd
from .raycast import Raycast, geom_to_trimesh, warp_from_trimesh

import carb

import gym

# import warp as wp
import omni.isaac.core.utils.nucleus as nucleus_utils

ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.core.simulation_context import SimulationContext

from omniisaacgymenvs.utils.tools.rotation_conversions import *
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.stage import get_current_stage

from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *

from cprint import *

DEBUG = True
DEBUG_WITH_TRIMESH = False
SENSORS = 3

class TofSensorTask(RLTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._device = self._cfg["rl_device"]

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = 3
        self._num_actions = 1

        RLTask.__init__(self, name, env)

        self._robot_positions = self._task_cfg['sim']["URRobot"]["position"]
        self._robot_rotations = self._task_cfg['sim']["URRobot"]["quaternion"]
        self._robot_dof_target = torch.tensor(
            self._task_cfg['sim']["URRobot"]["dof_target"],
            device=self._device)
        self._robot_dof_targets = self._robot_dof_target.repeat(
            self._num_envs, 1)
        # self._manipulated_object_positions = torch.tensor([-0.4, 0.0, 0.9])
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
        self.angle_dev = None
        self.current_euler_angles = torch.zeros((self.num_envs,3))

        # control parameter

        self.frame_skip = 1

        self.object_prim_path = []

        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.pre_action = torch.zeros((self.num_envs, 6), device=self.device)
        velocity_limit = torch.as_tensor([1.0] * 3 + [1.0] * 3,
                                         device=self.device)  #slow down

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

    def set_up_scene(self, scene) -> None:

        self.load_robot()
        # self.load_target()
        #self.load_manipulated_object()

        self.get_target_object()
        
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

        # target object
        self._manipulated_object = RigidPrimView(
            prim_paths_expr="/World/envs/.*/manipulated_object_1",
            name="manipulated_object_view",
            reset_xform_properties=False)
        scene.add(self._manipulated_object)

        # Raytracing
        self.raytracer = Raycast()
        self.init_data()

        return

    def compute_ik(self, target_position, target_orientation):

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=
            "/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/cfg/robot/robot_descriptor.yaml",
            urdf_path=
            "/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e/ur16e.urdf"
        )
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

    def load_robot(self):

        from omniisaacgymenvs.robots.articulations.ur10 import UR10
        self.robot = UR10(prim_path=self.default_zero_env_path + "/robot",
                          name="robot",
                          position=self._robot_positions,
                          orientation=self._robot_rotations,
                          attach_gripper=True,
                          usd_path=self.current_directory +
                          self._task_cfg['sim']["URRobot"]['robot_path'])

        self.robot.set_joint_positions(self._robot_dof_target)
        self.robot.set_joints_default_state(self._robot_dof_target)

        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(self.robot.prim_path),
            self._sim_config.parse_actor_config("robot"))

    def load_target(self):

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
        table_translation = np.array([0.25, 1.0, 1.])
        table_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        
        table = FixedCuboid(
            prim_path=self.default_zero_env_path + "/table",
            name="table",
            translation=table_translation,
            orientation=table_orientation,
            scale=np.array([
                0.6, 
                0.6, 
                0.5,
            ]),
            size=1.0,
            color=np.array([1, 0, 0]),
        )
    
    def get_target_object(self):
        target_object_1 = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/manipulated_object_1",
            name="manipulated_object_1",
            position=[0,0,2.02],
            size=.2,
            color=torch.tensor([0, 0, 1]))
        # target_object_2 = DynamicCylinder(prim_path=self.default_zero_env_path + "/target_object_2",
        #                        name="target_object",
        #                        position=self._target_object_positions[1],
        #                        radius=.1,
        #                        height=.5,
        #                        color=torch.tensor([1, 0, 1]))
        self._sim_config.apply_articulation_settings(
            "manipulated_object__1", get_prim_at_path(target_object_1.prim_path),
            self._sim_config.parse_actor_config("manipulated_object_1"))
        


    def load_manipulated_object(self):

        object_dir = self.current_directory + "/omniisaacgymenvs/assests/objects/shapenet_nomat/" + self._task_cfg[
            'sim']["Object"]["category"]
        object_list = os.listdir(object_dir)

        for i in range(self.num_envs):
            object_name = object_list[i]  #np.random.choice(object_list)

            object_path = object_dir + "/" + object_name + "/model_normalized_nomat.usd"
            self.load_object(usd_path=object_path, env_index=i, object_index=1)
            self.object_prim_path.append(object_path)

            # object_path = object_dir + "/" + np.random.choice(object_list) + "/model_normalized_nomat.usd"
            # self.load_object(usd_path=object_path,env_index=i,object_index=2)

    def get_observations(self) -> dict:

        self._robots.num_bodies

        _, current_orientation = self._end_effector.get_world_poses()

        current_euler_angles = quaternion_to_axis_angle(current_orientation)
        quaternion = quaternion_multiply(
            quaternion_invert(self.init_ee_link_orientation),
            current_orientation)
        self.current_euler_angles = quaternion_to_axis_angle(quaternion)

        robot_joint = self._robots.get_joint_positions()

        self.angle_dev = self.current_euler_angles[:, 1] - self.target_angle
    
        self.obs_buf = torch.cat([
             current_euler_angles[:, 1][:, None],
            self.target_angle[:, None], self.angle_dev[:, None]
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

    def _compute_osc_torques(self, dpose):
        self.update_cache_state()
        # Solve for Operational Space Control # Paper: khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf
        # Helpful resource: studywolf.wordpress.com/2013/09/17/robot-control-4-operation-space-control/
        q, qd = self._q[:, :6], self._qd[:, :6]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # Transform our cartesian action `dpose` into joint torques `u`
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
            self.kp * dpose[:, :6] - self.kd * self._ee_vel).unsqueeze(-1)

        # Nullspace control torques `u_null` prevents large changes in joint configuration
        # They are added into the nullspace of OSC so that the end effector orientation remains constant
        # roboticsproceedings.org/rss07/p31.pdf
        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
            (self.ur16e_default_dof_pos[:6] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null[:, self.num_ur16e_dofs:] *= 0
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye((6), device=self.device).unsqueeze(0) -
              torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # Clip the values to be within valid effort range
        u = torch.clamp(u.squeeze(-1),
                        -self._ur16e_effort_limits[:6].unsqueeze(0),
                        self._ur16e_effort_limits[:6].unsqueeze(0))

        return u

    def recover_action(self, action, limit):

        self.control_time = self._env._world.get_physics_dt() * self.frame_skip

        # delta pose
        action = torch.clip(action, -1, 1)
        self.pre_action[:, 5] = action.reshape(-1)

        # action[:,[0,1,2,3,4]] = 0 # rotate along z axis to rotation

        delta_pose = (self.pre_action + 1) / 2 * (limit[:, 1] -
                                                  limit[:, 0]) + limit[:, 0]

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
        self.target_object_pose, self.target_object_rot = self._manipulated_object.get_world_poses()

        # PT Multiple sensors TODO need to move this to utils file
        normals = find_plane_normal(self.num_envs, gripper_rot)
        circle = circle_points(self._task_cfg['sim']["URRobot"]['sensor_radius'], 
                            gripper_pose, 
                            normals, 
                            self._task_cfg['sim']["URRobot"]['num_sensors'])

        for i, env in zip(torch.arange(self._task_cfg['sim']["URRobot"]['num_sensors']).repeat(self.num_envs), torch.arange(self.num_envs).repeat_interleave(self._task_cfg['sim']["URRobot"]['num_sensors'])):
            trimesh_1 = geom_to_trimesh(
                UsdGeom.Cube(
                    get_prim_at_path(self._manipulated_object.prim_paths[env])),
                    self.target_object_pose[env].cpu(), transformations.euler_from_quaternion(self.target_object_rot[env].cpu())) #TODO Why to CPU?
            warp_mesh = warp_from_trimesh(trimesh_1, self._device)
            self.raytracer.set_geom(warp_mesh)
            ray_t, ray_dir = self.raytracer.render(
                int(np.random.normal(10, 10)), circle[env][i].cpu(),
                gripper_rot.cpu()[env])

            if self._cfg["debug_with_trimesh"]: #TODO Update with sensor circle
                ## Visualization for trimesh
                # Create axis for visualization
                axis_origins = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
                axis_directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
                # stack axis rays into line segments for visualization as Path3D
                axis_visualize = trimesh.load_path(np.hstack(( axis_origins,
                    axis_origins + axis_directions)).reshape(-1, 2, 3), colors=np.array([[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]]))
                # stack rays into line segments for visualization as Path3D
                ray_origins = np.repeat(np.expand_dims(gripper_pose.cpu()[0], axis=0),repeats=256, axis=0)
                ray_visualize = trimesh.load_path(np.hstack((ray_origins,
                    ray_origins + ray_dir.numpy())).reshape(-1, 2, 3))
                # trimesh_1.apply_transform(matrix)
                self.j = self.j + 1
                if self.j % 25 == 0:
                    scene = trimesh.Scene([
                        trimesh_1,
                        ray_visualize,
                        axis_visualize])

                    # display the scene
                    scene.show()

            if self._cfg["debug"]:
                sensor_ray_pos_np = circle[env][i].cpu().numpy()
                sensor_ray_pos_tuple = (sensor_ray_pos_np[0],
                                        sensor_ray_pos_np[1],
                                        sensor_ray_pos_np[2])

                ray_dir = ray_dir.numpy()
                ray_t = ray_t.numpy()
                line_vec = np.transpose(
                    np.multiply(np.transpose(ray_dir), ray_t))

                ray_t_nonzero = ray_t[np.nonzero(ray_t)]
                average_distance = np.average(ray_t_nonzero)
                standard_deviation = math.sqrt(
                            max(average_distance * 100 * 0.4795 - 3.2018,
                                0)) 
                noise_distance = np.random.normal(average_distance * 1000,
                                                        standard_deviation)
                # print(f'distance with noise sensor {i}: , {noise_distance}')
                
                # Get rid of ray misses (0 values)
                line_vec = line_vec[np.any(line_vec, axis=1)]
                ray_hit_points_list = line_vec + np.array(sensor_ray_pos_tuple)
                hits_len = len(ray_hit_points_list)
                sensor_ray_pos_list = [
                            sensor_ray_pos_tuple for _ in range(hits_len)
                        ]
                ray_colors = [(1, i, 0, 1) for _ in range(hits_len)]
                ray_sizes = [2 for _ in range(hits_len)]
                point_sizes = [7 for _ in range(hits_len)]
                start_point_colors = [(0, 0.75, 0, 1) for _ in range(hits_len)
                                        ]  # start (camera) points: green
                end_point_colors = [
                    (1, i, 1, 1) for _ in range(hits_len)
                ]  # end (ray hit) points: sensor 1 purple, sensor 2 white
                self.debug_draw.draw_lines(sensor_ray_pos_list,
                                            ray_hit_points_list, ray_colors,
                                            ray_sizes)

                self.debug_draw.draw_points(ray_hit_points_list,
                                            end_point_colors, point_sizes)
                self.debug_draw.draw_points(sensor_ray_pos_list,
                                            start_point_colors, point_sizes)
                # Debug draw the gripper pose
                self.debug_draw.draw_points([circle[env][i].cpu().numpy()],
                                            [(1, 0, 0, 1)], [10])
        
    def pre_physics_step(self, actions) -> None:

        dof_limits = self._robots.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :,
                                                 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :,
                                                 1].to(device=self._device)

        self._step += 1
        if not self._env._world.is_playing():
            return

        actions = actions.to(self._device)
        delta_dof_pos, delta_pose = self.recover_action(
            actions, self.velocity_limit)

        # current dof and current joint velocity
        current_dof = self._robots.get_joint_positions()
        targets_dof = current_dof + delta_dof_pos[:, :6] * self.control_time*2

        targets_dof = torch.clamp(targets_dof, self.robot_dof_lower_limits,
                                  self.robot_dof_upper_limits)

        targets_dof[:, -2] = torch.clamp(targets_dof[:, -2], 0, torch.pi/2)
        # set target dof and target velocity
        self._robots.set_joint_position_targets(targets_dof)
        #self._robots.set_joint_position_targets(self.target_joint_positions)
        # self._robots.set_joint_velocity_targets(delta_dof_pos)

        pre_position, pre_orientation = self._end_effector.get_world_poses(
            clone=False)
        target_position = pre_position + delta_pose[:, :3]

        # frame skip
        for i in range(10):
            self._env._world.step(render=False)

        current_position, current_orientation = self._end_effector.get_world_poses(
            clone=False)

        self.cartesian_error = torch.linalg.norm(target_position -
                                                 current_position,
                                                 dim=1)

        self.raytrace_step()

        #print(self._manipulated_object.get_world_poses()[0][1])

    def post_reset(self):

        self.robot.initialize()
        self.robot.disable_gravity()
        self.reset()
        self.reset_raytracer()

        # reset goal orientation

        self.target_angle = torch.zeros(self.num_envs, device=self.device)

        self.target_angle[:int(self.num_envs / 2)] = torch.as_tensor(
            np.random.uniform(low=0.1, high=0.5, size=int(self.num_envs / 2)) *
            np.pi,
            device=self.device) 
        self.target_angle[int(self.num_envs / 2):] = torch.as_tensor(
            np.random.uniform(low=-0.5, high=-0.1, size=int(
                self.num_envs / 2)) * np.pi,
            device=self.device)
        self.init_angle_dev = -self.target_angle.clone()
        self.stop_index = None
        # print("curr:",self.current_euler_angles[:,1])
        # print("dev:",self.angle_dev,"rew:",self.rew_buf)
    
        # print("rew:",self.rew_buf)
        # print('=====================')
        # print("target angle:",-self.target_angle)
        # frame skip

        # self.reset_internal()


    def calculate_metrics(self) -> None:

        dev_percentage = self.angle_dev / self.init_angle_dev

        index_list = []

        # exceed the target
        negative_index = torch.where(dev_percentage < 0)[0]
        if not negative_index.size()[0] == 0:
            dev_percentage[negative_index] = abs(
                dev_percentage[negative_index]) + 1
            self.stop_index = negative_index

        action_penalty = torch.sum(torch.clamp(
            self._robots.get_joint_velocities() - 1, 1),
                                   dim=1) * -0.0
       
        self.rew_buf = (1 - torch.clamp(dev_percentage, 0 , 1.2)) * 5
      
        
        
        
        return self.rew_buf

    def is_done(self) -> None:

        # return torch.full((self.num_envs,), 0, dtype=torch.int)
     
        
        if (self._step+1) %200 ==0:
            self._step = 0 
            self.post_reset()
            return [True for i in range(self.num_envs)]
        
            
        return [False for i in range(self.num_envs)]


    def reset_internal(self):

        from omni.isaac.core.utils.prims import is_prim_path_valid, get_prim_at_path, delete_prim, is_prim_no_delete

        self.scene.remove_object("robot_view")

        self.load_robot()
        # self.load_target()
        # self.load_manipulated_object()

        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/robot",
                                        name="robot_view",
                                        reset_xform_properties=False)
        self.scene.add(self._robots)

    def reset(self):

        # self._end_effector.get_world_poses()

        # target_joint_positions, _  = self.compute_ik(target_position=np.array([-0.0, 0.8, 1.3]),target_orientation=np.array([0.707,0.,0,0.707]))
        # target_joint_positions = target_joint_positions.joint_positions.astype(np.float)
        # # initial robot
        target_joint_positions = torch.zeros(6, device=self.device)
        target_joint_positions[0] = 1.57
        target_joint_positions[1] = -1.57
        target_joint_positions[2] = 1.57
        target_joint_positions[3] = 0
        target_joint_positions[4] = 1.57
        self.target_joint_positions = torch.tensor(target_joint_positions,
                                                   dtype=torch.float).repeat(
                                                       self.num_envs, 1)
        # [-1.29522039  3.66315136 -0.13982686  5.29926468 -0.22134689 -0.44278792] #
        # print(target_joint_positions)
        # target_joint_positions[5] = -1.57
        self._robots.set_joint_positions(
            torch.tensor(target_joint_positions,
                         dtype=torch.float).repeat(self.num_envs, 1))

        self.target_position, self.target_orientation = self._end_effector.get_world_poses(
        )  # wxyz

        object_target_position = self.target_position.clone()
        object_target_position[:, 1] += 0.6
        self._manipulated_object.set_world_poses(object_target_position)
        self.default_dof = torch.tensor(target_joint_positions,
                                        dtype=torch.float).repeat(
                                            self.num_envs, 1).clone()

        

        for i in range(10):
            self._env._world.step(render=False)
        _, self.init_ee_link_orientation = self._end_effector.get_world_poses()
        

    def reset_raytracer(self):
        self.target_object_pose, self.target_object_rot = self._manipulated_object.get_world_poses(clone=True)
        trimesh_1 = geom_to_trimesh(
            UsdGeom.Cube(
                get_prim_at_path(self._manipulated_object.prim_paths[0])),
                self.target_object_pose[0].cpu(),
                self.target_object_rot[0].cpu()) #TODO Why to CPU?
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
    while (normals * not_normals).sum(dim=-1).max() > 0.99:  # Ensure they're not too similar
        not_normals = torch.rand(batch_size, 3, device='cuda:0')

    # Compute the basis of the planes
    basis1 = torch.cross(normals, not_normals)
    basis1 /= torch.norm(basis1, dim=-1, keepdim=True)
    basis2 = torch.cross(normals, basis1)
    basis2 /= torch.norm(basis2, dim=-1, keepdim=True)

    # Generate points on the circles
    t = torch.arange(0, 2*torch.pi, step=2*torch.pi/num_points, device='cuda:0')
    circles = centers[:, None, :] + radius * (basis1[:, None, :] * torch.cos(t)[None, :, None] + basis2[:, None, :] * torch.sin(t)[None, :, None])
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

    rotation_matrix[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    rotation_matrix[:, 0, 1] = 2*x*y - 2*z*w
    rotation_matrix[:, 0, 2] = 2*x*z + 2*y*w
    rotation_matrix[:, 1, 0] = 2*x*y + 2*z*w
    rotation_matrix[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    rotation_matrix[:, 1, 2] = 2*y*z - 2*x*w
    rotation_matrix[:, 2, 0] = 2*x*z - 2*y*w
    rotation_matrix[:, 2, 1] = 2*y*z + 2*x*w
    rotation_matrix[:, 2, 2] = 1 - 2*x**2 - 2*y**2

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
    normals = torch.tensor([1.0, 0.0, 0.0], device='cuda:0').expand(num_env,-1)
    normals = normals.view(num_env,3,1)
    rotated_normals = torch.bmm(rotation_matrices, normals)
    return rotated_normals.view(num_env,3)
