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

# from omni.isaac.universal_robots import UR10
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
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

from omni.physx import  acquire_physx_interface
from pxr import Sdf

import carb

import gym

# import warp as wp
import omni.isaac.core.utils.nucleus as nucleus_utils
ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from omni.isaac.core.simulation_context import SimulationContext

from omniisaacgymenvs.utils.tools import rotation_conversions
from omni.isaac.core.simulation_context import SimulationContext


class TofSensorTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._device = self._cfg["rl_device"]

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])
        self._robot_positions = self._task_cfg['sim']["URRobot"]["position"]
        self._robot_rotations = self._task_cfg['sim']["URRobot"]["quaternion"]
        self._robot_dof_target = torch.tensor(
            self._task_cfg['sim']["URRobot"]["dof_target"], device=self._device)
        self._robot_dof_targets = self._robot_dof_target.repeat(
            self._num_envs, 1)
        # self._target_object_positions = torch.tensor([-0.4, 0.0, 0.9])
        self._target_object_positions = [torch.tensor(
            [-0.6, 0.0, 1.9]), torch.tensor([-0.6, -0.25, 1.9])]
        self.debug_draw = _debug_draw.acquire_debug_draw_interface()

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        self._end_effector_link = "ee_link"
        self.current_directory = os.getcwd()

        self._step = 0

        # control parameter
        velocity_limit = torch.Tensor([1.0] * 3 + [1]*3)  #slow down

        self.velocity_limit = np.stack([-velocity_limit, velocity_limit],
                                       axis=1)
        RLTask.__init__(self, name, env)
        
        self.frame_skip = 10
    


      

        return

    def set_up_scene(self, scene) -> None:
        # self.get_cartpole()
        self.load_robot()
        # self.load_target()
        self.load_target_object()
        # self.load_pod()
        super().set_up_scene(scene)
       
        self._robots = ArticulationView(
            prim_paths_expr="/World/envs/.*/robot", name="robot_view", reset_xform_properties=False)
        scene.add(self._robots)
        
      

        # end-effectors view
        self._end_effector = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/ee_link", name="hand_view", reset_xform_properties=False)
        scene.add(self._end_effector)
        # pdb.set_trace()
        
        
       

        # target view
        # self._targets = RigidPrimView(
        #     prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        # scene.add(self._targets)

        
        # target object
        self._target_object = RigidPrimView(prim_paths_expr="/World/envs/.*/target_object_1", name="target__object_view_1", reset_xform_properties=False)
        scene.add(self._target_object)
     

        # Raytracing
        self.raytracer = Raycast()

        self.init_data()

        return

    def init_data(self) -> None:
        # self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        self.jacobians = torch.zeros(
            (self._num_envs, 10, 7, 9), device=self._device)
        self.hand_pos, self.hand_rot = torch.zeros(
            (self._num_envs, 3), device=self._device), torch.zeros((self._num_envs, 4), device=self._device)

    def compute_ik(self,target_position,target_orientation):
      
       

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = "/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/cfg/robot/robot_descriptor.yaml",
            urdf_path ="/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e/ur16e.urdf"
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
       

        self._articulation_kinematics_solver = ArticulationKinematicsSolver(_robot,self._kinematics_solver, "ee_link")


        self._kinematics_solver.set_robot_base_pose(robot_position=np.array(self._robot_positions),robot_orientation=np.array(self._robot_rotations))
       
        joint_positions, success = self._articulation_kinematics_solver.compute_inverse_kinematics(target_position,target_orientation)
        return joint_positions,success
        

    def load_robot(self):

    
        from omniisaacgymenvs.robots.articulations.ur10 import UR10
        self.robot = UR10(prim_path=self.default_zero_env_path + "/robot", name="robot", position=self._robot_positions, orientation=self._robot_rotations, 
                          attach_gripper=True, usd_path=self.current_directory + self._task_cfg['sim']["URRobot"]['robot_path'])
        
        self.robot.set_joint_positions(self._robot_dof_target)
        self.robot.set_joints_default_state(self._robot_dof_target)
        

        self._sim_config.apply_articulation_settings("robot", get_prim_at_path(
            self.robot.prim_path), self._sim_config.parse_actor_config("robot"))
       

        

    def load_target(self):

        target = DynamicSphere(prim_path=self.default_zero_env_path + "/target",
                               name="target", radius=0.025, color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(
            target.prim_path), self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)
    

    def load_object(self,usd_path,env_index,object_index,translaton=[-0.69, 0.1, 1.3],orientation=[0,0,0.707,0.707],scale=[0.4,0.4,0.4]):
       

       # ================================= load object ========================================
        prim_utils.create_prim(
        f"/World/envs/env_{env_index}"+f"/target_object_{object_index}", 
        usd_path = usd_path, translation=translaton,orientation=orientation, scale=scale)
        

        stage = omni.usd.get_context().get_stage()
        object_prim = stage.GetPrimAtPath(
            f"/World/envs/env_{env_index}"+f"/target_object_{object_index}")
        
        # ================================= set property ========================================
        # Make it a rigid body
        # utils.setRigidBody(object_prim, "convexHull", True)
        # # mass_api = UsdPhysics.MassAPI.Apply(object_prim)
        # # mass_api.CreateMassAttr(10)
        # # # Alternatively set the density
        # # mass_api.CreateDensityAttr(1000)
        # UsdPhysics.CollisionAPI.Apply(object_prim)

        #self._sim_config.apply_rigid_body_settings("Object", object_prim.GetPrim(),self._sim_config.parse_actor_config("Object"),is_articulation=False)
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
        self._sim_config.apply_rigid_body_settings("Object", object_prim.GetPrim(),self._sim_config.parse_actor_config("Object"),is_articulation=False)

       


        # ================================= add texture ========================================
        # Change the server to your Nucleus install, default is set to localhost in omni.isaac.sim.base.kit
        default_server = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
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
            default_server + "/Isaac/Samples/DR/Materials/Textures/marble_tile.png",
            Sdf.ValueTypeNames.Asset,
        )
    
        # Bind the material to the prim
        cube_mat_shade = UsdShade.Material(mtl_prim)
        UsdShade.MaterialBindingAPI(object_prim).Bind(cube_mat_shade, UsdShade.Tokens.strongerThanDescendants)


    
    def load_pod(self):
        for i in range(self.num_envs):
            
            # add_reference_to_stage(
            #         prim_path=f"/World/envs/env_{i}" + "/pod", 
            #         usd_path = "/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/pod/pod.usd", 
            # )

            # initila toe orientation
            from scipy.spatial.transform import Rotation
            # Define the Euler angles in the format (roll, pitch, yaw)
            euler_angles = [np.pi/2, np.pi/2, np.pi/2]

            # Create a Rotation object from the Euler angles
            r = Rotation.from_euler('xyz', euler_angles, degrees=False)

            # Get the corresponding quaternion
            quaternion = r.as_quat()

            prim_utils.create_prim(
                 prim_path=f"/World/envs/env_{i}" + "/pod", 
                usd_path = self.current_directory + "/omniisaacgymenvs/assests/robots/pod/pod.usd", 
                translation=[-0.463, 0.95, 0], 
                # orientation=quaternion,
               
                )
            stage = omni.usd.get_context().get_stage()
            cube_prim = stage.GetPrimAtPath(
                f"/World/envs/env_{i}" + "/pod")

            # ================================= set property ========================================
            # Make it a rigid body with kinematic
            utils.setRigidBody(cube_prim, "convexMeshSimplification", True)

            mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
            mass_api.CreateMassAttr(10)
            # Alternatively set the density
            mass_api.CreateDensityAttr(1000)
            UsdPhysics.CollisionAPI.Apply(cube_prim)

           

                    
            


    def load_target_object(self):

        object_dir = self.current_directory + "/omniisaacgymenvs/assests/objects/shapenet_nomat/" + self._task_cfg['sim']["Object"]["category"]
        object_list = os.listdir(object_dir)
        

    
        for i in range(self.num_envs):

            object_path = object_dir + "/" + np.random.choice(object_list) + "/model_normalized_nomat.usd"
            self.load_object(usd_path=object_path,env_index=i,object_index=1)

            # object_path = object_dir + "/" + np.random.choice(object_list) + "/model_normalized_nomat.usd"
            # self.load_object(usd_path=object_path,env_index=i,object_index=2)


           

    def get_observations(self) -> dict:
        _, current_orientation = self._end_effector.get_local_poses()

        current_euler_angles = rotation_conversions.quaternion_to_axis_angle(current_orientation)
    
        self.obs_buf = None #target_pos


        return {self._robots.name: {"obs_buf": self.obs_buf}}
     

      


    def update_cache_state(self):
        self._q = self._robots.get_joint_positions()
        self._qd = self._robots.get_joint_velocities()
        self._mm = self._robots.get_mass_matrices()
        self._robots.set_gains(kds=torch.zeros((self.num_envs,6)),kps=torch.zeros((self.num_envs,6)))
        self._j_eef = self._robots.get_jacobians(clone=False)[:, 6, :, :]
        self.kp = torch.tensor(
            [150., 150., 150., 100., 100., 100.], device=self.device)/100
        self.kd = 2 * torch.sqrt(self.kp)
        self.kp_null = torch.tensor([10.]*6, device=self.device)
        self.kd_null = 2 * torch.sqrt(self.kp_null)
        self.num_ur16e_dofs = 6 
        self.ur16e_default_dof_pos = self.default_dof

        self._ee_vel = torch.concat([self._robots.get_linear_velocities(),self._robots.get_angular_velocities()],dim=1)
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
            self.kp * dpose[:, :6] - self.kd * self._ee_vel ).unsqueeze(-1)

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
                         -self._ur16e_effort_limits[:6].unsqueeze(0), self._ur16e_effort_limits[:6].unsqueeze(0))

        return u
    

    def action_space(self):
        return  gym.spaces.Box(np.ones(self._robots.num_dof) * -1.0, np.ones(self._robots.num_dof) * 1.0)

    def recover_action(self, action, limit):
        self.control_time = self._env._world.get_physics_dt()*self.frame_skip
        
        # delta pose
        action = torch.clip(action, -1, 1)
        action[:,[0,1,2,3,4]] = 0 # rotate along z axis to rotation
        
        delta_pose = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]

        self.jacobians = self._robots.get_jacobians(clone=False)
        delta_dof_pos = self.ik(jacobian_end_effector=self.jacobians[:, 6, :, :], 
                                delta_pose=delta_pose)
        delta_dof_pos = torch.clip(delta_dof_pos, -torch.pi, torch.pi)
    
        return delta_dof_pos,delta_pose
    
    

    def ik(self,jacobian_end_effector,
       delta_pose,
       damping_factor=0.05):
            
            """
            Damped Least Squares method: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
            """

            # compute position and orientation error
            delta_pose = delta_pose[:,:,None]

            # solve damped least squares (dO = J.T * V)
            transpose = torch.transpose(jacobian_end_effector, 1, 2)
            lmbda = torch.eye(6).to(jacobian_end_effector.device) * (damping_factor ** 2)
            return (transpose @ torch.inverse(jacobian_end_effector @ transpose + lmbda) @ delta_pose).squeeze(dim=2)
                

    def pre_physics_step(self, actions) -> None:

        self._step += 1
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        actions = actions.to(self._device)
        delta_dof_pos, delta_pose = self.recover_action(actions,self.velocity_limit)
        
        
        # current dof and current joint velocity
        current_dof = self._robots.get_joint_positions()                                            
        targets_dof = current_dof + delta_dof_pos[:,:6]*self.control_time
        
        targets_dof = torch.clamp(targets_dof, self.robot_dof_lower_limits[:7], self.robot_dof_upper_limits[:7])
        self._robots.set_joint_position_targets(targets_dof)
        self._robots.set_joint_velocity_targets(delta_dof_pos)


        pre_position, pre_orientation = self._end_effector.get_world_poses(clone=False)
        target_position = pre_position+delta_pose[:,:3]
        
        # frame skip
        for i in range(self.frame_skip):
            self._env._world.step(render=True)

        current_position, current_orientation = self._end_effector.get_world_poses(clone=False)
        
        cartesian_error = torch.linalg.norm(target_position-current_position,dim=1)
        print(cartesian_error)
        
            
        

           

       
    def reset_idx(self, env_ids):

        num_resets = len(env_ids)

        indices = env_ids.to(dtype=torch.int32)

        # reset robot
        dof_vel = torch.zeros(
            (len(indices), self._robots.num_dof), device=self._device)
        # self._robots.set_joint_position_targets(
        #     self.robot_dof_targets[env_ids], indices=indices)
        # # self._robots.set_joint_positions(
        # #     self._robot_dof_target, indices=indices)
        # self._robots.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        # PT
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', self.robot_dof_targets.shape)
        # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', self.robot_dof_targets)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@', self._robot_dof_targets)
        # reset target
        # pos = (torch.rand((len(env_ids), 3), device=self._device) - 0.5) * 2 \
        #     * torch.tensor([0.25, 0.25, 0.10], device=self._device) \
        #     + torch.tensor([0.50, 0.00, 0.20], device=self._device)

        # self._targets.set_world_poses(pos + self._env_pos[env_ids], indices=indices)
        self.hand_pos, self.hand_rot = self._end_effector.get_world_poses(clone=False)
        # self._targets.set_world_poses(self.hand_pos, indices=indices)
        # self._end_effector.set_world_poses(self.hand_pos)
        
        

        # Raytracing
        # print(type(get_prim_at_path(self._target_object.prim_paths[0])))
        # print(get_prim_at_path(self._target_object.prim_paths[0]).GetTypeName())
        # print(type(UsdGeom.Cube(get_prim_at_path(self._target_object.prim_paths[0]))))
        # TODO Move this to raytracer?
        # trimesh_1 = geom_to_trimesh(UsdGeom.Cube(get_prim_at_path(
        #     self._target_object[0].prim_paths[1])), self._target_object_positions[0])
        # trimesh_2 = geom_to_trimesh(UsdGeom.Cube(get_prim_at_path(
        #     self._target_object[1].prim_paths[1])), self._target_object_positions[0])
        # trimeshes = trimesh.util.concatenate(trimesh_2, trimesh_1)
        # # print(trimeshes)
        # warp_mesh = warp_from_trimesh(trimeshes, self._device)
        # self.raytracer.set_geom(warp_mesh)

        # scene = trimesh.Scene([trimeshes])
        # scene.show()
        # PT

    def post_reset(self):

        # PT
        # self.num_robot_dofs = self._robots.num_dof
        # self.robot_dof_pos = torch.zeros((self.num_envs, self.num_robot_dofs), device=self._device)

        self.robot.initialize()
        self.robot.disable_gravity()
                
        
        dof_limits = self._robots.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(
            device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(
            device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(
            self.robot_dof_lower_limits)
        self.robot_dof_targets = self._robot_dof_targets
        # torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)

        # # randomize all envs
        # indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        # self.reset_idx(indices)

        # randomize all envs
        indices = torch.arange(
            self._robots.count, dtype=torch.int64, device=self._device)
        # self.reset_idx(indices)
        # PT

        # self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        # self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # # randomize all envs
        # indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        # self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # cart_pos = self.obs_buf[:, 0]
        # cart_vel = self.obs_buf[:, 1]
        # pole_angle = self.obs_buf[:, 2]
        # pole_vel = self.obs_buf[:, 3]

        # reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        # self.rew_buf[:] = reward
        pass

    def is_done(self) -> None:
        # cart_pos = self.obs_buf[:, 0]
        # pole_pos = self.obs_buf[:, 2]

        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        # self.reset_buf[:] = resets

        # PT
        self.reset_buf.fill_(0)
        # # target reached
        # self.reset_buf = torch.where(self._computed_distance <= 0.035, torch.ones_like(self.reset_buf), self.reset_buf)
        # # max episode length
        # self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)
        # PT

    def reset(self):
      
        target_joint_positions, _  = self.compute_ik(target_position=np.array([-0.0, 0.8, 1.3]),target_orientation=np.array([0.707,0.,0,0.707]))
        target_joint_positions = target_joint_positions.joint_positions.astype(np.float)
        # initial robot
        target_joint_positions = np.zeros(6)
        target_joint_positions[0] = 1.57
        target_joint_positions[1] = -1.57
        target_joint_positions[2] = 1.57
        target_joint_positions[3] = 0
        target_joint_positions[4] = 1.57
        # [-1.29522039  3.66315136 -0.13982686  5.29926468 -0.22134689 -0.44278792] # 
        # print(target_joint_positions)
        # target_joint_positions[5] = -1.57
        self._robots.set_joint_positions(torch.tensor(target_joint_positions,dtype=torch.float).repeat(self.num_envs,1))


        self.target_position, self.target_orientation = self._end_effector.get_world_poses() # wxyz

        
      

        object_target_position = self.target_position.clone()
        object_target_position[:,1] += 0.6
        self._target_object.set_world_poses(object_target_position)
        self.default_dof = torch.tensor(target_joint_positions,dtype=torch.float).repeat(self.num_envs,1).clone()

        
      
