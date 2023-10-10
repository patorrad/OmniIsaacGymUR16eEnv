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

import numpy as np
import torch
import math
import trimesh

# from omni.isaac.universal_robots import UR10
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.debug_draw import _debug_draw

from skrl.utils import omniverse_isaacgym_utils

from pxr import Usd, UsdGeom
from .raycast import Raycast, geom_to_trimesh, warp_from_trimesh
import pdb

import omni.kit.commands
from pxr import Sdf
import omni.usd
# import warp as wp
import omni.isaac.core.utils.nucleus as nucleus_utils
ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
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
        self._robot_dof_target = torch.tensor(self._task_cfg['sim']["URRobot"]["dof_target"], device = self._device) 
        self._robot_dof_targets = self._robot_dof_target.repeat(self._num_envs, 1) 
        # self._target_object_positions = torch.tensor([-0.4, 0.0, 0.9])
        self._target_object_positions = [torch.tensor([-0.6, 0.0, 1.9]), torch.tensor([-0.6, -0.25, 1.9])]
        self.debug_draw = _debug_draw.acquire_debug_draw_interface()

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        self._end_effector_link = "ee_link"

        self._step = 0

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        # self.get_cartpole()
        self.get_ur10()
        self.get_target()
        self.get_target_object()
        super().set_up_scene(scene)
       
        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/ur16", name="ur16_view", reset_xform_properties=False)

        scene.add(self._robots)
      
       
        

        # end-effectors view
        self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/ur16/ee_link", name="hand_view", reset_xform_properties=False)
        scene.add(self._hands)

        # target view
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        scene.add(self._targets)

        # target object
        # self._target_objects = [RigidPrimView(prim_paths_expr="/World/envs/.*/target_object_1", name="target__object_view_1", reset_xform_properties=False),
        #                             RigidPrimView(prim_paths_expr="/World/envs/.*/target_object_2", name="target__object_view_2", reset_xform_properties=False)]

        
        self._target_objects = [RigidPrimView(prim_paths_expr="/World/envs/.*/target_object_1", name="target__object_view_1", reset_xform_properties=False),
                                    RigidPrimView(prim_paths_expr="/World/envs/.*/target_object_2", name="target__object_view_2", reset_xform_properties=False)]
        
        scene.add(self._target_objects[0])
        scene.add(self._target_objects[1])

        # Raytracing
        self.raytracer = Raycast()

    
        self.init_data()
       
       

        return
    
    def init_data(self) -> None:
        # self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        self.jacobians = torch.zeros((self._num_envs, 10, 7, 9), device=self._device)
        self.hand_pos, self.hand_rot = torch.zeros((self._num_envs, 3), device=self._device), torch.zeros((self._num_envs, 4), device=self._device)

    def get_cartpole(self):
        cartpole = Cartpole(prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole"))

    def get_ur10(self):

        # result, prim  = omni.kit.commands.execute('CreateReference',
        #     path_to=Sdf.Path(self.default_zero_env_path + "/ur16"),
        #     asset_path='/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e/ur16e.usd',
        #     usd_context=omni.usd.get_context())
        self.robot = UR(prim_path=self.default_zero_env_path + "/ur16"
                    ,usd_path="/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e/ur16e.usd", 
                    name="robot", position=self._robot_positions, orientation=self._robot_rotations, attach_gripper=False)
       
        self.robot.set_joint_positions(self._robot_dof_target)
        self.robot.set_joints_default_state(self._robot_dof_target)
       
        self._sim_config.apply_articulation_settings("robot", get_prim_at_path(self.robot.prim_path), self._sim_config.parse_actor_config("robot"))
       
      
        

    def get_target(self):

      
        target = DynamicSphere(prim_path=self.default_zero_env_path + "/target",
                               name="target",
                               radius=0.025,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)

    def get_target_object(self):

        import omni.isaac.core.utils.prims as prim_utils
        ycb_usd_paths = {
        "crackerBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        "sugarBox": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        "tomatoSoupCan": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        "mustardBottle": f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
    }   
        
        target_object_name = np.random.choice(["crackerBox","sugarBox","tomatoSoupCan","mustardBottle"])
        translation = torch.rand(3).tolist()
        target_object1 = prim_utils.create_prim(self.default_zero_env_path + "/target_object_1", usd_path=ycb_usd_paths[target_object_name], translation=translation)

        target_object_name = np.random.choice(["crackerBox","sugarBox","tomatoSoupCan","mustardBottle"])
        translation = torch.rand(3).tolist()
        target_object2 = prim_utils.create_prim(self.default_zero_env_path + "/target_object_2", usd_path=ycb_usd_paths[target_object_name], translation=translation)
        

    def get_observations(self) -> dict:
       
        # dof_pos = self._cartpoles.get_joint_positions(clone=False)
        # dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        # cart_pos = dof_pos[:, self._cart_dof_idx]
        # cart_vel = dof_vel[:, self._cart_dof_idx]
        # pole_pos = dof_pos[:, self._pole_dof_idx]
        # pole_vel = dof_vel[:, self._pole_dof_idx]

        # self.obs_buf[:, 0] = cart_pos
        # self.obs_buf[:, 1] = cart_vel
        # self.obs_buf[:, 2] = pole_pos
        # self.obs_buf[:, 3] = pole_vel

        # observations = {
        #     self._cartpoles.name: {
        #         "obs_buf": self.obs_buf
        #     },
        # }

        # PT
        # print("################################ observation")
        # robot_dof_pos = self._robots.get_joint_positions(clone=False)
        # robot_dof_vel = self._robots.get_joint_velocities(clone=False)
        # end_effector_pos, end_effector_rot = self._end_effectors.get_world_poses(clone=False)
        target_pos, target_rot = self._targets.get_world_poses(clone=False)

        # dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) \
        #     / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        # dof_vel_scaled = robot_dof_vel * self._dof_vel_scale

        # generalization_noise = torch.rand((dof_vel_scaled.shape[0], 7), device=self._device) + 0.5

        # self.obs_buf[:, 0] = self.progress_buf / self._max_episode_length
        # self.obs_buf[:, 1:8] = dof_pos_scaled[:, :7]
        # self.obs_buf[:, 8:15] = dof_vel_scaled[:, :7] * generalization_noise
        # self.obs_buf = target_pos - self._env_pos
        self.obs_buf = target_pos
        
        # self.obs_buf[:, 1] = self.hand_pos

        # # compute distance for calculate_metrics() and is_done()
        # self._computed_distance = torch.norm(end_effector_pos - target_pos, dim=-1)

        # if self._control_space == "cartesian":
            # self.jacobians = self._robots.get_jacobians(clone=False)[:,6:,:,:]
        self.hand_pos, self.hand_rot = self._hands.get_world_poses(clone=False)
        # print("$$$$$$$$$$$$$$$$$$$$$$ hand_rot", self.hand_rot)
        # print("$$$$$$$$$$$$$$$$$$$$$$ hand_pos", self.hand_pos)
            # self.hand_pos -= self._env_pos
        # print("################################ got observation")

        return {self._robots.name: {"obs_buf": self.obs_buf}}
        # PT

        # return observations

    def pre_physics_step(self, actions) -> None:

        
        self._step += 1
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        # forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        # indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        # self._cartpoles.set_joint_efforts(forces, indices=indices)
        
        # PT
        # 0.599
        # if self._step % 2 == 0:
        #     self.debug_draw.clear_lines()
        #     self.debug_draw.clear_points()
        #     indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        #     # self.hand_pos, self.hand_rot = self._hands.get_world_poses(clone=False)
        #     self._targets.set_world_poses(self.hand_pos, indices=indices)
        #     self.jacobians = self._robots.get_jacobians(clone=False)
        #     # print("~~~~~~~~~~~~~~~~~~~~~~~self.jacobians  ", self.jacobians.shape)
        #     # self.actions = actions.clone().to(self._device)
        #     env_ids_int32 = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        #     # goal_position = self.hand_pos #+ actions / 100.0
        #     goal_position, _ = self._hands.get_world_poses(clone=False)
        #     # if self._step < 200:
        #     #     goal_position[:,0] += 0.001
        #     # else:
        #     #     goal_position[:,0] -= 0.001 
        #     goal_position[:,1] -= 0.001 
        #     delta_dof_pos = omniverse_isaacgym_utils.ik(jacobian_end_effector=self.jacobians[:, 7, :, :],  # ur10 hand index: 7?
        #                                                     current_position=self.hand_pos,
        #                                                     current_orientation=self.hand_rot,
        #                                                     goal_position=goal_position,
        #                                                     goal_orientation=None)
        #                                                     # torch.tensor([[0.0, 0, 1.0, 0],
        #                                                     # [0.0, 0, 1.0, 0],
        #                                                     # [0.0, 0, 1.0, 0],
        #                                                     # [0.0, 0, 1.0, 0]], device=self._device)) #None)

        #     targets = self.robot_dof_targets[:, :7] + delta_dof_pos[:,:7]

        #     self.robot_dof_targets[:, :7] = torch.clamp(targets, self.robot_dof_lower_limits[:7], self.robot_dof_upper_limits[:7])
        #     self.robot_dof_targets[:, 7:] = 0

        #     self._robots.set_joint_position_targets(self.robot_dof_targets, indices=env_ids_int32)

        # # print("##################", self._robots._gripper.get_world_pose())
        # # test = self._robots._gripper.get_world_pose()
        

        #     # Raytracer
        #     target_object_pose, target_object_rot = self._target_objects[0].get_world_poses(clone=False)
        #     # get_relative_transform(self._target_objects)
        #     ## Normalize quaternion into vector
        #     # Step 1: Normalize the quaternion
        #     q_norm = np.linalg.norm(self.hand_rot.cpu()[1])
        #     q_normalized = self.hand_rot.cpu()[1] / q_norm
        #     # Step 2: Extract the vector part
        #     v = q_normalized[1:]
            
        #     # Step 3: Convert to Cartesian coordinates
        #     cartesian_vector = v
            
        #     # Step 4: Normalize the Cartesian vector
        #     cartesian_norm = np.linalg.norm(cartesian_vector)
        #     cartesian_normalized = cartesian_vector / cartesian_norm
        #     # print("euler", quat_to_euler_angles(self.hand_rot.cpu()[1]))
            
        #     # For 2 sensors
        #     for i in range(2):
        #         # Make the sensors in different positions
        #         hand_pos = self.hand_pos.cpu()[1]
        #         if i:
        #             hand_pos[1] += 0.06
        #         else:
        #             hand_pos[1] -= 0.03

        #         cam_pos = hand_pos - target_object_pose.cpu()[1]
            
        #         ray_t, ray_dir = self.raytracer.render(int(np.random.normal(10, 10)), cam_pos, self.hand_rot.cpu()[1])            

        #         sensor_ray_pos_np = self.hand_pos.cpu()[1].numpy()
        #         sensor_ray_pos_tuple = (sensor_ray_pos_np[0], sensor_ray_pos_np[1], sensor_ray_pos_np[2])

        #         ray_dir = np.array(ray_dir)
        #         ray_t = ray_t.numpy()
        #         line_vec = np.transpose(np.multiply(np.transpose(ray_dir), ray_t))

        #         # print("ray_t", ray_t)
        #         ray_t_nonzero = ray_t[np.nonzero(ray_t)]
        #         # print("ray_t nonzero", ray_t)
        #         average_distance = np.average(ray_t_nonzero)
        #         standard_deviation = math.sqrt(max(average_distance * 100 * 0.4795 - 3.2018, 0)) # variance equation was calculated in cm
        #         noise_distance = np.random.normal(average_distance * 1000, standard_deviation)
        #         # print("average distance", average_distance * 1000) # simulation: m, real sensor: mm
        #         print("distance with noise sensor ", i + 1, ":", noise_distance) # apply noise to average distance

        #         # print("ray_dir", ray_dir)
        #         # print(np.array(sensor_ray_pos_tuple))
        #         # print("line_vec original", line_vec)

        #         # Get rid of ray misses (0 values)
        #         line_vec = line_vec[np.any(line_vec, axis=1)]
        #         # print("line_vec nonzero", line_vec)

        #         ray_hit_points_list = line_vec + np.array(sensor_ray_pos_tuple)
        #         # print("ray_hit_points_list", ray_hit_points_list)

        #         hits_len = len(ray_hit_points_list)

        #         sensor_ray_pos_list = [
        #             sensor_ray_pos_tuple for _ in range(hits_len)
        #         ]

        #         # Sensor 1: red, sensor 2: yellow 
        #         ray_colors = [(1, i, 0, 1) for _ in range(hits_len)]
        #         normal_ray_colors = [(1, 0.5, 0, 1) for _ in range(hits_len)]

        #         ray_sizes = [2 for _ in range(hits_len)]
        #         point_sizes = [7 for _ in range(hits_len)]
        #         start_point_colors = [(0, 0.75, 0, 1) for _ in range(hits_len)] # start (camera) points: green
        #         end_point_colors = [(1, i, 1, 1) for _ in range(hits_len)] # end (ray hit) points: sensor 1 purple, sensor 2 white
        #         self.debug_draw.draw_lines(sensor_ray_pos_list, ray_hit_points_list, ray_colors, ray_sizes)

        #         self.debug_draw.draw_points(ray_hit_points_list, end_point_colors, point_sizes)
        #         self.debug_draw.draw_points(sensor_ray_pos_list, start_point_colors, point_sizes)


            # To view distinguished cone edge rays: uncomment these lines, and in raycast.py, change t in the bottom if branch to 1. instead of 0.
            # outside_rays = ray_hit_points_list[np.where(ray_t_nonzero == 1.)]
            # outside_ray_colors = [(0, 1, 0, 1) for _ in range(len(outside_rays))]
            # sensor_ray_pos_list_outside_only = [
            #     sensor_ray_pos_tuple for _ in range(len(outside_rays))
            # ]
            # self.debug_draw.draw_lines(sensor_ray_pos_list_outside_only, outside_rays, outside_ray_colors, [2 for _ in range(len(outside_rays))])

            
            # self.debug_draw.draw_lines([(0, 0, 0)], [(3000, 3000, 3000)], [(1, 0, 0, 1)], [20]) # uncomment for sanity check line
            

        # if self._step > 1500:
        #     self.raytracer.save()
        #     print("DDDDDDDDOOOOOOOOOOONNNNNNNNNNNNNEEEEEEEEEEEEEEEEE")
        # PT

    def reset_idx(self, env_ids):
      
        num_resets = len(env_ids)

        indices = env_ids.to(dtype=torch.int32)

        # reset robot
        dof_vel = torch.zeros((len(indices), self._robots.num_dof), device=self._device)
        self._robots.set_joint_position_targets(self.robot_dof_targets[env_ids], indices=indices)
        self._robots.set_joint_positions(self._robot_dof_target, indices=indices)
        self._robots.set_joint_velocities(dof_vel, indices=indices)

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
        self.hand_pos, self.hand_rot = self._hands.get_world_poses(clone=False)
        self._targets.set_world_poses(self.hand_pos, indices=indices)
        
        # Raytracing
        # print(type(get_prim_at_path(self._target_objects.prim_paths[0])))
        # print(get_prim_at_path(self._target_objects.prim_paths[0]).GetTypeName())
        # print(type(UsdGeom.Cube(get_prim_at_path(self._target_objects.prim_paths[0]))))
        # TODO Move this to raytracer?
        trimesh_1 = geom_to_trimesh(UsdGeom.Cube(get_prim_at_path(self._target_objects[0].prim_paths[1])), self._target_object_positions[0])
        trimesh_2 = geom_to_trimesh(UsdGeom.Cube(get_prim_at_path(self._target_objects[1].prim_paths[1])), self._target_object_positions[0])
        trimeshes = trimesh.util.concatenate(trimesh_2, trimesh_1)
        # print(trimeshes)
        warp_mesh = warp_from_trimesh(trimeshes, self._device)
        self.raytracer.set_geom(warp_mesh)

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
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = self._robot_dof_targets
        #torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)

        # # randomize all envs
        # indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        # self.reset_idx(indices)

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
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


