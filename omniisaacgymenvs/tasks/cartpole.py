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
from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omni.isaac.core.objects import DynamicSphere
from omniisaacgymenvs.robots.controller.ocs2 import Controller_ocs2
from omniisaacgymenvs.robots.controller.osc import Controller_osc

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
import math

# from omni.isaac.universal_robots import UR10
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from skrl.utils import omniverse_isaacgym_utils


class CartpoleTask(RLTask):
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

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])
        self._ur10_positions = torch.tensor([0.0, 0.0, 2.1])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        self._device = self._cfg["rl_device"]

        self._end_effector_link = "ee_link"

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_cartpole()
        self.get_ur10()
        self.get_target()
        super().set_up_scene(scene)
        self._cartpoles = ArticulationView(prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False)
        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/ur10", name="ur10_view", reset_xform_properties=False)
        scene.add(self._cartpoles)
        scene.add(self._robots)

        # end-effectors view
        self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/ur10/ee_link", name="hand_view", reset_xform_properties=False)
        scene.add(self._hands)

        # target view
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        scene.add(self._targets)

        self.controller = Controller_osc(self._robots, self._device)

        
        # # retrieve file path for the Cartpole USD file
        # assets_root_path = get_assets_root_path()
        # usd_path = assets_root_path + "/Isaac/Robots/UR10/ur10.usd"
        # # add the Cartpole USD to our stage
        # self._cartpole_position = [0.0, 0.0, 2.0]
        # create_prim(prim_path="/World/ur10", prim_type="Xform", position=self._cartpole_position)
        # add_reference_to_stage(usd_path, "/World/ur10")
        # super().set_up_scene(scene) # pass scene to parent class - this method in RLTask also uses GridCloner to clone the robot and adds a ground plane if desired
        # self._my_robots = ArticulationView(prim_paths_expr="/World/ur10", name="cartpole_view") # create a view of robots
        # scene.add(self._my_robots) # add view to scene for initialization
        self.init_data()

        return
    
    def init_data(self) -> None:
        # self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        self.jacobians = torch.zeros((self._num_envs, 10, 6, 9), device=self._device)
        self.hand_pos, self.hand_rot = torch.zeros((self._num_envs, 3), device=self._device), torch.zeros((self._num_envs, 4), device=self._device)

    def get_cartpole(self):
        cartpole = Cartpole(prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole"))

    def get_ur10(self):
        ur10 = UR10(prim_path=self.default_zero_env_path + "/ur10", name="ur10", position=self._ur10_positions, attach_gripper=True)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("ur10", get_prim_at_path(ur10.prim_path), self._sim_config.parse_actor_config("ur10"))

    def get_target(self):
        target = DynamicSphere(prim_path=self.default_zero_env_path + "/target",
                               name="target",
                               radius=0.025,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)

    def get_observations(self) -> dict:
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)

        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        self.obs_buf[:, 0] = cart_pos
        self.obs_buf[:, 1] = cart_vel
        self.obs_buf[:, 2] = pole_pos
        self.obs_buf[:, 3] = pole_vel

        ## PT
        robots_dof_pos = self._robots.get_joint_positions(clone=False)
        robots_dof_vel = self._robots.get_joint_velocities(clone=False)
        ## Pt
        observations = {
            self._cartpoles.name: {
                "obs_buf": self.obs_buf
            },
        }

        # PT
        print("################################ observation")
        # robot_dof_pos = self._robots.get_joint_positions(clone=False)
        # robot_dof_vel = self._robots.get_joint_velocities(clone=False)
        # end_effector_pos, end_effector_rot = self._end_effectors.get_world_poses(clone=False)
        # target_pos, target_rot = self._targets.get_world_poses(clone=False)

        # dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) \
        #     / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        # dof_vel_scaled = robot_dof_vel * self._dof_vel_scale

        # generalization_noise = torch.rand((dof_vel_scaled.shape[0], 7), device=self._device) + 0.5

        # self.obs_buf[:, 0] = self.progress_buf / self._max_episode_length
        # self.obs_buf[:, 1:8] = dof_pos_scaled[:, :7]
        # self.obs_buf[:, 8:15] = dof_vel_scaled[:, :7] * generalization_noise
        # self.obs_buf[:, 15:18] = target_pos - self._env_pos

        # # compute distance for calculate_metrics() and is_done()
        # self._computed_distance = torch.norm(end_effector_pos - target_pos, dim=-1)

        # if self._control_space == "cartesian":
        self.jacobians = self._robots.get_jacobians(clone=False)[:,6:,:,:]
            # self.hand_pos, self.hand_rot = self._hands.get_world_poses(clone=False)
            # self.hand_pos -= self._env_pos
        print("################################ got observation")
        # PT
        return {self._robots.name: {"obs_buf": self.obs_buf}}


        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)
        
        # PT
        self.jacobians = self._robots.get_jacobians(clone=False)
        print("~~~~~~~~~~~~~~~~~~~~~~~self.jacobians  ", self.jacobians.shape)
        # self.actions = actions.clone().to(self._device)
        env_ids_int32 = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        goal_position = self.hand_pos * 0.001 #+ actions / 100.0
        print("@@@@@@@@@@@@@@@@@@@@@@@@22", self.hand_pos)
        delta_dof_pos = omniverse_isaacgym_utils.ik(jacobian_end_effector=self.jacobians[:, 7 - 1, :, :7],  # ur10 hand index: 7?
                                                        current_position=self.hand_pos,
                                                        current_orientation=self.hand_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=None)
        
        print("~~~~~~~~~~~~~~~~~~~~~~~self.robot_dof_targets  ", self.robot_dof_targets.shape)
        print("~~~~~~~~~~~~~~~~~~~~~~~delta_dof_pos  ", delta_dof_pos.shape)
        targets = self.robot_dof_targets[:, :7] + delta_dof_pos

        self.robot_dof_targets[:, :7] = torch.clamp(targets, self.robot_dof_lower_limits[:7], self.robot_dof_upper_limits[:7])
        self.robot_dof_targets[:, 7:] = 0
        self._robots.set_joint_position_targets(self.robot_dof_targets, indices=env_ids_int32)
        # PT

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        # PT
        # TODO -> set starting position in ur10.py
        self._robots.set_joint_positions(torch.tensor(self._num_envs * [[0.06, -2.5, 2.03, 0.58, 1.67, 1.74]]))

        print('**********************************,', self._cartpoles.get_body_index('cart'))
        print('0000000000000000000000000000000000, ', self._cartpoles.get_velocities(clone=False)) #torch.tensor([self._cartpoles.get_body_index('cart')])
        print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@, ', self._cartpoles.get_joint_velocities(clone=False))
        # PT

    def post_reset(self):

        # PT
        self.num_robot_dofs = self._robots.num_dof
        self.robot_dof_pos = torch.zeros((self.num_envs, self.num_robot_dofs), device=self._device)
        dof_limits = self._robots.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)

        # # randomize all envs
        # indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        # self.reset_idx(indices)
        # PT

        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)


    def calculate_metrics(self) -> None:
        cart_pos = self.obs_buf[:, 0]
        cart_vel = self.obs_buf[:, 1]
        pole_angle = self.obs_buf[:, 2]
        pole_vel = self.obs_buf[:, 3]

        reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        cart_pos = self.obs_buf[:, 0]
        pole_pos = self.obs_buf[:, 2]

        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets
