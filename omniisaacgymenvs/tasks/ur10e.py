from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.objects import DynamicCuboid
from omniisaacgymenvs.robots.controller.ocs2 import Controller_ocs2
from omniisaacgymenvs.robots.controller.osc import Controller_osc

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.rotations import quat_to_euler_angles

import numpy as np
import torch
import math

# from omni.isaac.universal_robots import UR10
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

from skrl.utils import omniverse_isaacgym_utils

from pxr import Usd, UsdGeom

# import warp as wp

class UR10e(RLTask):
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

        self.dt = 1 / 120.0

        self._device = self._cfg["rl_device"]

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._action_scale = self._task_cfg["env"]["actionScale"]
        self._dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self._control_space = self._task_cfg["env"]["controlSpace"]

        # observation and action space
        self._num_observations = 18 # TODO how many?
        if self._control_space == "joint":
            self._num_actions = 6
        elif self._control_space == "cartesian":
            self._num_actions = 3
        else:
            raise ValueError("Invalid control space: {}".format(self._control_space))
        
        self._ur10_positions = torch.tensor([0.0, 0.0, 1.5])
        self._ur10_rotations = torch.tensor([0.0, 0.0, 1.0, 0.0])
        self._ur10_dof_target = torch.tensor([0.06, -2.5, 2.03, 0.58, 1.67, 1.74], device = self._device) 
        self._ur10_dof_targets = self._ur10_dof_target.repeat(self._num_envs, 1) 
        self._target_object_positions = torch.tensor([-0.4, 0.0, 0.9])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

        self._end_effector_link = "ee_link"

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        # self.get_cartpole()
        self.get_ur10()
        self.get_target()
        super().set_up_scene(scene)
        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/ur10", name="ur10_view", reset_xform_properties=False)
        scene.add(self._robots)

        # end-effectors view
        self._hands = RigidPrimView(prim_paths_expr="/World/envs/.*/ur10/ee_link", name="hand_view", reset_xform_properties=False)
        scene.add(self._hands)

        # target view
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        scene.add(self._targets)

        self.init_data()

        return
    
    def init_data(self) -> None:
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)

        if self._control_space == "cartesian":
            self.jacobians = torch.zeros((self._num_envs, 7, 6, 6), device=self._device)
            self.hand_pos, self.hand_rot = torch.zeros((self._num_envs, 3), device=self._device), torch.zeros((self._num_envs, 4), device=self._device)

    def get_ur10(self):
        self.ur10 = UR10(prim_path=self.default_zero_env_path + "/ur10", name="ur10", position=self._ur10_positions, orientation=self._ur10_rotations, attach_gripper=True)
        # applies articulation settings from the task configuration yaml file
        self.ur10.set_joint_positions(self._ur10_dof_target)
        self.ur10.set_joints_default_state(self._ur10_dof_target)
        self._sim_config.apply_articulation_settings("ur10", get_prim_at_path(self.ur10.prim_path), self._sim_config.parse_actor_config("ur10"))

    def get_target(self):
        target = DynamicSphere(prim_path=self.default_zero_env_path + "/target",
                               name="target",
                               radius=0.025,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)

    def get_observations(self) -> dict:
        robot_dof_pos = self._robots.get_joint_positions(clone=False)
        robot_dof_vel = self._robots.get_joint_velocities(clone=False)
        end_effector_pos, end_effector_rot = self._hands.get_world_poses(clone=False)
        target_pos, target_rot = self._targets.get_world_poses(clone=False)

        dof_pos_scaled = 2.0 * (robot_dof_pos - self.robot_dof_lower_limits) \
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits) - 1.0
        dof_vel_scaled = robot_dof_vel * self._dof_vel_scale

        generalization_noise = torch.rand((dof_vel_scaled.shape[0], 6), device=self._device) + 0.5
        self.obs_buf[:, 0] = self.progress_buf / self._max_episode_length
        self.obs_buf[:, 1:7] = dof_pos_scaled[:, :7]
        self.obs_buf[:, 7:13] = dof_vel_scaled[:, :7] * generalization_noise
        self.obs_buf[:, 13:16] = target_pos - self._env_pos

        # compute distance for calculate_metrics() and is_done()
        self._computed_distance = torch.norm(end_effector_pos - target_pos, dim=-1)

        if self._control_space == "cartesian":
            self.jacobians = self._robots.get_jacobians(clone=False)
            self.hand_pos, self.hand_rot = self._hands.get_world_poses(clone=False)
            self.hand_pos -= self._env_pos

        return {self._robots.name: {"obs_buf": self.obs_buf}}

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        env_ids_int32 = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)

        if self._control_space == "joint":
            targets = self.robot_dof_targets[:, :7] + self.robot_dof_speed_scales[:7] * self.dt * self.actions * self._action_scale

        elif self._control_space == "cartesian":
            goal_position = self.hand_pos + actions / 100.0
            delta_dof_pos = omniverse_isaacgym_utils.ik(jacobian_end_effector=self.jacobians[:, 7 - 1, :, :],  # franka hand index: 8
                                                        current_position=self.hand_pos,
                                                        current_orientation=self.hand_rot,
                                                        goal_position=goal_position,
                                                        goal_orientation=None)
            # print("self.robot_dof_targets", self.robot_dof_targets.shape)
            # print("delta_dof_pos", delta_dof_pos.shape)
            # print("self.jacobians", self.jacobians.shape)
            targets = self.robot_dof_targets[:, :7] + delta_dof_pos

        self.robot_dof_targets[:, :7] = torch.clamp(targets, self.robot_dof_lower_limits[:7], self.robot_dof_upper_limits[:7])
        self.robot_dof_targets[:, 7:] = 0
        self._robots.set_joint_position_targets(self.robot_dof_targets, indices=env_ids_int32)

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        # reset robot
        pos = torch.clamp(self._ur10_dof_target.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_robot_dofs), device=self._device) - 0.5),
                          self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        dof_pos = torch.zeros((len(indices), self._robots.num_dof), device=self._device)
        dof_pos[:, :] = pos
        dof_pos[:, 7:] = 0
        dof_vel = torch.zeros((len(indices), self._robots.num_dof), device=self._device)
        self.robot_dof_targets[env_ids, :] = pos
        self.robot_dof_pos[env_ids, :] = pos

        self._robots.set_joint_position_targets(self.robot_dof_targets[env_ids], indices=indices)
        self._robots.set_joint_positions(dof_pos, indices=indices)
        self._robots.set_joint_velocities(dof_vel, indices=indices)

        # reset target
        pos = (torch.rand((len(env_ids), 3), device=self._device) - 0.5) * 2 \
            * torch.tensor([0.25, 0.25, 0.10], device=self._device) \
            + torch.tensor([-0.50, 0.00, 0.50], device=self._device)

        self._targets.set_world_poses(pos + self._env_pos[env_ids], indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self.num_robot_dofs = self._robots.num_dof
        self.robot_dof_pos = torch.zeros((self.num_envs, self.num_robot_dofs), device=self._device)
        dof_limits = self._robots.get_dof_limits()
        self.robot_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.robot_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_targets = torch.zeros((self._num_envs, self.num_robot_dofs), dtype=torch.float, device=self._device)

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = -self._computed_distance

    def is_done(self) -> None:
        self.reset_buf.fill_(0)
        # target reached
        self.reset_buf = torch.where(self._computed_distance <= 0.035, torch.ones_like(self.reset_buf), self.reset_buf)
        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)