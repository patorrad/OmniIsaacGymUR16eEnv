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


import math

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
from omni.isaac.core.prims import RigidPrim, RigidPrimView, GeometryPrimView, XFormPrimView

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
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.objects import FixedCuboid
import omni.isaac.core.utils.nucleus as nucleus_utils
import carb
from omni.isaac.core.utils.types import ArticulationAction
from omniisaacgymenvs.tasks.utils.helper import add_extensions, add_robot_to_scene
from omni.isaac.core.utils.stage import get_current_stage
from cprint import *

class CartpoleTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.update_config(sim_config)
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

    def set_up_scene(self, scene) -> None:

        self.get_robot()
        self.add_cuboid()
        self.load_table()
        
        super().set_up_scene(scene)

        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/robot",
                                        name="robot_view",
                                        reset_xform_properties=False)
        scene.add(self._robots)
        
        self._target = XFormPrimView(
            prim_paths_expr="/World/envs/.*/Cuboid/cuboid", name="cuboid_view", reset_xform_properties=False
        )
        scene.add(self._target)

        self.motion_gen = self.init_curobo()
        self._step = 0
        return

    # def initialize_views(self, scene):
    #     super().initialize_views(scene)
    #     if scene.object_exists("cartpole_view"):
    #         scene.remove_object("cartpole_view", registry_only=True)
    #     self._cartpoles = ArticulationView(
    #         prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
    #     )
    #     scene.add(self._cartpoles)

    def get_robot(self):

        from omniisaacgymenvs.robots.articulations.ur10 import UR10

        self.robot = UR10(
            prim_path=self.default_zero_env_path + "/robot",
            name="robot",
            position=np.array([0, 0, 0]),
            orientation=np.array([1, 0, 0, 0]),
            attach_gripper=False,
            usd_path=self._task_cfg['sim']["URRobot"]['robot_path'])
      
        self.robot.set_joint_positions(torch.tensor([0, 0, 0, 0, 0, 0]))
        self.robot.set_joints_default_state(torch.tensor([0, 0, 0, 0, 0, 0]))

        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(self.robot.prim_path),
            self._sim_config.parse_actor_config("robot"))
        
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
        
    def add_cuboid(self):
        visual_cuboid = cuboid.VisualCuboid(
            self.default_zero_env_path + "/Cuboid/cuboid",
            position=np.array([0.5, 0, 0.5]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )
        self._sim_config.apply_articulation_settings(
            "cuboid", get_prim_at_path(visual_cuboid.prim_path), self._sim_config.parse_actor_config("cuboid")
        )

    def get_observations(self) -> dict:
        # dof_pos = self._robots.get_joint_positions(clone=False)
        # dof_vel = self._robots.get_joint_velocities(clone=False)

        # self.cart_pos = dof_pos[:, self._cart_dof_idx]
        # self.cart_vel = dof_vel[:, self._cart_dof_idx]
        # self.pole_pos = dof_pos[:, self._pole_dof_idx]
        # self.pole_vel = dof_vel[:, self._pole_dof_idx]

        # self.obs_buf[:, 0] = self.cart_pos
        # self.obs_buf[:, 1] = self.cart_vel
        # self.obs_buf[:, 2] = self.pole_pos
        # self.obs_buf[:, 3] = self.pole_vel

        observations = {} #{self._robots.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        self._step += 1
        if not self._env._world.is_playing():
            return

        # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(reset_env_ids) > 0:
        #     self.reset_idx(reset_env_ids)

        # actions = actions.to(self._device)

        # forces = torch.zeros((self._robots.count, self._robots.num_dof), dtype=torch.float32, device=self._device)
        # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

        # indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        # self._robots.set_joint_efforts(forces, indices=indices)
        
        ## <curobo
        robot_prim_path = "/World/env/envs_0/robot"
        if not self._env._world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1

        else:
            # if self._step < 2:
            #     self._robots._articulation_view.initialize()
            #     idx_list = [self._robots.get_dof_index(x) for x in self.j_names]
            #     cprint.err(f'idx_list {idx_list}')
            #     self._robots.set_joint_positions(self.default_config, torch.tensor(idx_list))

            #     self._robots._articulation_view.set_max_efforts(
            #         values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
            #     )
            # if self._step == 50 or self._step % 1000 == 0.0:
            #     print("Updating world, reading w.r.t.") #, robot_prim_path)
            #     obstacles = self.usd_help.get_obstacles_from_stage(
            #         # only_paths=[obstacles_path],
            #         reference_prim_path=robot_prim_path,
            #         ignore_substring=[
            #             # "/World/envs/env_0/robot",
            #             # # "/World/envs/env_0/manipulated_object_1",
            #             # # "/World/envs/env_0/table",
            #             "/World/defaultGroundPlane",
            #             "/curobo",
            #         ],
            #     ).get_collision_check_world()

                # self.motion_gen.update_world(obstacles)
                # print("Updated World")
                # carb.log_info("Synced CuRobo world from stage.")
            
            # position and orientation of target virtual cube:
            # target_position, target_rotation = self._manipulated_object.get_world_poses()
            target_position, target_rotation = self._target.get_local_poses()
            ik_goal = Pose(
                position=target_position,
                quaternion=target_rotation,
            )
            if self.prev_goal is None:
                self.prev_goal = ik_goal.clone()
            if self.past_goal is None:
                self.past_goal = ik_goal.clone()
            
            sim_js_names = self._robots.dof_names
            sim_js = self._robots.get_joints_state()
            cu_js = JointState(
                position=self.tensor_args.to_device(sim_js.positions),
                velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
                acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
                jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
                joint_names=sim_js_names,
            )
            # cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
            prev_distance = ik_goal.distance(self.prev_goal)
            past_distance = ik_goal.distance(self.past_goal)

            if False and self._step % 2 == 0:
                sph_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

                if self.spheres is None:
                    self.spheres = []
                    # create spheres:

                    for si, s in enumerate(sph_list[0]):
                        sp = sphere.VisualSphere(
                            prim_path="/curobo/robot_sphere_" + str(si),
                            position=np.ravel(s.position),
                            radius=float(s.radius),
                            color=np.array([0, 0.8, 0.2]),
                        )
                        self.spheres.append(sp)
                else:
                    for si, s in enumerate(sph_list[0]):
                        self.spheres[si].set_world_pose(position=np.ravel(s.position))
                        self.spheres[si].set_radius(float(s.radius))

            if (
                (torch.sum(prev_distance[0] > 1e-2) or torch.sum(prev_distance[1] > 1e-2))
                and (torch.sum(past_distance[0]) == 0.0 and torch.sum(past_distance[1] == 0.0))
                and torch.max(torch.abs(cu_js.velocity)) < 0.2
                and self.cmd_plan[0] is None
                and self.cmd_plan[1] is None
            ):
                cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
                result = self.motion_gen.plan_batch_env(cu_js, ik_goal, self.plan_config.clone())
                
                self.prev_goal.copy_(ik_goal)
                if torch.count_nonzero(result.success) > 0:
                    trajs = result.get_paths()
                    for s in range(len(result.success)):
                        if result.success[s]:
                            self.cmd_plan[s] = self.motion_gen.get_full_js(trajs[s])
                            # cmd_plan = result.get_interpolated_plan()
                            # cmd_plan = motion_gen.get_full_js(cmd_plan)
                            # get only joint names that are in both:
                            self.idx_list = []
                            common_js_names = []
                            for x in sim_js_names:
                                if x in self.cmd_plan[s].joint_names:
                                    self.idx_list.append(self._robots.get_dof_index(x))
                                    common_js_names.append(x)

                            self.cmd_plan[s] = self.cmd_plan[s].get_ordered_joint_state(common_js_names)

                        self.cmd_idx = 0

            for s in range(len(self.cmd_plan)):
                # cprint.ok(f's {s} length {len(self.cmd_plan)}  cmd_plan{self.cmd_plan}')
                if self.cmd_plan[s] is not None and self.cmd_idx < len(self.cmd_plan[s].position):
                    cmd_state = self.cmd_plan[s][self.cmd_idx]
                    
                    # get full dof state
                    # art_action = ArticulationAction(
                    #     cmd_state.position.cpu().numpy(),
                    #     cmd_state.velocity.cpu().numpy(),
                    #     joint_indices=self.idx_list,
                    # )
                    # print(cmd_state.position)
                    # set desired joint angles obtained from IK:
                    env_ids_int32 = torch.tensor([s], dtype=torch.int32, device=self._device)
                    # self._robots.apply_action(art_action, [s])
                    self._robots.set_joint_position_targets(cmd_state.position, indices=env_ids_int32)

                else:
                    self.cmd_plan[s] = None
            self.cmd_idx += 1
            self.past_goal.copy_(ik_goal)

            #     # Set EE teleop goals, use cube for simple non-vr init:
            #     ee_translation_goal = target_position
            #     ee_orientation_teleop_goal = target_rotation

            #     # compute curobo solution:
            #     ik_goal = Pose(
            #         position=self.tensor_args.to_device(ee_translation_goal),
            #         quaternion=self.tensor_args.to_device(ee_orientation_teleop_goal),
            #     )

            #     result = self.motion_gen.plan_batch_env(cu_js, ik_goal, self.plan_config)
            #     trajs = result.get_paths()
            #     for s in range(len(result.success)):
            #         if not result.success[s]:
            #             self.cmd_plan[s] = self.motion_gen.get_full_js(trajs[s])
            #         # get only joint names that are in both:
            #         self.idx_list = []
            #         common_js_names = []
            #         for x in sim_js_names:
            #             if x in self.cmd_plan[s].joint_names:
            #                 self.idx_list.append(self._robots.get_dof_index(x))
            #                 common_js_names.append(x)
            #         # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

            #         self.cmd_plan[s] = self.cmd_plan[s].get_ordered_joint_state(common_js_names)

            #         self.cmd_idx = 0

            #     else:
            #         carb.log_warn("Plan did not converge to a solution.  No action is being taken.")
            #     self.target_pose = target_position
            # self.past_pose = target_position
            # for s in range(len(self.cmd_plan)):
            #     if self.cmd_plan[s] is not None and self.cmd_idx < len(self.cmd_plan[s].position):
            #         cmd_state_tmp = self.cmd_plan[s][self.cmd_idx]

            #         # # get full dof state
            #         # position = [self.cmd_state.position.cpu().numpy(),self.cmd_state.position.cpu().numpy(),self.cmd_state.position.cpu().numpy(),self.cmd_state.position.cpu().numpy()]
            #         # velocity = [self.cmd_state.velocity.cpu().numpy(),self.cmd_state.velocity.cpu().numpy(),self.cmd_state.velocity.cpu().numpy(),self.cmd_state.velocity.cpu().numpy()] 
            #         # art_action = ArticulationAction(
            #         #     position,
            #         #     velocity,
            #         #     joint_indices=self.idx_list,
            #         # )
            #         # # set desired joint angles obtained from IK:
            #         env_ids_int32 = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
            #         # self._robots.apply_action(art_action, env_ids_int32)
            #         self._robots.set_joint_position_targets(cmd_state_tmp.position, indices=env_ids_int32)
            #     else:
            #         self.cmd_plan[s] = None
            # self.cmd_idx += 1
            
            # curobo>

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # # randomize DOF positions
        # dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        # dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # # randomize DOF velocities
        # dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        # dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # # apply resets
        # indices = env_ids.to(dtype=torch.int32)
        # self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        # self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        pass
        # self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        # self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        # # randomize all envs
        # indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        # self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        pass
        # reward = 1.0 - self.pole_pos * self.pole_pos - 0.01 * torch.abs(self.cart_vel) - 0.005 * torch.abs(self.pole_vel)
        # reward = torch.where(torch.abs(self.cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # reward = torch.where(torch.abs(self.pole_pos) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

        # self.rew_buf[:] = reward

    def is_done(self) -> None:
        pass
        # resets = torch.where(torch.abs(self.cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(self.pole_pos) > math.pi / 2, 1, resets)
        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        # self.reset_buf[:] = resets

    def init_curobo(self):

            self.tensor_args = TensorDeviceType()
            robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                            "ur16e.yml"))["robot_cfg"]
            self.j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
            self.default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

            # self._robots, robot_prim_path = add_robot_to_scene(robot_cfg, self._env._world)
    
            # articulation_controller = self._robots.get_articulation_controller()

            # world_cfg_table = WorldConfig.from_dict(
            #     load_yaml(
            #         join_path(get_world_configs_path(), "collision_table.yml")))
            # world_cfg_table.cuboid[0].pose[2] -= 0.04
            # world_cfg1 = WorldConfig.from_dict(
            #     load_yaml(
            #         join_path(get_world_configs_path(),
            #                 "collision_table.yml"))).get_mesh_world()
            # world_cfg1.mesh[0].name += "_mesh"
            # world_cfg1.mesh[0].pose[2] = -10.5

            n_obstacle_cuboids = 30
            n_obstacle_mesh = 10

            # world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid,
            #                         mesh=world_cfg1.mesh)
            # world_cfg_list = []
            # for i in range(self._num_envs):
            #     world_cfg_list.append(WorldConfig("collision_test.yml"))
            world_file = ["collision_test.yml" for i in range(self.num_envs)]
            world_cfg_list = []
            for i in range(self.num_envs):
                world_cfg = WorldConfig.from_dict(
                    load_yaml(join_path(get_world_configs_path(), world_file[i]))
                ) 
                world_cfg.objects[0].pose[2] -= 0.02
                world_cfg.randomize_color(r=[0.2, 0.3], b=[0.0, 0.05], g=[0.2, 0.3])
                # usd_help.add_world_to_stage(world_cfg, base_frame="/World/world_" + str(i))
                world_cfg_list.append(world_cfg)

            motion_gen_config = MotionGenConfig.load_from_robot_config(
                robot_cfg,
                world_cfg_list,
                self.tensor_args,
                trajopt_tsteps=32,
                collision_checker_type=CollisionCheckerType.MESH,
                use_cuda_graph=True,
                num_trajopt_seeds=12,
                num_graph_seeds=12,
                interpolation_dt=0.03,
                collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
                collision_activation_distance=0.025,
                acceleration_scale=1.0,
                self_collision_check=True,
                maximum_trajectory_dt=0.25,
                fixed_iters_trajopt=True,
                finetune_dt_scale=1.05,
                velocity_scale=[1.0, 1., 1., 1., 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            motion_gen = MotionGen(motion_gen_config)
            print("warming up...")
            motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False,
                              batch=self.num_envs, batch_env_mode=True)

            self.past_goal = None
            self.prev_goal = None
            self.target_pose = None
            self.cmd_plan = [None] * self.num_envs
            self.cmd_idx = 0
            self.spheres = None
            
            self.plan_config = MotionGenPlanConfig(
                enable_graph=False, enable_graph_attempt=4, max_attempts=2, enable_finetune_trajopt=True
            )

            self.usd_help = UsdHelper()
            self.usd_help.stage = get_current_stage()
    
            print("Curobo is Ready")

            return motion_gen
    

# # Copyright (c) 2018-2022, NVIDIA Corporation
# # All rights reserved.
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #
# # 1. Redistributions of source code must retain the above copyright notice, this
# #    list of conditions and the following disclaimer.
# #
# # 2. Redistributions in binary form must reproduce the above copyright notice,
# #    this list of conditions and the following disclaimer in the documentation
# #    and/or other materials provided with the distribution.
# #
# # 3. Neither the name of the copyright holder nor the names of its
# #    contributors may be used to endorse or promote products derived from
# #    this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# import math

# import numpy as np
# import torch
# from omni.isaac.core.articulations import ArticulationView
# from omni.isaac.core.utils.prims import get_prim_at_path
# from omniisaacgymenvs.tasks.base.rl_task import RLTask
# from omniisaacgymenvs.robots.articulations.cartpole import Cartpole
# from omni.isaac.core.prims import RigidPrim, RigidPrimView, GeometryPrimView, XFormPrimView

# # CuRobo
# # from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
# from curobo.geom.sdf.world import CollisionCheckerType
# from curobo.geom.types import WorldConfig
# from curobo.types.base import TensorDeviceType
# from curobo.types.math import Pose
# from curobo.types.robot import JointState
# from curobo.types.state import JointState
# from curobo.util.logger import setup_curobo_logger
# from curobo.util.usd_helper import UsdHelper
# from curobo.util_file import (
#     get_assets_path,
#     get_filename,
#     get_path_of_dir,
#     get_robot_configs_path,
#     get_world_configs_path,
#     join_path,
#     load_yaml,
# )
# from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
# from omni.isaac.core.objects import cuboid, sphere
# import carb
# from omni.isaac.core.utils.types import ArticulationAction
# from omniisaacgymenvs.tasks.utils.helper import add_extensions, add_robot_to_scene
# from omni.isaac.core.utils.stage import get_current_stage
# from cprint import *

# class CartpoleTask(RLTask):
#     def __init__(self, name, sim_config, env, offset=None) -> None:

#         self.update_config(sim_config)
#         self._max_episode_length = 500

#         self._num_observations = 4
#         self._num_actions = 1

#         RLTask.__init__(self, name, env)
#         return

#     def update_config(self, sim_config):
#         self._sim_config = sim_config
#         self._cfg = sim_config.config
#         self._task_cfg = sim_config.task_config

#         self._num_envs = self._task_cfg["env"]["numEnvs"]
#         self._env_spacing = self._task_cfg["env"]["envSpacing"]
#         self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])

#         self._reset_dist = self._task_cfg["env"]["resetDist"]
#         self._max_push_effort = self._task_cfg["env"]["maxEffort"]

#     def set_up_scene(self, scene) -> None:

#         self.get_robot()
#         self.add_cuboid()
#         # articulation_controller = robot.get_articulation_controller()
        
#         super().set_up_scene(scene)

#         # self._robots = robot.get_articulation_controller()
#         # cprint.err(f'robot_prim_path {robot_prim_path}')
#         self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/robot",
#                                         name="robot_view",
#                                         reset_xform_properties=False)
#         scene.add(self._robots)
        
#         # # Make a target to follow
#         # self.visual_target = cuboid.VisualCuboid(
#         #     "/World/envs/.*/target",
#         #     position=np.array([0.5, 0, 0.5]),
#         #     orientation=np.array([0, 1, 0, 0]),
#         #     color=np.array([1.0, 0, 0]),
#         #     size=0.05,
#         # )
        
#         self._target = XFormPrimView(
#             prim_paths_expr="/World/envs/.*/Cuboid/cuboid", name="cuboid_view", reset_xform_properties=False
#         )
#         scene.add(self._target)

#         self.motion_gen = self.init_curobo()
#         self._step = 0
#         return

#     # def initialize_views(self, scene):
#     #     super().initialize_views(scene)
#     #     if scene.object_exists("cartpole_view"):
#     #         scene.remove_object("cartpole_view", registry_only=True)
#     #     self._cartpoles = ArticulationView(
#     #         prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
#     #     )
#     #     scene.add(self._cartpoles)

#     def get_robot(self):

#         from omniisaacgymenvs.robots.articulations.ur10 import UR10

#         self.robot = UR10(
#             prim_path=self.default_zero_env_path + "/robot",
#             name="robot",
#             position=np.array([0, 0, 0]),
#             orientation=np.array([1, 0, 0, 0]),
#             attach_gripper=False,
#             usd_path=self._task_cfg['sim']["URRobot"]['robot_path'])
      
#         self.robot.set_joint_positions(torch.tensor([0, 0, 0, 0, 0, 0]))
#         self.robot.set_joints_default_state(torch.tensor([0, 0, 0, 0, 0, 0]))

#         self._sim_config.apply_articulation_settings(
#             "robot", get_prim_at_path(self.robot.prim_path),
#             self._sim_config.parse_actor_config("robot"))
        
#     def add_cuboid(self):
#         visual_cuboid = cuboid.VisualCuboid(
#             self.default_zero_env_path + "/Cuboid/cuboid",
#             position=np.array([0.5, 0, 0.5]),
#             orientation=np.array([0, 1, 0, 0]),
#             color=np.array([1.0, 0, 0]),
#             size=0.05,
#         )
#         self._sim_config.apply_articulation_settings(
#             "cuboid", get_prim_at_path(visual_cuboid.prim_path), self._sim_config.parse_actor_config("cuboid")
#         )

#     def get_observations(self) -> dict:
#         # dof_pos = self._robots.get_joint_positions(clone=False)
#         # dof_vel = self._robots.get_joint_velocities(clone=False)

#         # self.cart_pos = dof_pos[:, self._cart_dof_idx]
#         # self.cart_vel = dof_vel[:, self._cart_dof_idx]
#         # self.pole_pos = dof_pos[:, self._pole_dof_idx]
#         # self.pole_vel = dof_vel[:, self._pole_dof_idx]

#         # self.obs_buf[:, 0] = self.cart_pos
#         # self.obs_buf[:, 1] = self.cart_vel
#         # self.obs_buf[:, 2] = self.pole_pos
#         # self.obs_buf[:, 3] = self.pole_vel

#         observations = {} #{self._robots.name: {"obs_buf": self.obs_buf}}
#         return observations

#     def pre_physics_step(self, actions) -> None:
#         self._step += 1
#         if not self._env._world.is_playing():
#             return

#         # reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
#         # if len(reset_env_ids) > 0:
#         #     self.reset_idx(reset_env_ids)

#         # actions = actions.to(self._device)

#         # forces = torch.zeros((self._robots.count, self._robots.num_dof), dtype=torch.float32, device=self._device)
#         # forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]

#         # indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
#         # self._robots.set_joint_efforts(forces, indices=indices)
        
#         ## <curobo
#         robot_prim_path = "/World/env/envs_0/robot"
#         if not self._env._world.is_playing():
#             if i % 100 == 0:
#                 print("**** Click Play to start simulation *****")
#             i += 1

#         else:
#             # if self._step < 2:
#             #     self._robots._articulation_view.initialize()
#             #     idx_list = [self._robots.get_dof_index(x) for x in self.j_names]
#             #     cprint.err(f'idx_list {idx_list}')
#             #     self._robots.set_joint_positions(self.default_config, torch.tensor(idx_list))

#             #     self._robots._articulation_view.set_max_efforts(
#             #         values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
#             #     )
#             # if self._step == 50 or self._step % 1000 == 0.0:
#             #     print("Updating world, reading w.r.t.") #, robot_prim_path)
#             #     obstacles = self.usd_help.get_obstacles_from_stage(
#             #         # only_paths=[obstacles_path],
#             #         # reference_prim_path=robot_prim_path,
#             #         ignore_substring=[
#             #             # "/World/envs/env_0/robot",
#             #             # # "/World/envs/env_0/manipulated_object_1",
#             #             # # "/World/envs/env_0/table",
#             #             # "/World/defaultGroundPlane",
#             #             "/curobo",
#             #         ],
#             #     ).get_collision_check_world()

#             #     self.motion_gen.update_world(obstacles)
#             #     print("Updated World")
#             #     carb.log_info("Synced CuRobo world from stage.")
            
#             # position and orientation of target virtual cube:
#             # target_position, target_rotation = self._manipulated_object.get_world_poses()
#             target_position, target_rotation = self._target.get_world_poses()
#             target_position = target_position[0]
#             target_rotation = target_rotation[0]
#             if self.past_pose is None:
#                 self.past_pose = target_position
#             if self.target_pose is None:
#                 self.target_pose = target_position
            
#             sim_js = self._robots.get_joints_state()
#             sim_js_names = self._robots.dof_names
#             cu_js = JointState(
#                 position=self.tensor_args.to_device(sim_js.positions[0]),
#                 velocity=self.tensor_args.to_device(sim_js.velocities[0]) * 0.0,
#                 acceleration=self.tensor_args.to_device(sim_js.velocities[0]) * 0.0,
#                 jerk=self.tensor_args.to_device(sim_js.velocities[0]) * 0.0,
#                 joint_names=sim_js_names,
#             )
#             cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

#             if True and self._step % 2 == 0:
#                 sph_list = self.motion_gen.kinematics.get_robot_as_spheres(cu_js.position)

#                 if self.spheres is None:
#                     self.spheres = []
#                     # create spheres:

#                     for si, s in enumerate(sph_list[0]):
#                         sp = sphere.VisualSphere(
#                             prim_path="/curobo/robot_sphere_" + str(si),
#                             position=np.ravel(s.position),
#                             radius=float(s.radius),
#                             color=np.array([0, 0.8, 0.2]),
#                         )
#                         self.spheres.append(sp)
#                 else:
#                     for si, s in enumerate(sph_list[0]):
#                         self.spheres[si].set_world_pose(position=np.ravel(s.position))
#                         self.spheres[si].set_radius(float(s.radius))

#             if (
#                 torch.linalg.norm(target_position - self.target_pose) > 1e-3
#                 and torch.linalg.norm(self.past_pose - target_position) == 0.0
#                 and torch.linalg.norm(sim_js.velocities) < 0.2
#             ):
#                 # Set EE teleop goals, use cube for simple non-vr init:
#                 ee_translation_goal = target_position
#                 ee_orientation_teleop_goal = target_rotation

#                 # compute curobo solution:
#                 ik_goal = Pose(
#                     position=self.tensor_args.to_device(ee_translation_goal),
#                     quaternion=self.tensor_args.to_device(ee_orientation_teleop_goal),
#                 )

#                 result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config)
#                 # ik_result = ik_solver.solve_single(ik_goal, cu_js.position.view(1,-1), cu_js.position.view(1,1,-1))

#                 succ = result.success.item()  # ik_result.success.item()
#                 if succ:
#                     self.cmd_plan = result.get_interpolated_plan()
#                     self.cmd_plan = self.motion_gen.get_full_js(self.cmd_plan)
#                     # get only joint names that are in both:
#                     self.idx_list = []
#                     common_js_names = []
#                     for x in sim_js_names:
#                         if x in self.cmd_plan.joint_names:
#                             self.idx_list.append(self._robots.get_dof_index(x))
#                             common_js_names.append(x)
#                     # idx_list = [robot.get_dof_index(x) for x in sim_js_names]

#                     self.cmd_plan = self.cmd_plan.get_ordered_joint_state(common_js_names)

#                     cmd_idx = 0

#                 else:
#                     carb.log_warn("Plan did not converge to a solution.  No action is being taken.")
#                 self.target_pose = target_position
#             self.past_pose = target_position
#             if self.cmd_plan is not None:
#                 self.cmd_state = self.cmd_plan[self.cmd_idx]

#                 # # get full dof state
#                 # position = [self.cmd_state.position.cpu().numpy(),self.cmd_state.position.cpu().numpy(),self.cmd_state.position.cpu().numpy(),self.cmd_state.position.cpu().numpy()]
#                 # velocity = [self.cmd_state.velocity.cpu().numpy(),self.cmd_state.velocity.cpu().numpy(),self.cmd_state.velocity.cpu().numpy(),self.cmd_state.velocity.cpu().numpy()] 
#                 # art_action = ArticulationAction(
#                 #     position,
#                 #     velocity,
#                 #     joint_indices=self.idx_list,
#                 # )
#                 # # set desired joint angles obtained from IK:
#                 env_ids_int32 = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
#                 # self._robots.apply_action(art_action, env_ids_int32)
#                 self._robots.set_joint_position_targets(self.cmd_state.position, indices=env_ids_int32)
#                 self.cmd_idx += 1
#                 # for _ in range(2): #TODO Need this?
#                 #     self._env._world.step(render=False)
#                 if self.cmd_idx >= len(self.cmd_plan.position):
#                     self.cmd_idx = 0
#                     self.cmd_plan = None
#             # curobo>

#     def reset_idx(self, env_ids):
#         num_resets = len(env_ids)

#         # # randomize DOF positions
#         # dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
#         # dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
#         # dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

#         # # randomize DOF velocities
#         # dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
#         # dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
#         # dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

#         # # apply resets
#         # indices = env_ids.to(dtype=torch.int32)
#         # self._cartpoles.set_joint_positions(dof_pos, indices=indices)
#         # self._cartpoles.set_joint_velocities(dof_vel, indices=indices)

#         # bookkeeping
#         self.reset_buf[env_ids] = 0
#         self.progress_buf[env_ids] = 0

#     def post_reset(self):
#         pass
#         # self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
#         # self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
#         # # randomize all envs
#         # indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
#         # self.reset_idx(indices)

#     def calculate_metrics(self) -> None:
#         pass
#         # reward = 1.0 - self.pole_pos * self.pole_pos - 0.01 * torch.abs(self.cart_vel) - 0.005 * torch.abs(self.pole_vel)
#         # reward = torch.where(torch.abs(self.cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
#         # reward = torch.where(torch.abs(self.pole_pos) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)

#         # self.rew_buf[:] = reward

#     def is_done(self) -> None:
#         pass
#         # resets = torch.where(torch.abs(self.cart_pos) > self._reset_dist, 1, 0)
#         # resets = torch.where(torch.abs(self.pole_pos) > math.pi / 2, 1, resets)
#         # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
#         # self.reset_buf[:] = resets

#     def init_curobo(self):

#             self.tensor_args = TensorDeviceType()
#             robot_cfg = load_yaml(join_path(get_robot_configs_path(),
#                                             "ur16e.yml"))["robot_cfg"]
#             self.j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
#             self.default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

#             # self._robots, robot_prim_path = add_robot_to_scene(robot_cfg, self._env._world)
    
#             # articulation_controller = self._robots.get_articulation_controller()

#             # world_cfg_table = WorldConfig.from_dict(
#             #     load_yaml(
#             #         join_path(get_world_configs_path(), "collision_table.yml")))
#             # world_cfg_table.cuboid[0].pose[2] -= 0.04
#             # world_cfg1 = WorldConfig.from_dict(
#             #     load_yaml(
#             #         join_path(get_world_configs_path(),
#             #                 "collision_table.yml"))).get_mesh_world()
#             # world_cfg1.mesh[0].name += "_mesh"
#             # world_cfg1.mesh[0].pose[2] = -10.5

#             n_obstacle_cuboids = 30
#             n_obstacle_mesh = 10

#             # world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid,
#             #                         mesh=world_cfg1.mesh)
#             world_cfg_list = []
#             for i in range(self._num_envs):
#                 world_cfg_list.append(WorldConfig())

#             motion_gen_config = MotionGenConfig.load_from_robot_config(
#                 robot_cfg,
#                 world_cfg_list,
#                 self.tensor_args,
#                 trajopt_tsteps=32,
#                 collision_checker_type=CollisionCheckerType.MESH,
#                 use_cuda_graph=True,
#                 num_trajopt_seeds=12,
#                 num_graph_seeds=12,
#                 interpolation_dt=0.03,
#                 collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
#                 collision_activation_distance=0.025,
#                 acceleration_scale=1.0,
#                 self_collision_check=True,
#                 maximum_trajectory_dt=0.25,
#                 fixed_iters_trajopt=True,
#                 finetune_dt_scale=1.05,
#                 velocity_scale=[0.25, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0],
#             )
#             motion_gen = MotionGen(motion_gen_config)
#             print("warming up...")
#             motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False,
#                               batch=self.num_envs, batch_env_mode=True)

#             self.past_pose = None
#             self.target_pose = None
#             self.cmd_plan = None
#             self.cmd_idx = 0
#             self.spheres = None
            
#             self.plan_config = MotionGenPlanConfig(
#                 enable_graph=False, enable_graph_attempt=4, max_attempts=2, enable_finetune_trajopt=True
#             )

#             self.usd_help = UsdHelper()
#             self.usd_help.stage = get_current_stage()
    
#             print("Curobo is Ready")

#             return motion_gen
