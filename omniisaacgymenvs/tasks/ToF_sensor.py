# init extension
import imp
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.debug_draw")
enable_extension("omni.isaac.manipulators")
enable_extension("omni.isaac.motion_generation")

# import env setting
from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omniisaacgymenvs.utils.tools.rotation_conversions import *
from omni.isaac.core.utils.torch.transformations import *
from omni.isaac.core.utils.torch.rotations import *
import omniisaacgymenvs.utils.tools.transform_utils as tf

from .raycast import Raycast
from omniisaacgymenvs.utils.camera_renderer import Renderer

# import util package
import numpy as np
import torch

from cprint import *
import time
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_multiply, Transform3d, quaternion_to_matrix

from omniisaacgymenvs.controller.controller import Controller

from omni.isaac.core.objects import cuboid

import pandas as pd


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
            torch.tensor([-0.6, -0.25, 1.9]),
            torch.tensor([-0.6, -0.5, 1.9]),
        ]

        # control parameter
        self._step = 0
        self.frame_skip = 2
        self.angle_z_dev = torch.zeros((self.num_envs, 1)).to(self._device)
        velocity_limit = torch.as_tensor([1.0] * 3 + [3.0] * 3,
                                         device=self.device)  # slow down

        self.velocity_limit = torch.as_tensor(torch.stack(
            [-velocity_limit, velocity_limit], dim=1),
                                              device=self.device)

        self.robot_joints_buffer = []

        self.target = cuboid.VisualCuboid(
            "/World/envs/env_0/target",
            position=np.array([0.18432, 0.59626, 1.2]),
            orientation=np.array([0, 1, 0, 0]),
            color=np.array([1.0, 0, 0]),
            size=0.05,
        )

        self.episode = 0

        if self._task_cfg["sim"]["Dataset"]:
            data = {
                'Episode': [],
                'Env': [],
                'Step': [],
                'Joints': [],
                'Gripper_pose': [],
                'Hits_Pose': [],
                'Tof_reading': [],
                'Object_hit': [],
                'Object1_position': [],
                'Object2_position': [],
            }
            self.dataset = pd.DataFrame(data)

        rot_mat = quaternion_to_matrix(torch.tensor([self._robot_rotations], dtype=torch.float))
        self.t = Transform3d().rotate(rot_mat).translate(torch.tensor([self._robot_positions], dtype = torch.float))
        self.t_inv = self.t.inverse()
        self.target_pose = torch.tensor([0.18432, 0.59626, 1.2  ]).repeat(self.num_envs, 1).to(self.device)
        
        return

    def set_up_scene(self, scene) -> None:

        from omniisaacgymenvs.utils.robot_loader import ROBOT
        from omniisaacgymenvs.utils.object_loader import Object

        # load robot
        robot = ROBOT(self.num_envs, self.default_zero_env_path,
                      self._robot_positions, self._robot_rotations,
                      self._robot_dof_target, self._sim_config,
                      self._task_cfg['sim']["URRobot"]['robot_path'])

        self.robot = robot.load_UR()
        self.grippers = robot.add_gripper()

        # load object
        object_loader = Object(self._sim_config, self.num_envs, self.device,
                               self.default_zero_env_path)

        if self.object_category in ['cube']:
            self.scale_size = object_loader.load_cube(
                [[0,0,0], self._task_cfg["sim"]["Object"]["scale"]], 2, ['DynamicCylinder', 'DynamicCuboid'])

        # LOAD BIN
        #Bin Base
        object_loader.load_table(
            self._task_cfg['sim']["Table"]["position"],
            self._task_cfg['sim']["Table"]["quaternion"],
            np.array(self._task_cfg['sim']["Table"]["scale"]), "table")
        
        #[0.18432, 0.59626, 0.52]
        #[0.2286, 0.1524, 1.04] #0.8, 0.8, 1.04]  0.2032
        #Bin Back
        object_loader.load_table(
            [0.21505, 0.67514, 0.635],
            self._task_cfg['sim']["Table"]["quaternion"],
            [0.2286, 0.005, 1.27], "back")
        
        #Bin Left
        object_loader.load_table(
            [0.10046, 0.59646, 0.635],
            self._task_cfg['sim']["Table"]["quaternion"],
            [0.005, 0.1524, 1.27], "left")
        
        #Bin Right
        object_loader.load_table(
            [0.33005, 0.59646, 0.635],
            self._task_cfg['sim']["Table"]["quaternion"],
            [0.005, 0.1524, 1.27], "right")
        
        #Bin Top 
        object_loader.load_table(
            [0.21445, 0.59494, 1.27231],
            [0.0, 0.0, -0.707, 0.707],
            [0.2286, 0.005, 0.1524], "top") #0.015
        
        # #Table
        # object_loader.load_table(
        #     self._task_cfg['sim']["Table"]["position"],
        #     self._task_cfg['sim']["Table"]["quaternion"],
        #     np.array(self._task_cfg['sim']["Table"]["scale"]), "table")

        super().set_up_scene(scene)

        self._robots, self._end_effector, self.wrist_2_link= robot.add_scene(
            scene)

        self.manipulated_objects = []
        self._manipulated_object = object_loader.add_scene(
            scene, "/World/envs/.*/manipulated_object_1",
            "manipulated_object_view")
        self.manipulated_objects.append(self._manipulated_object)

        self._manipulated_object_2 = object_loader.add_scene(
            scene, "/World/envs/.*/manipulated_object_2",
            "manipulated_object_view_2")
        self.manipulated_objects.append(self._manipulated_object_2)

        # self._manipulated_object_3 = object_loader.add_scene(
        #     scene, "/World/envs/.*/manipulated_object_3",
        #     "manipulated_object_view_3")
        # self.manipulated_objects.append(self._manipulated_object_3)

        self._base = object_loader.add_scene(scene, "/World/envs/.*/table",
                                              "table_view")
        
        self._back = object_loader.add_scene(scene, "/World/envs/.*/back",
                                              "back_view")
        
        self.back_positions, self.back_orientations = self._back.get_world_poses()

        self._left = object_loader.add_scene(scene, "/World/envs/.*/left",
                                              "left_view")
        
        self.left_positions, self.left_orientations = self._left.get_world_poses()

        self._right = object_loader.add_scene(scene, "/World/envs/.*/right",
                                              "right_view")
        
        self.right_positions, self.right_orientations = self._right.get_world_poses()

        self._top = object_loader.add_scene(scene, "/World/envs/.*/top",
                                              "top_view")
        
        self.top_positions, self.top_orientations= self._top.get_world_poses()

        self.bin = []
        self.bin.append(self._base)
        self.bin.append(self._back)
        self.bin.append(self._left)
        self.bin.append(self._right)
        self.bin.append(self._top)


        # self.old_target_pose = self._manipulated_object_2.get_local_poses()[0]
        self.old_target_pose = self._manipulated_object.get_local_poses()[0]

        if self._cfg["raycast"]:

            self.sensor_radius = torch.as_tensor(
                self._task_cfg['sim']["URRobot"]['sensor_radius']).repeat(
                    self.num_envs, 1).to(self._device)

            self.raytracer = Raycast(
                self._cfg["raycast_width"], self._cfg["raycast_height"], [
                    self._manipulated_object.prim_paths[0],
                    self._manipulated_object_2.prim_paths[0],
                    # self._manipulated_object_3.prim_paths[0]
                    self._base.prim_paths[0],
                    self._back.prim_paths[0],
                    self._left.prim_paths[0],
                    self._right.prim_paths[0],
                    self._top.prim_paths[0]
                ], ['Cylinder', 'Cube', 'Cube', 'Cube','Cube', 'Cube', 'Cube'], self._task_cfg, self._cfg, self.num_envs, self._device,
                self.sensor_radius)
        if self._cfg["depth_renderer"]:

            self.sensor_radius = torch.as_tensor(
                self._task_cfg['sim']["URRobot"]['sensor_radius']).repeat(
                    self.num_envs, 1).to(self._device) * 0.0
            self.depth_renderer = Renderer(
                self._cfg["depth_width"], self._cfg["depth_height"],
                self._manipulated_object.prim_paths[0], self._task_cfg,
                self._cfg, self.num_envs, self._device, self.sensor_radius)

        if not self._task_cfg["sim"]["Design"]:
            self.controller = Controller(
                self.robot,
                self._robots,
                self._env,
                self._end_effector,
                self.velocity_limit,
                self._device,
                self.num_envs,
                control_type=self._task_cfg["sim"]["Control"],
                datagen=self._task_cfg["sim"]["Dataset"])

    def update_cache_state(self):

        self.robot_joints = self._robots.get_joint_positions()
        self._wrist2_local_pos, _ = self.wrist_2_link.get_local_poses()
        self._ee_local_pos, _ = self._end_effector.get_local_poses()
        self._target_pose = self._manipulated_object.get_local_poses()[0]

    def get_observations(self) -> dict:

        self.update_cache_state()
        
        # self.obs_buf = torch.cat([self.robot_joints, self._target_pose],
        #                              dim=1)
        
        # current_euler_angles_x = torch.atan2(
        #     self._ee_local_pos[:, 1] - self._wrist2_local_pos[:, 1],
        #     self._ee_local_pos[:, 0] - self._wrist2_local_pos[:, 0])

        # self.angle_x_dev = torch.atan2(
        #     self._ee_local_pos[:, 2] - self._wrist2_local_pos[:, 2],
        #     torch.linalg.norm(self._ee_local_pos[:, :2] -
        #                       self._wrist2_local_pos[:, :2],
        #                       dim=1))

        # self.angle_z_dev = (current_euler_angles_x -
        #                     torch.pi / 2) - self.target_angle

        cur_position = self._ee_local_pos.clone()
        cur_position[:, 0] = -cur_position[:, 0]
        self.ee_object_dist = torch.linalg.norm(self.target_ee_position -
                                                cur_position,
                                                dim=1)

        if self._cfg["raycast"]:
            gripper_pose, gripper_rot = self._end_effector.get_world_poses()

            cur_object_pose = []
            cur_object_rot = []
            
            for object in self.manipulated_objects + self.bin:
                pose, rot = object.get_world_poses()
                
                cur_object_pose.append(pose)
                cur_object_rot.append(rot)
            
            # 0 - cylinder,1 - box,2 - top ,3 - back,4 - base, 5 -left, 6 right
            #manip1, manip2, base, back, left,right, top
            other_scale = torch.tensor([[0.2286, 0.1524, 1.0668], [0.2286, 0.005, 1.27] , [0.005, 0.1524, 1.27], [0.005, 0.1524, 1.27], [0.2286, 0.005, 0.1524]], device = 'cuda')
            self.scale_sizes = torch.cat((self.scale_size, other_scale))

            self.raycast_reading, self.raytrace_cover_range, self.raytrace_dev , self.debug_ray_hit_points_list, self.object_tracker = self.raytracer.raytrace_step(
                gripper_pose,
                gripper_rot,
                cur_object_pose,
                cur_object_rot,
                self.scale_sizes,
                sensor_radius=self.sensor_radius)

            self.obs_buf = torch.cat([self.robot_joints, self.raycast_reading],
                                     dim=1)


        # if isinstance(self._num_observations, dict):
        #     self.obs_buf = {}
        #     self.obs_buf["state"] = self.robot_joints
        #     self.obs_buf["image"] = self.raycast_reading * 255
        #     return self.obs_buf

        if self._task_cfg['Training']["use_oracle"]:
            
            self.obs_buf = torch.cat([
                # current_euler_angles_x[:, None], # 1
                # self.target_angle[:, None], # 1
                # self.angle_z_dev[:, None], # 1
                # cur_position, # 3
                # self.target_ee_position, # 3
                # self.target_ee_position - cur_position, # 3
                # self.raycast_reading, # 128
                # self._end_effector.get_local_poses()[0], # 3
                self.robot_joints # 6
            ], dim=1)

        if self._cfg["depth_renderer"]:
            gripper_pose, gripper_rot = self._end_effector.get_world_poses()

            cur_object_pose, cur_object_rot = self._manipulated_object.get_world_poses(
            )
            self.depth_renderer.raytrace_step(gripper_pose,
                                              gripper_rot,
                                              cur_object_pose,
                                              cur_object_rot,
                                              self.scale_size,
                                              sensor_radius=self.sensor_radius)

        return self.obs_buf

    def get_target_pose(self):

        target_x = 0.2 * torch.sin(torch.as_tensor(self.target_angle)).to(
            self.device) + self.init_ee_dev_local_pos[:, 0]

        target_y = 0.2 * (1 - torch.cos(torch.as_tensor(self.target_angle))
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

        if self._task_cfg["sim"]["Design"] and self._cfg["raycast"]:
            # if self._step == 1:
            #     self.sensor_radius = self.raytracer.update_params(actions[:,
            #                                                               6:])
            #     target_ee_pos, _ = self._end_effector.get_local_poses()
            # elif self._step >= 1:
            #     target_ee_pos = self.controller.forward(actions[:, :6])
            
            # self._robots.set_joint_position_targets(actions[:, :6] + self._robots.get_joint_positions())
            pass

        elif self._task_cfg["sim"]["Design"]:
            # self._robots.set_joint_position_targets(actions[:, :6] + self._robots.get_joint_positions())
            pass

        else:

            # from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion

            # # rot = torch.stack([torch.tensor([0.707,0.,0.,0.707]), torch.tensor([0.707,0.,0.,0.707])])
            # rot = torch.tensor([0.707,0.,0.,0.707]).repeat(self.num_envs, 1).to(self.device)
            # actions = torch.zeros_like(actions)
            # target_ee_pos = self.controller.forward(actions,
            #                                         self._manipulated_object.get_local_poses()[0], #self.target_ee_position,
            #                                         rot,
            #                                         self.angle_z_dev)
            # Follow target that follows manipulated object 2
            pose_test, rot_test = self._manipulated_object.get_local_poses()
            # pose_test, rot_test = self.target.get_local_pose()
            pose = torch.tensor(pose_test, device='cuda:0')
            # pose = torch.stack([pose, pose], dim=0)
            rot = torch.tensor(rot_test, device='cuda:0') 

            rot_mat = quaternion_to_matrix(torch.tensor([self._robot_rotations], dtype=torch.float))
            t = Transform3d().rotate(rot_mat).translate(torch.tensor([self._robot_positions], dtype = torch.float))
            t_inv = t.inverse()

            if torch.linalg.norm(self.old_target_pose[0] - pose[0][0], dim=0) > 0.001:
                self.target.set_local_pose(pose[0].cpu(), rot[0].cpu())
                pose = t_inv.to('cuda:0').transform_points(pose)

            rot = torch.tensor([0.707,0.,0.,0.707]).repeat(self.num_envs, 1).to(self.device)

            # Don't follow target for first few steps; let is stabilize
            if self._step < 15:
                pose = torch.zeros_like(pose) 

            target_ee_pos = self.controller.forward(actions[:, :6],
                                                    pose, #self.target_ee_position,.
                                                    rot,
                                                    self.angle_z_dev)
            
            self.old_target_pose = pose

        curr_position, _ = self._end_effector.get_local_poses()
        # self.cartesian_error = torch.linalg.norm(curr_position - target_ee_pos,
        #                                          dim=1)

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

        # dist_reward = self._end_effector.get_local_poses()[0] - self._manipulated_object.get_local_poses()[0]
        
        return dist_reward

    def calculate_raytrace_dev_reward(self):
        dev = torch.mean(self.raytrace_dev / 0.04, axis=1)

        dev_reward = torch.clip(1 - dev, -0.5, 1)**3 * 5
        return dev_reward

    def calculate_metrics(self) -> None:

        # Distance to target reward
        # self.target = torch.tensor([[ 12.2123,  -9.4055,   1.2737], [ 12.2123,  -5.4055,   1.2737], [ 12.2123,  -1.4055,   1.2737], [ 12.2123,   2.5945,   1.2737], [ 12.2123,   6.5945,   1.2737],
        #                        [ 12.2123,  10.5945,   1.2737], [  8.2123,  -9.4055,   1.2737], [  8.2123,  -5.4055,   1.2737], [  8.2123,  -1.4055,   1.2737],
        #                        [  8.2123,   2.5945,   1.2737], [  8.2123,   6.5945,   1.2737], [  8.2123,  10.5945,   1.2737], [  4.2123,  -9.4055,   1.2737],
        #                        [  4.2123,  -5.4055,   1.2737], [  4.2123,  -1.4055,   1.2737], [  4.2123,   2.5945,   1.2737], [  4.2123,   6.5945,   1.2737],
        #                        [  4.2123,  10.5945,   1.2737], [  0.2123,  -9.4055,   1.2737], [  0.2123,  -5.4055,   1.2737], [  0.2123,  -1.4055,   1.2737],
        #                        [  0.2123,   2.5945,   1.2737], [  0.2123,   6.5945,   1.2737], [  0.2123,  10.5945,   1.2737], [ -3.7877,  -9.4055,   1.2737],
        #                        [ -3.7877,  -5.4055,   1.2737], [ -3.7877,  -1.4055,   1.2737], [ -3.7877,   2.5945,   1.2737], [ -3.7877,   6.5945,   1.2737],
        #                        [ -3.7877,  10.5945,   1.2737], [ -7.7877,  -9.4055,   1.2737], [ -7.7877,  -5.4055,   1.2737], [ -7.7877,  -1.4055,   1.2737],
        #                        [ -7.7877,   2.5945,   1.2737], [ -7.7877,   6.5945,   1.2737], [ -7.7877,  10.5945,   1.2737], [-11.7877,  -9.4055,   1.2737],
        #                        [-11.7877,  -5.4055,   1.2737], [-11.7877,  -1.4055,   1.2737], [-11.7877,   2.5945,   1.2737]]).to(self.device)
        # target = torch.tensor([0.18432, 0.59626, 1.2  ]).repeat(self.num_envs, 1).to(self.device)
        robot_pose = self.t.to('cuda:0').transform_points(self._end_effector.get_local_poses()[0])
        # dist = torch.cdist(robot_pose, self.target_pose, p=2)
        dist = torch.norm(robot_pose - self.target_pose, dim=1)

        # self.rew_buf = -torch.where(rew <= 0.1, torch.tensor(-100.0), rew * 100)
        self.rew_buf = -10 * (dist - 0.5)

        # Angle of gripper reward
        
        end_effector_target = torch.tensor([ 0.6995, 0.1027, 0.1306, 0.6951]).repeat(self.num_envs, 1).to(self.device)
        angle = self._end_effector.get_local_poses()[1]
        # angle_dev = torch.abs(torch.mean(angle - end_effector_target, dim = 1))
        angle_dev = torch.norm(angle - end_effector_target, dim=1)
        self.rew_buf -= angle_dev * 5

        # 2*acos(abs(parts(p*conj(q))))
        # quaternion_multiply(end_effector_target, torch.conj(angle[1]))

        # self.rew_buf = self.calculate_dist_reward()

        # self.rew_buf += self.calculate_angledev_reward()
        # self.rew_buf += self.calculate_targetangledev_reward()
        # self.rew_buf += self.calculate_raytrace_reward()
        # self.rew_buf += self.calculate_raytrace_dev_reward()
        # self.rew_buf /= 1.2

        # controller_penalty = (self.cartesian_error**2) * -1e3
        # self.rew_buf += controller_penalty

        action_penalty = torch.sum(
            torch.clip(self._robots.get_joint_velocities(), -1, 1)**2, dim=1
        ) * -1 + torch.sum(torch.clip(self.actions, -1, 1)**2, dim=1) * -0.5

        self.rew_buf += action_penalty / 10

        if self._step % 100 == 0:
            print("Reward: ", dist)
            # print("Angle: ", angle_dev)
            # print("Action Penalty: ", action_penalty)
            print("Total Reward: ", self.rew_buf)

        return self.rew_buf

    def is_done(self) -> None:

        # return torch.full((self.num_envs,), 0, dtype=torch.int)

        if (self._step + 1) % 11 == 0: # Was 201 Episode length or horizon *1001*
            if self._task_cfg["sim"]["Dataset"]:
                self.dataset.to_pickle('dataset.pkl')
            self.episode += 1
            self._step = 0
            self.post_reset()
            return [True for i in range(self.num_envs)]

        return [False for i in range(self.num_envs)]

    def reset(self):

        self._robots.set_joint_positions(
            torch.tensor([1.3648, -0.8152, -1.8983, -0.4315, -1.3999,  1.5710],
            # torch.tensor([0.4173, -0.9501, -1.9673, -0.2257, -0.4174,  1.5721],
            # torch.tensor([1.57, -1.57, 1.57 / 2 * 2, -1.57 * 2, -1.57, 0],
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

        self.rand_orientation[:, 2] = rand_ori_z * torch.pi / 2 / 0.7 * 0.5 * (
            torch.randint(0, 2, (self.num_envs, )) * 2 - 1).to(self._device)
        object_target_quaternion = tf.axis_angle_to_quaternion(
            self.rand_orientation)

        # init base position
        base_position, _ = self._base.get_world_poses()
        base_position[:, 0] = self.init_ee_link_position[:, 0]
        self._base.set_world_poses(base_position)

        # #init back position
        # self._back.set_world_poses(self.back_positions, self.back_orientations)

        # # init left position
        # self._left.set_world_poses(self.left_positions, self.left_orientations)

        # # init right position
        # self._right.set_world_poses(self.right_positions, self.right_orientations)

        # # init top position
        # self._top.set_world_poses(self.top_positions, self.top_orientations)
        
        # init table position
        # table_position, _ = self._table.get_world_poses()
        # table_position[:, 0] = self.init_ee_link_position[:, 0]
        # self._table.set_world_poses(table_position)

        # init bin_bottom position
        # bin_bottom_position, _ = self._bin_bottom.get_world_poses()
        # bin_bottom_position[:, 0] = self.init_ee_link_position[:, 0]
        # bin_bottom_position[:, 2] = 1
        # self._bin_bottom.set_world_poses(bin_bottom_position)

        # # init bin_top position
        # bin_top_position, _ = self._bin_top.get_world_poses()
        # bin_top_position[:, 0] = self.init_ee_link_position[:, 0]
        # bin_top_position[:, 2] = 1
        # self._bin_top.set_world_poses(bin_top_position)

        # init position
        object_target_position = target_obj_position.clone()
        object_target_position[:, 1] += 0.4 
        random_x = torch.rand(self.num_envs).to(self.device) * 0.0
        object_target_position[:, 0] -= random_x
        object_target_position[:, 1] += 0.2
        object_target_position[:, 0] += 0.05
        self._manipulated_object.set_world_poses(object_target_position,
                                                 object_target_quaternion)

        object_target_position[:, 1] -= 0.2
        object_target_position[:, 0] -= 0.05
        # self._manipulated_object_2.set_world_poses(object_target_position,
        #                                            object_target_quaternion)

        if self._task_cfg["sim"]["Dataset"]:
            # real life bin bounds for env 1
            bound1 = torch.tensor([-1.8 - 0.12, 0.55, 1.0668], device='cuda:0') # middle of table: -1.8
            bound2 = torch.tensor([-1.8 + 0.12, 0.55 + 0.13 , 1.0668], device='cuda:0') #0.1524
 
            # real life bin bounds for env 0
            bound3 = torch.tensor([2.2 - 0.10, 0.55, 1.0668], device='cuda:0') # middle of table: 2.2 #0.1143
            bound4 = torch.tensor([2.2 + 0.10, 0.55 + 0.13, 1.0668], device='cuda:0')


            object_target_position[0] = (bound4 - bound3) * torch.rand(3, device='cuda:0') + bound3
            object_target_position[1] = (bound2 - bound1) * torch.rand(3, device='cuda:0') + bound1

            # self._manipulated_object_2.set_local_poses(object_target_position,
            #                                             object_target_quaternion)
            
            object_target_position[0] = (bound4 - bound3) * torch.rand(3, device='cuda:0') + bound3
            object_target_position[1] = (bound2 - bound1) * torch.rand(3, device='cuda:0') + bound1
            
            self._manipulated_object.set_world_poses(object_target_position,
                                                 object_target_quaternion)


        for i in range(2): 
            self._env._world.step(render=False)

        self.init_ee_dev_local_pos, _ = self._end_effector.get_local_poses()
        self.init_ee_dev_local_pos[:, 0] += random_x

        # reset goal orientation
        self.target_angle = -self.rand_orientation[:, 2].clone()
        self.init_angle_z_dev = -self.target_angle.clone()
        self.get_target_pose()
        self._step = 0

