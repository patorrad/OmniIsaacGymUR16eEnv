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
from omniisaacgymenvs.utils.plot_utils import plot_pointcloud

from .raycast import Raycast
from omniisaacgymenvs.utils.camera_renderer import Renderer

# import util package
import numpy as np
import torch

from cprint import *
import time
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_multiply, Transform3d, quaternion_to_matrix
from pytorch3d.structures import join_meshes_as_batch
from pytorch3d.utils import ico_sphere

from omniisaacgymenvs.controller.controller import Controller

from omni.isaac.core.objects import cuboid

import pandas as pd

# import stable baselines
from stable_baselines3 import PPO


class TofOptTask(RLTask):

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

        rot_mat = quaternion_to_matrix(torch.tensor([self._robot_rotations], dtype=torch.float))
        self.t = Transform3d().rotate(rot_mat).translate(torch.tensor([self._robot_positions], dtype = torch.float))
        self.t_inv = self.t.inverse()
        self.target_pose = torch.tensor([0.18432, 0.59626, 1.2  ]).repeat(self.num_envs, 1).to(self.device)
        
        # self.model = PPO.load("/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-05-20/15-32-49/results/520/TofSensor2/model/model_50.zip")
        self.model = PPO.load("/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-05-13/09-59-21/results/513/TofSensor2/model/model_200.zip")
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
        self.back_dimensions = [10.2286, 0.1524, 5.04]
        object_loader.load_table(
            # [0.21505, 0.67514, 0.635], #[0.21505, 0.67514, 0.635],
            [0.21505, 0.67514, 0.635],
            self._task_cfg['sim']["Table"]["quaternion"],
            # self.back_dimensions, "back") #[0.2286, 0.005, 1.27], "back")
            [0.2286, 0.005, 1.27], "back")
        
        #Bin Left
        object_loader.load_table(
            [0.10046, 0.29646, 0.635], #[0.10046, 0.59646, 0.635],
            self._task_cfg['sim']["Table"]["quaternion"],
            [0.005, 0.01524, 1.27], "left") #[0.005, 0.1524, 1.27]
        
        #Bin Right
        object_loader.load_table(
            [0.33005, 0.29646, 0.635], #[0.33005, 0.59646, 0.635],
            self._task_cfg['sim']["Table"]["quaternion"],
            [0.005, 0.01524, 1.27], "right") #[0.005, 0.1524, 1.27]
        
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

            self.sensor_meshes = join_meshes_as_batch([ico_sphere(level=0).to(self._device) for mesh in range(self._num_envs)])
            self.sensor_meshes_verts = self.sensor_meshes.verts_padded() / self._cfg["raycast_mesh_size"]
            self.sensor_meshes_norml = self.sensor_meshes.verts_normals_padded()
            
            plot_pointcloud(self.sensor_meshes[0], "orig_mesh")
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
                sensor_mesh=self.sensor_meshes_verts, sensor_norml=self.sensor_meshes_norml)

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

    # def get_info(self) -> dict:
    #     pass

    #     info = {
    #         "robot_joints": self.robot_joints,
    #         "tof_readings": self._step,
    #     }

    #     return info
    
    def get_observations(self) -> dict:

        self.update_cache_state()
        
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
        
            # Bin
            # other_scale = torch.tensor([[0.2286, 0.1524, 1.0668], [0.2286, 0.005, 1.27] , [0.005, 0.1524, 1.27], [0.005, 0.1524, 1.27], [0.2286, 0.005, 0.1524]], device = 'cuda')
            other_scale = torch.tensor([[0.2286, 0.1524, 1.0668], [0.2286, 0.005, 1.27], [0.005, 0.01524, 1.27], [0.005, 0.01524, 1.27], [0.2286, 0.005, 0.1524]], device = 'cuda')
            # Concatenate to mianipulated objects
            self.scale_sizes = torch.cat((self.scale_size, other_scale))

            # Get new vertices
            self.sensor_meshes_verts = self.sensor_meshes.verts_padded() / self._cfg["raycast_mesh_size"]
            self.sensor_meshes_norml = self.sensor_meshes.verts_normals_padded()

            self.raycast_reading, self.raytrace_cover_range, self.raytrace_dev , self.debug_ray_hit_points_list, self.object_tracker = self.raytracer.raytrace_step(
                gripper_pose,
                gripper_rot,
                cur_object_pose,
                cur_object_rot,
                self.scale_sizes,
                sensor_mesh=self.sensor_meshes_verts,
                sensor_norml=self.sensor_meshes_norml)

            self.obs_buf = torch.cat([self.robot_joints, self.raycast_reading],
                                     dim=1)

        if self._task_cfg['Training']["use_oracle"]:
            
            self.obs_buf = torch.cat([
                self.raycast_reading,
                self.robot_joints # 6
            ], dim=1)

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

        # Deform mesh
        # deform_verts = torch.full(self.sensor_meshes.verts_packed().shape, 0.0, device=self.device, requires_grad=True)
        actions /= 100

        self.sensor_meshes = self.sensor_meshes.offset_verts(actions.reshape(self.num_envs * self.sensor_meshes.num_verts_per_mesh()[0], 3)) # 3 is for x, y, z
        
        # Use model trained on control
        obs = torch.cat([
                # self.raycast_reading, # 128
                self.robot_joints # 6
                ], dim=1)
        actions, _states = self.model.predict(obs)
        # actions.zero_()
        # actions[:,3] += 0.01
        self._robots.set_joint_position_targets(actions[:, :6] + self._robots.get_joint_positions())

        self._step += 1
        if not self._env._world.is_playing():
            return

    def post_reset(self):

        self.robot.initialize()
        self.robot.disable_gravity()

        for i in range(self.num_envs):
            self.grippers[i].initialize(
                articulation_num_dofs=self._robots.num_dof)
        self.reset()

    def calculate_metrics(self) -> None:

        # Reward from number of hits
        self.rew_buf = torch.sum(torch.gt(self.raycast_reading, 0), dim=1).float() / 100

        # import pdb; pdb.set_trace()
        cprint.ok(self.rew_buf)
        return self.rew_buf

    def is_done(self) -> None:

        if (self._step + 1) % 401 == 0: # Was 201 Episode length or horizon *1001*
            plot_pointcloud(self.sensor_meshes[0], "final_mesh")
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

        for i in range(2): 
            self._env._world.step(render=False)

        self.init_ee_dev_local_pos, _ = self._end_effector.get_local_poses()
        self.init_ee_dev_local_pos[:, 0] += random_x

        # reset goal orientation
        self.target_angle = -self.rand_orientation[:, 2].clone()
        self.init_angle_z_dev = -self.target_angle.clone()
        self.get_target_pose()
        self._step = 0

        # Reset mesh
        self.sensor_meshes = join_meshes_as_batch([ico_sphere(level=0).to(self._device) for mesh in range(self._num_envs)])
        self.sensor_meshes_verts = self.sensor_meshes.verts_padded() / self._cfg["raycast_mesh_size"]
        self.sensor_meshes_norml = self.sensor_meshes.verts_normals_padded()

