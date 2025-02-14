# init extension
import imp
from omni.isaac.core.utils.extensions import enable_extension

enable_extension("omni.isaac.debug_draw")
enable_extension("omni.isaac.manipulators")
enable_extension("omni.isaac.motion_generation")
enable_extension("omni.replicator.isaac")

# import env setting
from omniisaacgymenvs import envs
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
from pytorch3d.transforms import axis_angle_to_quaternion

from omniisaacgymenvs.controller.controller import Controller

from omni.isaac.core.objects import cuboid

import pandas as pd

from omniisaacgymenvs.utils.domain_randomization.randomize import Randomizer
from scipy.spatial.transform import Rotation as R

import wandb

from omniisaacgymenvs.utils.bagpy_read_3_classification import CNNClassifier


class CustomGripperTask(RLTask):

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
        self._num_steps = self._task_cfg["env"]["num_steps"]

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

        # Raycasting variables
        self.object_tracker = np.array([])
        self.raycast_reading = np.array([]) 

        self.target = cuboid.VisualCuboid(
            "/World/envs/env_0/target",
            position=np.array([0., 0., 0.]),
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
        
        self.randomization_params = self._task_cfg["domain_randomization"]["randomization_params"]
        self.randomization_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.classifier_model = CNNClassifier()
        self.classifier_model.load_state_dict(torch.load("/home/heisenberg/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/utils/model.pth"))
        self.classifier_model.to(self.device)
        self.classifier_model.eval()
        self.object_grasped = torch.zeros(self.num_envs, device=self.device)

        return

    def set_up_scene(self, scene) -> None:

        from omniisaacgymenvs.utils.robot_loader import ROBOT
        from omniisaacgymenvs.utils.object_loader import Object

        # load robot
        robot = ROBOT(self.num_envs, self.default_zero_env_path,
                      self._robot_positions, self._robot_rotations,
                      self._robot_dof_target, self._sim_config,
                      self._task_cfg['sim']["URRobot"]['robot_path'],
                      self._task_cfg['sim']["URRobot"]['num_sensors'])

        self.robot = robot.load_UR()
        # self.grippers = robot.add_gripper()

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
        
        # Pod
        # object_loader.load_pod(
        #     [-0.3, 0.45, 0.0],
        #     self._task_cfg['sim']["Table"]["quaternion"],
        #     np.array(self._task_cfg['sim']["Table"]["scale"]))


        super().set_up_scene(scene)

        self._robots, self._end_effector, self.wrist_2_link, self.sensor_0, self.sensor_1, self.sensor_2, self.sensor_3, self.finger_0  = robot.add_scene(scene)

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

        self.old_target_pose, _ = self._manipulated_object_2.get_world_poses()
        self.old_stacked_pose, self.old_stacked_rot = self._manipulated_object.get_world_poses()

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

        # self._pod = object_loader.add_scene(scene, "/World/envs/.*/pod",
        #                                       "pod_view")

        # self.pod_positions = self._pod.get_world_poses()

        if self._cfg["raycast"]:

            self.sensor_radius = torch.as_tensor(
                self._task_cfg['sim']["URRobot"]['sensor_radius']).repeat(
                    self.num_envs, 1).to(self._device)
            
            self.raytracer = Raycast(
                self._cfg["raycast_width"], self._cfg["raycast_height"], [
                    self._manipulated_object.prim_paths[0],
                    self._manipulated_object_2.prim_paths[0],
                    self._base.prim_paths[0],
                    self._back.prim_paths[0],
                    self._left.prim_paths[0],
                    self._right.prim_paths[0],
                    self._top.prim_paths[0],
                ], ['Cylinder', 'Cube', 'Cube', 'Cube','Cube', 'Cube', 'Cube'], self._task_cfg, self._cfg, self.num_envs, self._device,
                self.sensor_radius)

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
        
        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)

         # Viewer location
        self.set_initial_camera_params(camera_position=self._task_cfg["viewport"]["camera_position"])

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
            
            # Retrieve sensor poses
            self.sensor_world_poses = [self.sensor_0.get_world_poses()[0], self.sensor_1.get_world_poses()[0], self.sensor_2.get_world_poses()[0], self.sensor_3.get_world_poses()[0]]
         
            self.raycast_reading, self.debug_ray_hit_points_list, self.object_tracker = self.raytracer.raytrace_step(
                gripper_pose,
                gripper_rot,
                cur_object_pose,
                cur_object_rot,
                self.scale_sizes,
                sensor_radius=self.sensor_radius,
                sensor_poses=self.sensor_world_poses)

            # import pdb; pdb.set_trace()
            # Normalize sensor readings and normalize robot joint reading

            sensor_reading = (self.raycast_reading[:,132::8][:,1:7] -  0.02) / (0.12 - 0.02)
            self.obs_buf = torch.cat([self.robot_joints[:,7:10] * 10, sensor_reading, self.object_grasped.unsqueeze(dim=1)], dim=1).cpu().numpy()
            # SAC
            # self.obs_buf = self.obs_buf.cpu().numpy()
            # # Try quaternion representation
            # object_pose, q_batch = self._manipulated_object_2.get_world_poses()
            # w, x, y, z = q_batch[:, 0], q_batch[:, 1], q_batch[:, 2], q_batch[:, 3]
            # yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # Compute yaw from quaternion
            # obs = torch.cat([torch.sin(yaw).unsqueeze(1), torch.cos(yaw).unsqueeze(1)], dim=1)  # Shape: (num_envs, 2)
            # self.obs_buf = torch.cat([self.robot_joints[:,7:8] * 10, obs], dim=1)

            # self.obs_buf = self.robot_joints[:,7:8] * 10


        if self._task_cfg['Training']["use_oracle"]:
            self.obs_buf = torch.cat([
                current_euler_angles_x[:, None], self.target_angle[:, None],
                self.angle_z_dev[:,
                                 None], cur_position, self.target_ee_position,
                self.target_ee_position - cur_position, self.robot_joints
            ],
                                     dim=1)

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

        ###################### Rotate object
        if self._step == 2:
            pose, rotation = self._manipulated_object_2.get_world_poses()
            yaw_max = 0# -torch.pi/16
            yaw_min = -torch.pi/4
            yaw = np.random.uniform(yaw_min, yaw_max)
            pitch, roll = 0.0, 0.0
            quaternion = np.tile(R.from_euler('xyz', [yaw, pitch, roll]).as_quat(), (self.num_envs, 1)).astype(np.float32)
            # import pdb; pdb.set_trace()
            self._manipulated_object_2.set_world_poses(self.old_target_pose, torch.from_numpy(quaternion).to(device="cuda"))
            self._manipulated_object.set_world_poses(self.old_stacked_pose, self.old_stacked_rot)
            print(f"Rotating object by {yaw} degrees {quaternion}")

        #####################

        self._step += 1
        if not self._env._world.is_playing():
            return

        if self._task_cfg["sim"]["Design"] and self._cfg["raycast"]:
            if self._step == 1:
                self.sensor_radius = self.raytracer.update_params(actions[:,
                                                                          6:])
                target_ee_pos, _ = self._end_effector.get_local_poses()
            elif self._step >= 1:
                target_ee_pos = self.controller.forward(actions[:, :6])
        elif self._task_cfg["sim"]["Dataset"]:
            # Check object velocity to start robot control
            object_vel = self._manipulated_object_2.get_linear_velocities().norm(dim=1)
            indices = torch.nonzero(object_vel < 0.1).flatten()

            self.target_ee_position, self.target_ee_rotation = self._manipulated_object_2.get_local_poses()
            
            self.target_ee_position = self.target_ee_position - torch.tensor([[0.0, 0.3, 0.2]]*self._num_envs, device='cuda:0')
            self.target.set_local_pose(self.target_ee_position[0].cpu(), self.target_ee_rotation[0].cpu())
            

            target_ee_pos = self.controller.forward(actions,
                                                    self.target_ee_position,
                                                    angle_z_dev=self.angle_z_dev,
                                                    envs=indices,
                                                    rays=self.object_tracker, # 0 - cylinder,1 - box,2 - top ,3 - back,4 - base, 5 -left, 6 right 
                                                    ray_readings=self.raycast_reading)
        else:
            # Check object velocity to start robot control
            object_vel = self._manipulated_object.get_linear_velocities().norm(dim=1)
            indices = torch.nonzero(object_vel < 0.15).flatten()

            self.target_ee_position, self.target_ee_rotation = self._manipulated_object_2.get_local_poses()
            
            self.target_ee_position = self.target_ee_position - torch.tensor([[0.0, 0.3, 0.2]]*self._num_envs, device='cuda:0')
            self.target.set_local_pose(self.target_ee_position[0].cpu(), self.target_ee_rotation[0].cpu())
            
            # Convert PPO action from [-1,1] to [0,0.85]
            self.actions = (actions + 1) / 2 * 0.85 / 10  # Now in [0, 0.85]
            # SAC
            # self.actions = torch.tensor(self.actions, device=self.device)
            target_ee_pos = self.controller.forward(self.actions,
                                                    self.target_ee_position,
                                                    angle_z_dev=self.angle_z_dev,
                                                    envs=indices,
                                                    rays=self.object_tracker, # 0 - cylinder,1 - box,2 - top ,3 - back,4 - base, 5 -left, 6 right 
                                                    ray_readings=self.raycast_reading)

        curr_position, _ = self._end_effector.get_local_poses()
        self.cartesian_error = torch.linalg.norm(curr_position - target_ee_pos,
                                                 dim=1)
        
        # Grasp object
        with torch.no_grad():
            outputs = self.classifier_model(self.raycast_reading[:,128:192] * 10)
            self.predicted_classes = torch.argmax(outputs, 1)
            # ✅ New detected indices
            new_indices = torch.nonzero(self.predicted_classes == 2).flatten()

            # ✅ Append new indices, ensuring they remain in self.indices
            self.indices = torch.cat((self.indices, new_indices)).unique()

            # Update grasp flag for new indices
            self.grasp_flags[self.indices] = True


        if not hasattr(self, "old_finger_pose"):
                self.old_finger_pose, _ = self.sensor_2.get_world_poses()
        if True: # Input from neural network
            finger_pose, _ = self.sensor_2.get_world_poses()
            object_pose, object_rot = self._manipulated_object_2.get_world_poses()
            object_pose_trans = object_pose + (finger_pose - self.old_finger_pose)
            # import pdb; pdb.set_trace()
            self._manipulated_object_2.set_world_poses(object_pose_trans[self.indices,:], object_rot[self.indices,:], indices=self.indices)
            self.old_finger_pose = finger_pose.clone()

    def post_reset(self):

        self.robot.initialize()
        self.robot.disable_gravity()
        self.reset()

    def calculate_metrics(self) -> None:

        if self._step > 2: 

            self.rew_buf = torch.tensor([0.0] * self.num_envs, device=self.device)
            
            joint_position = self.robot_joints[:, 7]*10  # Shape: (num_envs,)

            # Compute absolute distance to target (0.5)
            distance_to_target = torch.abs(joint_position - 0.5)
            
            # Object rotation
            object_pose, q_batch = self._manipulated_object_2.get_world_poses()
            w, x, y, z = q_batch[:, 0], q_batch[:, 1], q_batch[:, 2], q_batch[:, 3]

            # Compute yaw difference: 2 * atan2(z, w)
            yaw_diff_rad = np.pi + torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)) #approximation if object is on table: 2 * torch.atan2(z, w)

            # Convert to degrees and return absolute yaw difference
            yaw_diff_deg = torch.abs(torch.rad2deg(yaw_diff_rad))
            yaw_diff_rew = 1.0 * torch.cos(yaw_diff_rad)

            # Base reward: Penalize distance to target
            position_reward = -distance_to_target  # Closer = higher reward

            # # ✅ Stability Bonus: Give extra reward for being near 0.5 (within ±0.05)
            # threshold = 0.1
            # within_target = (distance_to_target < threshold).float()
            # extra_bonus = within_target * 1.0  # Large bonus for staying close
            # Optional bonus for being near 0° yaw (e.g., within ±5° radians)
            within_threshold = yaw_diff_rew > torch.tensor([0.95] * self.num_envs, device=self.device)
            within_threshold = within_threshold.float() * 1.0  # Extra reward for near-perfect alignment
            

            # # Extension penalty
            # finger_extension = self.actions.squeeze()  # Assuming action represents finger position
            # finger_penalty = -5.0 * torch.clamp(finger_extension - 0.7, min=0) ** 2  # Penalize when > 0.7
            # ✅ Get Angular Velocity of the Object
            angular_velocity = self._manipulated_object_2.get_angular_velocities()  # Assuming this returns (num_envs, 3)
            yaw_velocity = torch.abs(angular_velocity[:, 2])  # Extract yaw (Z-axis) velocity

            # ✅ Angular Velocity Penalty (Scaled by cos(yaw))
            angular_velocity_penalty = -0.5 * yaw_velocity * torch.abs(yaw_diff_rew)

            # Get current and previous joint velocity
            joint_velocity = self._robots.get_joint_velocities()[:, 7]  # Current velocity
            if not hasattr(self, "prev_joint_velocity"):
                self.prev_joint_velocity = torch.zeros_like(joint_velocity)  # Initialize if first step

            # ✅ Velocity penalty (encourages smooth speed)
            velocity_penalty = -2.0 * torch.abs(joint_velocity)  

            # ✅ Jerk penalty (encourages smooth acceleration)
            jerk = joint_velocity - self.prev_joint_velocity
            jerk_penalty = -2.5 * torch.abs(jerk)  # Higher penalty for sudden changes

            # Grasp object reward
            # import pdb; pdb.set_trace()
            object_grasped_rew = torch.tensor([0.0] * self.num_envs, device=self.device)
            object_grasped_rew[self.indices] = 3.0 
            self.object_grasped = torch.tensor([0.0] * self.num_envs, device=self.device)
            self.object_grasped[self.indices] = 1.0

            ##################
            object_positions, _ = self._manipulated_object_2.get_local_poses()
            # ✅ Identify environments where the object was grasped for the first time
            newly_grasped_mask = self.saved_positions == float('inf')  # Check which haven't been stored
            update_indices = self.indices[newly_grasped_mask[self.indices]]  # Only update first-time grasps

            # ✅ Store object positions only in those environments
            self.saved_positions[update_indices] = object_positions[:,1][update_indices]
            # self.saved_positions[self.grasp_flags] = object_positions[self.grasp_flags, 1].clone()
            
            # print(object_positions[:,1]*1.1, self.saved_positions)
            # print(object_grasped_rew)
            # object_grasped_rew[(object_grasped_rew == 4.0).bool() & (object_positions[:,1]*1.1 > self.saved_positions).bool()] = -2.0 + object_grasped_rew[(object_grasped_rew == 4.0).bool() & (object_positions[:,1]*1.1 > self.saved_positions).bool()] 
            # object_grasped_rew[(object_grasped_rew == 4.0).bool() & (object_positions[:,1]*1.1 < self.saved_positions).bool()] = 5.0 + object_grasped_rew[(object_grasped_rew == 4.0).bool() & (object_positions[:,1]*1.0001 < self.saved_positions).bool()]
            # Compute object movement relative to saved position
            
            self.object_movement[self.grasp_flags] = self.saved_positions[self.grasp_flags] - object_positions[self.grasp_flags, 1]  # Positive if moving closer

            # Define scaling factor for linear reward
            scaling_factor = 50.0  # Adjust this to control sensitivity

            # Compute the linear reward (closer = higher reward)
            linear_reward = scaling_factor * self.object_movement

            # Apply rewards based on object movement
            mask_farther = self.object_movement < 0  # Object moved farther
            mask_closer = self.object_movement > 0  # Object moved closer

            # ✅ Penalize if object moves farther
            object_grasped_rew[mask_farther] = linear_reward[mask_farther] + object_grasped_rew[mask_farther]  
            # if torch.any(self.saved_positions != torch.inf):
            #     import pdb; pdb.set_trace()
            # ✅ Reward based on how much closer the object is (linear scaling)
            object_grasped_rew[mask_closer] = -2*linear_reward[mask_closer] + object_grasped_rew[mask_closer]

            print(self.object_movement, object_grasped_rew)
            # if torch.any((object_positions[:,1]*0.999 > self.saved_positions).bool()):
            #     import pdb; pdb.set_trace()
            # object_retrieved = torch.zeros_like(self.rew_buf)
            # import pdb; pdb.set_trace()
            # object_pose_retrieved, _ = self._manipulated_object_2.get_local_poses()
            # object_grasped_rew[(object_grasped_rew == 3).bool() & (object_pose_retrieved[:,1] > 0.6).bool()] = -2.0
            # object_grasped_rew[(object_grasped_rew == 3).bool() & (object_pose_retrieved[:,1] < 0.58).bool()] = -1.0
            # object_grasped_rew[(object_grasped_rew == 3).bool() & (object_pose_retrieved[:,1] < 0.56).bool()] = 1.0
            # object_grasped_rew[(object_grasped_rew == 3).bool() & (object_pose_retrieved[:,1] < 0.55).bool()] = 3.0
            # object_grasped_rew[(object_grasped_rew == 3).bool() & (object_pose_retrieved[:,1] < 0.545).bool()] = 4.0
            # object_grasped_rew[(object_grasped_rew == 3).bool() & (object_pose_retrieved[:,1] < 0.54).bool()] = 7.0
            # print(f"{self._step} Object retrieved: {object_grasped_rew} {object_pose_retrieved[:,1]}")
            ##################
            # Push to object
            object_pushed_pose, _ = self._manipulated_object.get_world_poses()
            # if not hasattr(self, "prev_object_pose"):
            #     self.prev_object_pose = top_object_pose.clone()
            # object_pushed_rew = 10 * (top_object_pose[:, 1] - self.prev_object_pose[:, 1])
            # If object falls:
            # object_pushed_rew = object_pushed_rew - 10 * (top_object_pose[:, 2] - self.prev_object_pose[:, 2])

            object_pushed_rew = object_pushed_pose[:,2].clone()
            object_pushed_rew[object_pushed_rew > 1.15] = 0.0

            # ✅ Final Reward Update
            # self.rew_buf = position_reward + extra_bonus + velocity_penalty + jerk_penalty
            self.rew_buf = yaw_diff_rew + within_threshold + velocity_penalty + jerk_penalty + angular_velocity_penalty \
                            + object_grasped_rew  + object_pushed_rew
            
            self.rew_buf = object_grasped_rew #+ velocity_penalty #+ 3 * object_pushed_rew
            # SAC
            # self.rew_buf = self.rew_buf.cpu().numpy()
            # ✅ Store previous velocity for next step
            self.prev_joint_velocity = joint_velocity.clone()

            if not self._cfg["evaluation"]:
                wandb.log({'reward': self.rew_buf.mean().item(),
                        'yaw_diff_rew': yaw_diff_rew.mean().item(),
                        # 'control_penalty': control_penalty.mean().item(),
                        'within_threshold': (within_threshold.float() * 0.5).mean().item(),
                        'angular_velocity': angular_velocity.mean().item(),
                        'jerk_penalty': jerk_penalty.mean().item(),
                        'velocity_penalty': velocity_penalty.mean().item()})

            # if not self._cfg["evaluation"]:
            #     wandb.log({'reward': self.rew_buf.mean().item(),
            #             'proximity_reward': proximity_reward.mean().item(),
            #             'action_penalty': action_penalty.mean().item(),
            #             # 'early_action_penalty': early_action_penalty.mean().item(),
            #             'object_moved_penalty': object_moved_penalty.mean().item(),
            #             'finger_proximity_rewards': finger_proximity_rewards.mean().item()})
        else:
            self.rew_buf = torch.zeros_like(self.rew_buf)
            # SAC
            # self.rew_buf = self.rew_buf.cpu().numpy()

        return self.rew_buf

    def is_done(self) -> None:

        # return torch.full((self.num_envs,), 0, dtype=torch.int)

        if (self._step + 1) % self._num_steps == 0: # Was 201 Episode length or horizon *1001*

            #SAVE DATA TO DISK
            if self._task_cfg["sim"]["Dataset"]:
                self.dataset.to_pickle('dataset.pkl')

            self.episode += 1
            self._step = 0
            self.epidsode_data = []
            self.post_reset()
            return [True for i in range(self.num_envs)]

        return [False for i in range(self.num_envs)]

    def reset(self):

        self._robots.set_joint_positions(
            torch.tensor([1.5606, -0.5066, -1.5923, -1.05, -1.5696,  1.5604,  0, 0, 0, 0],
                         dtype=torch.float).repeat(self.num_envs,
                                                   1).clone().detach())

        for i in range(1):
            self._env._world.step(render=False)
        self.init_ee_link_position, self.init_ee_link_orientation = self._end_effector.get_world_poses(
        )

        # Randomize the manipulated object position
        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

        # if not self._dr_randomizer.randomize:
        #     base_poses, base_rot = self._base.get_world_poses()
        #     for i in range(self.num_envs):
        #         distribution = [(base_poses[i] + torch.tensor([-.08,0,.65], device="cuda:0")).tolist(), (base_poses[i] + torch.tensor([.1,0,.65], device="cuda:0")).tolist()]
        #         self._dr_randomizer.set_dr_distribution_parameters(
        #             distribution,
        #             "rigid_prim_views",
        #             "manipulated_object_view",
        #             "position",
        #             "on_reset")
        #         self._dr_randomizer.set_dr_distribution_parameters(
        #             distribution,
        #             "rigid_prim_views",
        #             "manipulated_object_view_2",
        #             "position",
        #             "on_reset")

        #         self._dr_randomizer.dr.physics_view.step_randomization(torch.tensor([i]))

        for i in range(2): 
            self._env._world.step(render=False)

        self.init_ee_dev_local_pos, _ = self._end_effector.get_local_poses()

        # Old entong code
        self.rand_orientation = torch.zeros((self.num_envs, 3)).to(self.device)
        # reset goal orientation
        self.target_angle = -self.rand_orientation[:, 2].clone() # z axis?
        self.init_angle_z_dev = -self.target_angle.clone()
        self.get_target_pose()
        self._step = 0

        # ✅ Reset self.indices on environment reset
        self.indices = torch.tensor([], dtype=torch.long, device=self.device)

        # self.saved_positions = torch.zeros_like(self._manipulated_object_2.get_local_poses()[0][:,1])
        self.saved_positions = torch.full_like(self._manipulated_object_2.get_local_poses()[0][:,1], fill_value=float('inf'))
        self.object_movement = torch.zeros_like(self.saved_positions)
        self.grasp_flags = torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
