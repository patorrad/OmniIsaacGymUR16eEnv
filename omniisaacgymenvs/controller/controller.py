from omniisaacgymenvs.controller.ik import recover_action, recover_rule_based_action, diffik

from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion
from omni.isaac.core.utils.types import ArticulationActions
from omniisaacgymenvs.controller.curobo import MotionGeneration

from cprint import *
import torch

class Controller:

    def __init__(self,
                 _robots,
                 isaac_sim_robot,
                 _env,
                 _end_effector,
                 velocity_limit,
                 _device,
                 num_envs,
                 control_type="diffik",
                 datagen=True) -> None:

        self._robots = _robots
        self._env = _env
        self._end_effector = _end_effector
        self.velocity_limit = velocity_limit
        self._device = _device
        self.control_type = control_type
        self.isaac_sim_robot = isaac_sim_robot
        
        self.num_envs = num_envs
        
        self.data_gen = datagen
        if control_type == "MotionGeneration":
            self.motion_generation = MotionGeneration(self._robots,
                                                      self._env._world,
                                                      n_envs=self.num_envs)

        self.joint_positions = torch.zeros((self.num_envs, 10)).to(self._device)

        self.condition = torch.tensor([]*self.num_envs, device=self._device)

    def forward(
        self,
        actions,
        target_ee_position=None,
        target_ee_orientation=None,
        angle_z_dev=None,
        envs=[], # What environments are ready to be controlled once object settle
        rays=torch.tensor([]),
        ray_readings=torch.tensor([]),
    ):
        actions = actions.to(self._device)
        actions[:, [2, 3, 4]] = 0
        delta_dof_pos, delta_pose = recover_action(actions,
                                                   self.velocity_limit,
                                                   self._env, self.isaac_sim_robot)
        cur_ee_pos, cur_ee_orientation = self._end_effector.get_local_poses()
        target_ee_pos = cur_ee_pos + delta_pose[:, :3]

        if self.control_type == "diffik":

            current_dof = self.isaac_sim_robot.get_joint_positions()
            targets_joint = current_dof[:, :6] + delta_dof_pos[:, :6]
            # import pdb; pdb.set_trace()
            import torch
            zeros = torch.zeros((2,4), device="cuda:0")
            targets_joint = torch.cat((targets_joint, zeros), dim=1)


            # targets_joint[:, -1] = 0

            self.isaac_sim_robot.set_joint_position_targets(targets_joint)

            for i in range(1):
                self._env._world.step(render=False)

        elif self.data_gen and self.control_type == "MotionGeneration":

            # target_ee_orientation = quaternion_multiply(
            #     quaternion_invert(axis_angle_to_quaternion(delta_pose[:, 3:])),
            #     cur_ee_orientation)

            for i in range(2):

                robot_joint = self.isaac_sim_robot.get_joint_positions()
             
              
                robot_joint = self.motion_generation.step_path(
                    target_ee_position, target_ee_orientation, robot_joint)

                self.isaac_sim_robot.apply_action(ArticulationActions(robot_joint, ))
                for i in range(1):
                    self._env._world.step(render=False)

        elif self.control_type == "MotionGeneration":

            target_ee_orientation = quaternion_multiply(
                quaternion_invert(axis_angle_to_quaternion(delta_pose[:, 3:])),
                cur_ee_orientation)

            for i in range(4):

                robot_joint = self.isaac_sim_robot.get_joint_positions()
             
              
                robot_joint = self.motion_generation.step_path(
                    target_ee_pos, target_ee_orientation, robot_joint)

                self.isaac_sim_robot.apply_action(ArticulationActions(robot_joint, ))
                for i in range(1):
                    self._env._world.step(render=False)

        else:
            import torch

            delta_dof_pos, delta_pose = recover_rule_based_action(
                self.num_envs, self._device, self._end_effector,
                target_ee_position, angle_z_dev, self.isaac_sim_robot)
            current_dof = self.isaac_sim_robot.get_joint_positions()
            
            targets_dof = torch.zeros((self.num_envs, 10)).to(self._device)
            targets_dof[:,:6] = current_dof[:,:6] + delta_dof_pos[:,:6]

            # Gripper Control
            if rays.shape[0] == 0:
                targets_dof[:, 6:] = torch.zeros((self.num_envs, 4)).to(self._device)
            
            elif rays.shape[0] > 0 and ray_readings.shape[0] > 0:
                # Find rays belonging to target object
                ray_index = torch.where(rays == 1)
                readings = ray_readings.clone()
                mask = torch.zeros(readings.flatten().shape[0], dtype=torch.bool, device='cuda:0')
                mask[ray_index] = True
                mask = mask.reshape(self.num_envs, 4, 64)
                readings = readings.reshape(self.num_envs, 4, 64)

                # Apply the mask to the tensor, setting masked elements to NaN
                masked_tensor = torch.where(mask, readings, torch.tensor(float('nan'), device='cuda:0'))

                # Calculate the mean along a specific dimension (e.g., axis=2) while ignoring NaN values
                mean_value = torch.nanmean(masked_tensor, dim=2)
                mean_value = torch.nan_to_num(mean_value, nan=0.)
                if torch.any(torch.isnan(mean_value)):
                    import pdb; pdb.set_trace()
            
                vel = self.isaac_sim_robot.get_joint_velocities()[:,:6].norm(dim=1) < 0.01
                dist = delta_pose.norm(dim=1) < 0.05
                condition = vel * dist
                self.condition = condition
                mean_value[~condition] = 0. 
                # targets_dof[~condition, :] = self.joint_positions[~condition, :]
                # import pdb; pdb.set_trace()
                # print(mean_value)
                
                # target_subset = targets_dof[:, 6:]
                # mask = target_subset < mean_value
                # target_subset[mask] = mean_value[mask]
                # target_subset[target_subset < 0.1] = 0.1
                # targets_dof[:, 6:] = target_subset

                # targets_dof[:, 6:] = mean_value
                
                # targets_dof[:, 6:] = torch.zeros((self.num_envs, 4)).to(self._device)

            self.joint_positions = targets_dof[:, :]
            self.isaac_sim_robot.set_joint_position_targets(targets_dof[envs,:], indices=envs)

            # for i in range(1):
                # self._env._world.step(render=False)

        return target_ee_pos, self.condition


