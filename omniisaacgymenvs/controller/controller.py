from omniisaacgymenvs.controller.ik import recover_action, recover_rule_based_action, diffik

from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion
from omni.isaac.core.utils.types import ArticulationActions
from omniisaacgymenvs.controller.curobo import MotionGeneration


class Controller:

    def __init__(self,
                 _robots,
                 isaac_sim_robot,
                 _env,
                 _end_effector,
                 velocity_limit,
                 _device,
                 num_envs,
                 control_type="diffik") -> None:

        self._robots = _robots
        self._env = _env
        self._end_effector = _end_effector
        self.velocity_limit = velocity_limit
        self._device = _device
        self.control_type = control_type
        self.isaac_sim_robot = isaac_sim_robot
        
        self.num_envs = num_envs

        if control_type == "MotionGeneration":
            self.motion_generation = MotionGeneration(self._robots,
                                                      self._env._world,
                                                      n_envs=self.num_envs)

    def forward(
        self,
        actions,
        target_ee_position=None,
        angle_z_dev=None,
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
            targets_dof = current_dof + delta_dof_pos[:, :6]

            targets_dof[:, -1] = 0

            self.isaac_sim_robot.set_joint_position_targets(targets_dof)

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
            targets_dof = torch.zeros((self.num_envs, 6)).to(self._device)
            targets_dof = current_dof + delta_dof_pos[:,:6]
            
            self.isaac_sim_robot.set_joint_position_targets(targets_dof)

            for i in range(1):
                self._env._world.step(render=False)

        return target_ee_pos
