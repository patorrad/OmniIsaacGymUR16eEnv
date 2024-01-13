import torch


def recover_rule_based_action(num_envs, device, _end_effector, target_position,
                              angle_dev, _robots):

    delta_pose = torch.zeros((num_envs, 6)).to(device)

    delta_pose[:, 5] = torch.as_tensor(torch.pi / 200)

    cur_pos, _ = _end_effector.get_local_poses()
    cur_pos[:, 0] = -cur_pos[:, 0]

    delta_pose[:, 0] = target_position[:, 0] - cur_pos[:, 0]
    delta_pose[:, 1] = target_position[:, 1] - cur_pos[:, 1]

    satified_index = torch.where(abs(angle_dev) < 0.02)[0]

    if torch.numel(satified_index) != 0:
        delta_pose[satified_index, 5] = 0

    jacobians = _robots.get_jacobians(clone=False)
    delta_dof_pos = ik(jacobian_end_effector=jacobians[:, 8, :, :],
                       delta_pose=delta_pose)

    return delta_dof_pos / 2, delta_pose


def recover_action(action, limit, _env, _robots):

    # delta pose
    action = torch.clip(action, -1, 1)

    delta_pose = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]
    control_time = _env._world.get_physics_dt()
    delta_pose = delta_pose * control_time

    jacobians = _robots.get_jacobians(clone=False)
    delta_dof_pos = ik(jacobian_end_effector=jacobians[:, 6, :, :],
                       delta_pose=delta_pose)
    delta_dof_pos = torch.clip(delta_dof_pos, -torch.pi, torch.pi)

    return delta_dof_pos, delta_pose


def ik(jacobian_end_effector, delta_pose, damping_factor=0.05):
    """
        Damped Least Squares method: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
        """

    # compute position and orientation error
    delta_pose = delta_pose[:, :, None]

    # solve damped least squares (dO = J.T * V)
    transpose = torch.transpose(jacobian_end_effector, 1, 2)
    lmbda = torch.eye(6).to(jacobian_end_effector.device) * (damping_factor**2)
    return (transpose @ torch.inverse(jacobian_end_effector @ transpose +
                                      lmbda) @ delta_pose).squeeze(dim=2)

# from omni.isaac.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
# def compute_ik(target_position, target_orientation):

#         self._kinematics_solver = LulaKinematicsSolver(
#             robot_description_path=
#             "/omniisaacgymenvs/cfg/robot/robot_descriptor.yaml",
#             urdf_path="/omniisaacgymenvs/assests/robots/ur16e/ur16e.urdf")
#         from omni.isaac.core.articulations import Articulation

#         # robot_prim_path = "/panda"
#         # path_to_robot_usd = self.current_directory + self._task_cfg['sim']["URRobot"]['robot_path']

#         # add_reference_to_stage(path_to_robot_usd, robot_prim_path)
#         # articulation = Articulation(self.default_zero_env_path + "/robot")
#         # self._robots = ArticulationView(
#         #     prim_paths_expr="/World/envs/.*/robot", name="robot_view", reset_xform_properties=False)
#         from omni.isaac.core.robots.robot import Robot
#         _robot = Robot(self.default_zero_env_path + "/robot")
#         _robot.initialize()

#         self._articulation_kinematics_solver = ArticulationKinematicsSolver(
#             _robot, self._kinematics_solver, "ee_link")

#         self._kinematics_solver.set_robot_base_pose(
#             robot_position=np.array(self._robot_positions),
#             robot_orientation=np.array(self._robot_rotations))

#         joint_positions, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
#             target_position, target_orientation)
#         return joint_positions, success
