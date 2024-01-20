from omniisaacgymenvs.controller.ik import recover_action, recover_rule_based_action, diffik

from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion
from omni.isaac.core.utils.types import ArticulationActions


def controller(actions,
               _robots,
               _env,
               _end_effector,
               motion_generation,
               velocity_limit,
               _device,
               control_type="diffik"):
    actions = actions.to(_device)
    actions[:, [2, 3, 4]] = 0
    delta_dof_pos, delta_pose = recover_action(actions, velocity_limit, _env,
                                               _robots)
    cur_ee_pos, cur_ee_orientation = _end_effector.get_local_poses()
    target_ee_pos = cur_ee_pos + delta_pose[:, :3]

    if control_type == "diffik":

        current_dof = _robots.get_joint_positions()
        targets_dof = current_dof + delta_dof_pos[:, :6]

        targets_dof[:, -1] = 0

        _robots.set_joint_position_targets(targets_dof)

        for i in range(1):
            _env._world.step(render=False)

    elif control_type == "MotionGeneration":

        target_ee_orientation = quaternion_multiply(
            quaternion_invert(axis_angle_to_quaternion(delta_pose[:, 3:])),
            cur_ee_orientation)

        for i in range(4):

            robot_joint = _robots.get_joint_positions()

            robot_joint = motion_generation.step_path(target_ee_pos,
                                                      target_ee_orientation,
                                                      robot_joint)

            _robots.apply_action(ArticulationActions(robot_joint, ))
            for i in range(1):
                _env._world.step(render=False)

    # else:
    #     delta_dof_pos, delta_pose = recover_rule_based_action(
    #         num_envs, device, _end_effector,
    #         target_ee_position, angle_z_dev, _robots)
    #     current_dof = _robots.get_joint_positions()
    #     targets_dof = torch.zeros((num_envs, 6)).to(device)
    #     targets_dof = current_dof + delta_dof_pos[:6]
    return target_ee_pos
