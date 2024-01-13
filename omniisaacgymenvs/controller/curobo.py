def init_curobo(self):

        # # CuRobo
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

        # from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig
        from curobo.rollout.rollout_base import Goal
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.robot import JointState, RobotConfig
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
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
        from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig

        self.tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                        "ur16e.yml"))["robot_cfg"]
        j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

        world_cfg_table = WorldConfig.from_dict(
            load_yaml(
                join_path(get_world_configs_path(), "collision_table.yml")))
        world_cfg_table.cuboid[0].pose[2] -= 0.002
        world_cfg1 = WorldConfig.from_dict(
            load_yaml(
                join_path(get_world_configs_path(),
                          "collision_table.yml"))).get_mesh_world()
        world_cfg1.mesh[0].name += "_mesh"
        world_cfg1.mesh[0].pose[2] = -10.5

        n_obstacle_cuboids = 30
        n_obstacle_mesh = 10

        world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid,
                                mesh=world_cfg1.mesh)

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                "obb": n_obstacle_cuboids,
                "mesh": n_obstacle_mesh
            },
            # use_fixed_samples=True,
        )
        ik_solver = IKSolver(ik_config)

        self.position_grid_offset = self.tensor_args.to_device(
            self.get_pose_grid(10, 10, 5, 0.5, 0.5, 0.5))

        fk_state = ik_solver.fk(ik_solver.get_retract_config().view(1, -1))
        self.goal_pose = fk_state.ee_pose
        self.goal_pose = self.goal_pose.repeat(
            self.position_grid_offset.shape[0])
        self.goal_pose.position += self.position_grid_offset

        return ik_solver


# def render_curobo(self):

#         # frame skip
#         ee_translation_goal, ee_orientation_teleop_goal = self._manipulated_object.get_local_poses(
#         )

#         ee_world_pose, _ = self._manipulated_object.get_world_poses()

#         self.goal_pose.position[:] = torch.as_tensor([-0.3, 0.6, 1.04]).to(
#             self.device) + self.position_grid_offset
#         self.goal_pose.quaternion[:] = torch.as_tensor([1, 0, 0, 0]).to(
#             self.device)  # ik_goal.quaternion[:]
#         # tranformed
#         pre_transoform_goal_pose = self.goal_pose.clone()

#         pre_transoform_goal_pose.position[:, 0] += ee_world_pose[
#             0, 0] - ee_translation_goal[0, 0]

#         transform = Transform3d(device=self.device).rotate(
#             quaternion_to_matrix((torch.as_tensor(
#                 self._robot_rotations).to(device=self.device))))

#         # rotation_matrix = quaternion_to_matrix(
#         #     (torch.as_tensor(orientation).to(device)))

#         self.goal_pose.position[:] -= torch.as_tensor(
#             [self._robot_positions]).to(device=self.device)

#         self.goal_pose.position[:] = transform.transform_points(
#             self.goal_pose.position[:].clone().to(device=self.device))

#         self.goal_pose.quaternion[:] = quaternion_multiply(
#             quaternion_invert(
#                 torch.as_tensor(self._robot_rotations).to(device=self.device)),
#             self.goal_pose.quaternion[:])

#         result = self.curo_ik_solver.solve_batch(self.goal_pose)

#         # succ = torch.any(result.success)

#         # get spheres and flags:

#         # transform back

#         self.draw_points(pre_transoform_goal_pose, result.success)