from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omni.isaac.core.utils.prims import get_prim_at_path, get_first_matching_child_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.robots.articulations.surface_gripper import SurfaceGripper
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims.geometry_prim_view import GeometryPrimView

from omni.physx.scripts import utils
from omni.isaac.core.utils.stage import get_current_stage

class ROBOT:

    def __init__(self, num_envs, default_zero_env_path, _robot_positions,
                 _robot_rotations, _robot_dof_target, _sim_config,
                 usd_path, num_sensors) -> None:

        self.num_envs = num_envs

        self.default_zero_env_path = default_zero_env_path
        self._robot_positions = _robot_positions
        self._robot_rotations = _robot_rotations
        self._robot_dof_target = _robot_dof_target
        self._robot_dof_target = _robot_dof_target
        self._sim_config = _sim_config

        self.usd_path = usd_path
        self.num_sensors = num_sensors

    def load_UR(self):

        self.robot = UR10(prim_path=self.default_zero_env_path + "/robot",
                          name="robot",
                          position=self._robot_positions,
                          orientation=self._robot_rotations,
                          attach_gripper=True,
                          usd_path=self.usd_path)
        
        self.robot.set_joint_positions(self._robot_dof_target)
        self.robot.set_joints_default_state(self._robot_dof_target)

        self.robot.num_sensors = self.num_sensors

        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(self.robot.prim_path),
            self._sim_config.parse_actor_config("robot"))

        return self.robot

    def add_gripper(self):
        assets_root_path = get_assets_root_path()

        # gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"
        gripper_usd = "/home/shaktis/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e/short_gripper.usd"

        self.grippers = []

        for i in range(self.num_envs):

            add_reference_to_stage(
                usd_path=gripper_usd,
                # prim_path=f"/World/envs/env_{i}/robot/ee_link")
                prim_path=f"/World/envs/env_{i}/gripper")
            
            # import pdb; pdb.set_trace()
            # import torch
            
            # get_prim_at_path(f"/World/envs/env_{i}/gripper").set_local_pose(torch.tensor([0, 0, 0]), torch.tensor([0, 1, 0, 0]))
            # pose, rot = self.wrist_2_link.get_local_poses()
            # self._end_effector.set_local_poses(torch.tensor([[0, 0, 0] , [0, 0, 0]]), torch.tensor([[0, 1, 0, 0] , [0, 1, 0, 0]]))

            surface_gripper = SurfaceGripper(
                # end_effector_prim_path=f"/World/envs/env_{i}/robot/ee_link",
                end_effector_prim_path=f"/World/envs/env_{i}/gripper",
                translate=0.,#1611,
                direction="y")
            surface_gripper.set_force_limit(value=8.0e1)
            surface_gripper.set_torque_limit(value=10.0e0)

            # surface_gripper.set_world_pose(self._robot_positions, self._robot_rotations)

            # import pdb; pdb.set_trace()
            # import torch
            # from pytorch3d.transforms import Transform3d, quaternion_to_matrix 

            # # Follow target that follows manipulated object 2
            # pose_test, rot_test = surface_gripper.get_local_pose()
            # pose = torch.tensor(pose_test, device='cuda:0')
            # rot = torch.tensor(rot_test, device='cuda:0')

            # rot_mat = quaternion_to_matrix(torch.tensor([self._robot_rotations], dtype=torch.float))
            # t = Transform3d().rotate(rot_mat).translate(torch.tensor([self._robot_positions], dtype = torch.float))
            # pose = t.to('cuda:0').transform_points(pose)
            # surface_gripper.set_local_pose(pose, rot)
            # surface_gripper.initialize(physics_sim_view=None, articulation_num_dofs=self.robot.num_dof)
            self.grippers.append(surface_gripper)

        return self.grippers

    def add_scene(self, scene):

        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/robot",
                                        name="robot_view",
                                        reset_xform_properties=False)

        scene.add(self._robots)

        # end-effectors view
        self._end_effector = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/ee_link",
            name="end_effector_view",
            reset_xform_properties=False)

        scene.add(self._end_effector)

        self.wrist_2_link = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/wrist_2_link",
            name="wrist_2_link_view",
            reset_xform_properties=False)
        scene.add(self.wrist_2_link)

        self.body = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/gripper_mod_01/Simp_suction_array/Simp_suction_array/Body",
            name="finger_0_view",
            reset_xform_properties=False,
            track_contact_forces=True)
        scene.add(self.body)

        # Add sensor views   
        self.sensor_0 = ArticulationView(
            prim_paths_expr="/World/envs/.*/robot/gripper_mod_01/Simp_suction_array/Simp_suction_array/Finger_0_01/Sensor_0_proxy",
            name="sensor_0_view",
            reset_xform_properties=False)
        self.sensor_1 = ArticulationView(
            prim_paths_expr="/World/envs/.*/robot/gripper_mod_01/Simp_suction_array/Simp_suction_array/Finger_1_01/Sensor_1_proxy",
            name="sensor_1_view",
            reset_xform_properties=False)
        self.sensor_2 = ArticulationView(
            prim_paths_expr="/World/envs/.*/robot/gripper_mod_01/Simp_suction_array/Simp_suction_array/Finger_2_01/Sensor_2_proxy",
            name="sensor_2_view",
            reset_xform_properties=False)
        self.sensor_3 = ArticulationView(
            prim_paths_expr="/World/envs/.*/robot/gripper_mod_01/Simp_suction_array/Simp_suction_array/Finger_3_01/Sensor_3_proxy",
            name="sensor_3_view",
            reset_xform_properties=False)
        
        return self._robots, self._end_effector, self.wrist_2_link, self.sensor_0, self.sensor_1, self.sensor_2, self.sensor_3, self.body
