from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.robots.articulations.surface_gripper import SurfaceGripper
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class ROBOT:

    def __init__(self, num_envs, default_zero_env_path, _robot_positions,
                 _robot_rotations, _robot_dof_target, _sim_config,
                 usd_path, gripper_usd_path) -> None:

        self.num_envs = num_envs

        self.default_zero_env_path = default_zero_env_path
        self._robot_positions = _robot_positions
        self._robot_rotations = _robot_rotations
        self._robot_dof_target = _robot_dof_target
        self._robot_dof_target = _robot_dof_target
        self._sim_config = _sim_config

        self.usd_path = usd_path
        self.gripper_usd_path = gripper_usd_path

    def load_UR(self):

        self.robot = UR10(prim_path=self.default_zero_env_path + "/robot",
                          name="robot",
                          position=self._robot_positions,
                          orientation=self._robot_rotations,
                          attach_gripper=False,
                          usd_path=self.usd_path,
                          )  #self._task_cfg['sim']["URRobot"]['robot_path']

        self.robot.set_joint_positions(self._robot_dof_target)
        self.robot.set_joints_default_state(self._robot_dof_target)

        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(self.robot.prim_path),
            self._sim_config.parse_actor_config("robot"))

        return self.robot

    def add_gripper(self):
        assets_root_path = get_assets_root_path()
        
        gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"

        self.grippers = []

        for i in range(self.num_envs):

            add_reference_to_stage(
                usd_path=gripper_usd,
                prim_path=f"/World/envs/env_{i}/robot/ee_link")

            surface_gripper = SurfaceGripper(
                end_effector_prim_path=f"/World/envs/env_{i}/robot/ee_link",
                translate=0.1611,
                direction="y")
            surface_gripper.set_force_limit(value=8.0e1)
            surface_gripper.set_torque_limit(value=10.0e0)
            # surface_gripper.initialize(physics_sim_view=None, articulation_num_dofs=self.robot.num_dof)
            self.grippers.append(surface_gripper)

        return self.grippers
    
    def add_custom_gripper(self):
        
        gripper_usd = self.gripper_usd_path

        self.grippers = []

        for i in range(self.num_envs):

            add_reference_to_stage(
                usd_path=gripper_usd,
                prim_path=f"/World/envs/env_{i}/robot/ee_link")

            surface_gripper = SurfaceGripper(
                end_effector_prim_path=f"/World/envs/env_{i}/robot/ee_link",
                translate=0.1611,
                direction="y")
            surface_gripper.set_force_limit(value=8.0e1)
            surface_gripper.set_torque_limit(value=10.0e0)
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
        
        return self._robots, self._end_effector, self.wrist_2_link
