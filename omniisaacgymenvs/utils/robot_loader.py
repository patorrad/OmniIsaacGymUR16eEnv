from omniisaacgymenvs.robots.articulations.ur10 import UR10
from omni.isaac.core.utils.prims import get_prim_at_path


class ROBOT:

    def __init__(self, default_zero_env_path, _robot_positions,
                 _robot_rotations, _robot_dof_target, _sim_config,usd_path) -> None:
        
        self.default_zero_env_path = default_zero_env_path
        self._robot_positions = _robot_positions
        self._robot_rotations = _robot_rotations
        self._robot_dof_target = _robot_dof_target
        self._robot_dof_target = _robot_dof_target
        self._sim_config = _sim_config
        
        self.usd_path = usd_path

    def load_UR(self):

        self.robot = UR10(
            prim_path=self.default_zero_env_path + "/robot",
            name="robot",
            position=self._robot_positions,
            orientation=self._robot_rotations,
            attach_gripper=False,
            usd_path=self.usd_path)  #self._task_cfg['sim']["URRobot"]['robot_path']

        self.robot.set_joint_positions(self._robot_dof_target)
        self.robot.set_joints_default_state(self._robot_dof_target)

        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(self.robot.prim_path),
            self._sim_config.parse_actor_config("robot"))

        return self.robot
