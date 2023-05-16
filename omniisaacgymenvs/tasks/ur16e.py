from omni.isaac.gym.vec_env import VecEnvBase
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omni.isaac.core.articulations import ArticulationView
import torch

from omni.isaac.universal_robots import UR10
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.stage import get_current_stage

env = VecEnvBase(headless=False)

class UR16e(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

        self.distX_offset = 0.04
        self.dt = 1/60.

        self._num_observations = 23
        self._num_actions = 9

        self._cartpole_position = [0.0, 0.0, 2.0]

        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        # retrieve file path for the Cartpole USD file
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/UR10/ur10.usd"
        # add the Cartpole USD to our stage
        create_prim(prim_path="/World/ur10", prim_type="Xform", position=self._cartpole_position)
        add_reference_to_stage(usd_path, "/World/ur10")
        super().set_up_scene(scene) # pass scene to parent class - this method in RLTask also uses GridCloner to clone the robot and adds a ground plane if desired
        self._my_robots = ArticulationView(prim_paths_expr="/World/ur10", name="cartpole_view") # create a view of robots
        scene.add(self._my_robots) # add view to scene for initialization

        self.init_data()
        return 
    
    def init_data(self) -> None:

        # self.franka_default_dof_pos = torch.tensor(
        #     [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        # )

        # self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)
        pass

    def post_reset(self):
        # implement any logic required for simulation on-start here
        pass

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        # self.perform_reset()
        # self.apply_action(actions)
        pass

    def get_observations(self) -> dict:
        # implement logic to retrieve observation states
        # self.obs_buf = self.compute_observations()
        pass

    def calculate_metrics(self) -> None:
        # implement logic to compute rewards
        # self.rew_buf = self.compute_rewards()
        pass

    def is_done(self) -> None:
        # implement logic to update dones/reset buffer
        # self.reset_buf = self.compute_resets()
        pass