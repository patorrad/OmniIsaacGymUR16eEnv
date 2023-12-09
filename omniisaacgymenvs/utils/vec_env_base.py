# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import os
import signal

import carb
import gymnasium as gym
from omni.isaac.kit import SimulationApp


class VecEnvBase(gym.Env):
    """This class provides a base interface for connecting RL policies with task implementations.
    APIs provided in this interface follow the interface in gym.Env.
    This class also provides utilities for initializing simulation apps, creating the World,
    and registering a task.
    """

    def __init__(
        self,
        headless: bool,
        sim_device: int = 0,
        enable_livestream: bool = False,
        enable_viewport: bool = False,
        launch_simulation_app: bool = True,
        experience: str = None,
    ) -> None:
        """Initializes RL and task parameters.

        Args:
            headless (bool): Whether to run training headless.
            sim_device (int): GPU device ID for running physics simulation. Defaults to 0.
            enable_livestream (bool): Whether to enable running with livestream.
            enable_viewport (bool): Whether to enable rendering in headless mode.
            launch_simulation_app (bool): Whether to launch the simulation app (required if launching from python). Defaults to True.
            experience (str): Path to the desired kit app file. Defaults to None, which will automatically choose the most suitable app file.
        """

        if launch_simulation_app:
            if experience is None:
                experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.kit'
                if headless:
                    if enable_livestream:
                        experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.kit'
                    elif enable_viewport:
                        experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.kit'
                    else:
                        experience = f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.gym.headless.kit'

            self._simulation_app = SimulationApp(
                {"headless": headless, "physics_gpu": sim_device}, experience=experience
            )

            if enable_livestream:
                from omni.isaac.core.utils.extensions import enable_extension

                self._simulation_app.set_setting("/app/livestream/enabled", True)
                self._simulation_app.set_setting("/app/window/drawMouse", True)
                self._simulation_app.set_setting("/app/livestream/proto", "ws")
                self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
                self._simulation_app.set_setting("/ngx/enabled", False)
                enable_extension("omni.kit.livestream.native")
                enable_extension("omni.services.streaming.manager")

            # handle ctrl+c event
            signal.signal(signal.SIGINT, self.signal_handler)

        self._render = not headless or enable_livestream or enable_viewport
        self.sim_frame_count = 0
        self._world = None

    def signal_handler(self, sig, frame):
        self.close()

    def set_task(self, task, backend="numpy", sim_params=None, init_sim=True, rendering_dt=1.0 / 60.0) -> None:
        """Creates a World object and adds Task to World.
            Initializes and registers task to the environment interface.
            Triggers task start-up.

        Args:
            task (RLTask): The task to register to the env.
            backend (str): Backend to use for task. Can be "numpy" or "torch". Defaults to "numpy".
            sim_params (dict): Simulation parameters for physics settings. Defaults to None.
            init_sim (Optional[bool]): Automatically starts simulation. Defaults to True.
            rendering_dt (Optional[float]): dt for rendering. Defaults to 1/60s.
        """

        from omni.isaac.core.world import World

        # parse device based on sim_param settings
        if sim_params and "sim_device" in sim_params:
            device = sim_params["sim_device"]
        else:
            device = "cpu"
            physics_device_id = carb.settings.get_settings().get_as_int("/physics/cudaDevice")
            gpu_id = 0 if physics_device_id < 0 else physics_device_id
            if sim_params and "use_gpu_pipeline" in sim_params:
                # GPU pipeline must use GPU simulation
                if sim_params["use_gpu_pipeline"]:
                    device = "cuda:" + str(gpu_id)
            elif sim_params and "use_gpu" in sim_params:
                if sim_params["use_gpu"]:
                    device = "cuda:" + str(gpu_id)

        self._world = World(
            stage_units_in_meters=1.0, rendering_dt=rendering_dt, backend=backend, sim_params=sim_params, device=device
        )
        self._world._current_tasks = dict()
        self._world.add_task(task)
        self._task = task
        self._num_envs = self._task.num_envs

        self.observation_space = self._task.observation_space
        self.action_space = self._task.action_space

        if sim_params and "enable_viewport" in sim_params:
            self._render = sim_params["enable_viewport"]

        if init_sim:
            self._world.reset()

    def update_task_params(self):
        self._num_envs = self._task.num_envs
        self.observation_space = self._task.observation_space
        self.action_space = self._task.action_space

    def render(self, mode="human") -> None:
        """Step the renderer.

        Args:
            mode (str): Select mode of rendering based on OpenAI environments.
        """

        if mode == "human":
            self._world.render()
        else:
            gym.Env.render(self, mode=mode)
        return

    def close(self) -> None:
        """Closes simulation."""

        if self._world:
            self._world.stop()

        # bypass USD warnings on stage close
        self._simulation_app.close(wait_for_replicator=False)
        return

    def seed(self, seed=-1):
        """Sets a seed. Pass in -1 for a random seed.

        Args:
            seed (int): Seed to set. Defaults to -1.
        Returns:
            seed (int): Seed that was set.
        """

        from omni.isaac.core.utils.torch.maths import set_seed

        return set_seed(seed)

    def step(self, actions):
        """Basic implementation for stepping simulation.
            Can be overriden by inherited Env classes
            to satisfy requirements of specific RL libraries. This method passes actions to task
            for processing, steps simulation, and computes observations, rewards, and resets.

        Args:
            actions (Union[numpy.ndarray, torch.Tensor]): Actions buffer from policy.
        Returns:
            observations(Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
            rewards(Union[numpy.ndarray, torch.Tensor]): Buffer of rewards data.
            dones(Union[numpy.ndarray, torch.Tensor]): Buffer of resets/dones data.
            info(dict): Dictionary of extras data.
        """

        if not self._world.is_playing():
            self.close()

        self._task.pre_physics_step(actions)
        self._world.step(render=self._render)

        self.sim_frame_count += 1

        if not self._world.is_playing():
            self.close()

        observations = self._task.get_observations()
        rewards = self._task.calculate_metrics()
        terminated = self._task.is_done()
        truncated = self._task.is_done() * 0
        info = {}

        return observations, rewards, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the task and updates observations.

        Args:
            seed (Optional[int]): Seed.
            options (Optional[dict]): Options as used in gymnasium.
        Returns:
            observations(Union[numpy.ndarray, torch.Tensor]): Buffer of observation data.
            info(dict): Dictionary of extras data.
        """
        if seed is not None:
            seed = self.seed(seed)
            super().reset(seed=seed)

        self._task.reset()
        self._world.step(render=self._render)
        observations = self._task.get_observations()

        return observations, {}

    @property
    def num_envs(self):
        """Retrieves number of environments.

        Returns:
            num_envs(int): Number of environments.
        """
        return self._num_envs