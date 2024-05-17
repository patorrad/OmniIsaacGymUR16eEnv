import logging
import os
from typing import Optional, Callable, List


import wandb
from stable_baselines3.common.callbacks import BaseCallback
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry

import numpy as np
import torch
logger = logging.getLogger(__name__)



class WandbCallback(BaseCallback):
    """Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
    """

    def __init__(
            self,
            verbose: int = 0,
            model_save_path: str = None,
            model_save_freq: int = 100,
            eval_freq: Optional[int] = None,
            eval_env_fn: Optional[Callable] = None,
            eval_cam_names: Optional[List[str]] = None,
            viz_point_cloud=False,
            gradient_save_freq: int = 0,
    ):
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbCallback()")
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path

        self.eval_freq = eval_freq
        self.eval_env_fn = eval_env_fn
        self.eval_cam_names = eval_cam_names
        self.viz_point_cloud = viz_point_cloud

        self.gradient_save_freq = gradient_save_freq
        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
        else:
            assert (
                    self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

        self.roll_out = 0

        self.total_reward = 0

    def _init_callback(self) -> None:
        d = {}
        if "algo" not in d:
            d["algo"] = type(self.model).__name__
        for key in self.model.__dict__:
            if key in wandb.config:
                continue
            if type(self.model.__dict__[key]) in [float, int, str]:
                d[key] = self.model.__dict__[key]
            else:
                d[key] = str(self.model.__dict__[key])
        if self.gradient_save_freq > 0:
            wandb.watch(self.model.policy, log_freq=self.gradient_save_freq, log="all")
        wandb.config.setdefaults(d)

    def _on_rollout_end(self) -> None:
        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.roll_out % self.model_save_freq == 0:
                    self.save_model()

        if self.eval_freq is not None and self.eval_env_fn is not None:
            self.eval_env_fn._task.post_reset()
            self.eval_env_fn._task.curr_env_step *= 0
            
            for i in range(10):
                self.eval_env_fn._task._env._world.step(render=True)

            if self.roll_out % self.eval_freq == 0:
                env = self.eval_env_fn
                reward_sum = 0
                obs = env.reset()
                image_list = []
                for i in range(200):
                    
                    action = self.model.policy.predict(observation=obs, deterministic=True)[0]
                    obs, reward, done, _ = env.step(action)
                    rgb_image = self.eval_env_fn._task.render_image()
                    

                    reward_sum += reward.sum().cpu().detach().item()/env._num_envs
                    image_list.append(np.array(rgb_image))

             
                
                video_array = (np.stack(image_list, axis=0)).astype(np.uint8)[:,:,:,:3]
                video_array = np.transpose(video_array, (0, 3, 1, 2))
                wandb.log(
                    {"test_view": wandb.Video(video_array, fps=20, format="mp4",
                                                        caption=f"Reward: {reward_sum:.2f}")})
        self.roll_out += 1

        self.logger.record('reward', self.total_reward)
        self.total_reward = 0

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        path = os.path.join(self.model_save_path, f"model_{self.roll_out}")
        self.model.save(path)
        wandb.save(path, base_path=self.model_save_path)
        if self.verbose > 1:
            logger.info("Saving model checkpoint to " + path)

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        # self.logger.record('reward', torch.mean(self.locals['rewards']))
        # import pdb; pdb.set_trace()
        self.total_reward += torch.mean(self.locals['rewards'])

        return True



def setup_wandb(parser_config,
                exp_name,
                tags=None,
                project="TofSensor"):
    wandb.login()
    run = wandb.init(
        project=project,
        name=exp_name,
        config=parser_config,
        monitor_gym=False,
        save_code=True,  # optional
        tags=tags,
        )
    return run