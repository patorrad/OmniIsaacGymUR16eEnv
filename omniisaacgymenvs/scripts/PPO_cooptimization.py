import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path
import torch.nn as nn

import argparse
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.wandb_util import setup_wandb,WandbCallback
from omniisaacgymenvs.utils.sencond_agent import CustomCallback

from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.ppo import PPO
import os

os.environ["WANDB_MODE"] = "disabled"



@hydra.main(config_name="config_opt", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    # init env
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = not headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    from omniisaacgymenvs.utils.vec_env_base import VecEnvBase
   
    env = VecEnvBase(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
   
    task = initialize_task(cfg_dict, env)

    training_config = cfg["train"]["params"]["config"]
    
    import datetime
    current_time = datetime.datetime.now()
    result_path = Path("results/"+str(current_time.month)+str(current_time.day)+"/TofSensor2")
    result_path.mkdir(exist_ok=True, parents=True)

    policy = "MlpPolicy"
    policy_kwargs = {
            'activation_fn': nn.ReLU,
            "log_std_init": np.log(training_config['std'])
        }
    
    # init wandb
    horizon = 1000
    env_iter = 10000000
    exp_name = "TofSensor_state"
    config = {
        'n_env_horizon': training_config['n_env_horizon'],
        
        'update_iteration': training_config['iteraction'],
        'total_step': env_iter,
    
    }

    model = PPO(
        policy,
        env,
        n_epochs=training_config['n_epochs'],
        n_steps=horizon,
        learning_rate=3e-4,
        batch_size=training_config['minibatch_size'],
        seed=cfg.seed,
        tensorboard_log=str(result_path/"log"),  #str(result_path / "log"),
        verbose=1
        )

    model.learn(total_timesteps=env_iter)
   

    env.close()

  

if __name__ == '__main__':
   
    parse_hydra_configs()
