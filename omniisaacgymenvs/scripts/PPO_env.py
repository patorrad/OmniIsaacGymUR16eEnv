# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import torch
import hydra
import pdb
from omegaconf import DictConfig
from pathlib import Path
import torch.nn as nn

import argparse
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.wandb_util import setup_wandb,WandbCallback

from stable_baselines3.ppo import PPO
import pdb



@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    
    
    
    # init env
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)
#    
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

    # from stable_baselines3.common.torch_layers import CombinedExtractor,NatureCNN
    # policy = "MultiInputPolicy"

    # feature_extractor_class = CombinedExtractor

    # feature_extractor_kwargs = {
    #     "key": "tactile_image",
    #     "features_extractor_class": NatureCNN,  #NatureCNN,
    #     "state_key": "state",
    #     "cnn_output_dim":64
    # }
    # policy_kwargs = {
    #     "features_extractor_class": feature_extractor_class,
    #     "features_extractor_kwargs": feature_extractor_kwargs,
    #     "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
    #     "activation_fn": nn.ReLU,
    #     }
    
    # init wandb
    horizon = 200
    env_iter = training_config['iteraction'] * horizon * training_config['n_env_horizon']
    exp_name = "TofSensor_state"
    config = {
        'n_env_horizon': training_config['n_env_horizon'],
        
        'update_iteration': training_config['iteraction'],
        'total_step': env_iter,
    
    }
    
    wandb_run = setup_wandb(config,
                            exp_name,
                            tags=["PPO", "Tof"],
                            project="TofSensor")


    
    model = PPO(
        policy,
        env,
        n_epochs=training_config['n_epochs'],
        n_steps=(training_config['n_env_horizon'] //env._num_envs) * horizon,
        learning_rate=3e-4,
        batch_size=training_config['minibatch_size'],
        seed=cfg.seed,
        tensorboard_log=str(result_path/"log"),  #str(result_path / "log"),
        policy_kwargs=policy_kwargs,
        verbose=1
        )
    
    model.learn(total_timesteps=env_iter,
                callback=WandbCallback(model_save_freq=10,
                model_save_path=str(result_path / "model"),
                eval_freq=10,
                eval_env_fn=None
                ))
   

    env.close()

  

if __name__ == '__main__':
   
    parse_hydra_configs()