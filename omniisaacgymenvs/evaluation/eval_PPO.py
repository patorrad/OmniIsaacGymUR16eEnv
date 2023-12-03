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
from omniisaacgymenvs.utils.wandb_util import setup_wandb, WandbCallback

from stable_baselines3.ppo import PPO
import pdb


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    # init env
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = not headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    from omni.isaac.gym.vec_env import VecEnvBase

    env = VecEnvBase(headless=headless,
                     sim_device=cfg.device_id,
                     enable_livestream=cfg.enable_livestream,
                     enable_viewport=enable_viewport)
    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    task = initialize_task(cfg_dict, env)

    policy = PPO.load("results/122/TofSensor/model/model_100.zip", env, "cuda")

    while True:
        reward_sum = 0
        obs = env.reset()
        print('===============')
        print(
            "init dist deviation:",
            abs(env._task.dist_dev).sum().cpu().detach().item() /
            env._num_envs)
        print(
            "init angle deviation:",
            abs(env._task.angle_dev).sum().cpu().detach().item() /
            env._num_envs / torch.pi * 180)

        for i in range(180):

            action = policy.predict(observation=obs, deterministic=True)[0]

            obs, reward, done, info = env.step(action)
            reward_sum += reward.sum().cpu().detach().item() / env._num_envs

        print("reward:", reward_sum)
        print(
            "dist deviation:",
            abs(env._task.dist_dev).sum().cpu().detach().item() /
            env._num_envs)

        print(
            "angle deviation:",
            abs(env._task.angle_dev).sum().cpu().detach().item() /
            env._num_envs/ torch.pi * 180)


if __name__ == '__main__':

    parse_hydra_configs()
