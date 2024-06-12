# create isaac environment
from omniisaacgymenvs.utils.vec_env_base import VecEnvBase
from omniisaacgymenvs.utils.task_util import initialize_task
from omegaconf import DictConfig
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict

@hydra.main(config_name="config_opt", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    env = VecEnvBase(headless=False)

    cfg_dict = omegaconf_to_dict(cfg)
    cfg_dict['task']['env']['numEnvs'] = 2

    # create task and register task
    task = initialize_task(cfg_dict, env)

    # import stable baselines
    from stable_baselines3 import PPO

    # Run inference on the trained policy
    model = PPO.load("/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-06-01/21-52-38/results/61/TofSensor2/model/model_20.zip")
    env._world.reset()
    obs = env.reset()
    while env._simulation_app.is_running():
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)

    env.close()

if __name__ == '__main__':
   
    parse_hydra_configs()