# Things to change locally at the moment
Update all usd file sources to locals (ie config.yaml or CustomGripper.yaml). Ensure references in usd are correct. Gripper usd reference still needs to be localed to repo (ie 'AURMR - Paolo_35'). 

# Dependencies
omni_python -m pip install pandas cprint trimesh[easy] 
omni_python -m pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# How to run branch custom_gripper

omni_python PPO_env_custom_gripper.py task=CustomGripper