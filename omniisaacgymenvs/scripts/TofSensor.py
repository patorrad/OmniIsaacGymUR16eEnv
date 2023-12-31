
from omniisaacgymenvs.robots.articulations.ur10 import UR10


import numpy as np
import torch
import math
import trimesh


import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view
import omni.isaac.orbit.utils.kit as kit_utils

from omni.isaac.kit import SimulationApp

import argparse

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)



def get_robot():
       
        ur10_positions = torch.tensor([0.0, 0.0, 1.5])
        ur10_rotations = torch.tensor([0.0, 0.0, 1.0, 0.0])
        ur10_dof_target = torch.tensor([0.06, -2.5, 2.03, 0.58, 1.67, 1.74,0], device = "GPU") 
        ur10 = UR10(prim_path="/World/envs/.*/ur10"
                    ,usd_path="/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e/ur16e.usd", 
                    name="ur10", position=ur10_positions, orientation=ur10_rotations, attach_gripper=False)
        # applies articulation settings from the task configuration yaml file
        ur10.set_joint_positions(ur10_dof_target)
        ur10.set_joints_default_state(ur10_dof_target)

        return ur10
       


def main():

    # init sim
        
    #load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )
    
    robot = get_robot()
    sim_dt = sim.get_physics_dt()


    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=True)
            continue
        # reset
        if ep_step_count % 1000 == 0:
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            # reset command
            actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
            # reset gripper
            
            print("[INFO]: Resetting robots state...")
     
        # apply action to the robot
        robot.apply_action(actions)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()

    
        