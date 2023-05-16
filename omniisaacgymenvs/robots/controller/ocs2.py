from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.objects.sphere import VisualSphere
from omni.isaac.core.utils.extensions import enable_extension

# isaac-universal_robots
from omni.isaac.universal_robots import UR10

# isaac-ocs2
enable_extension("omni.isaac.ocs2")
from omni.isaac.ocs2.end_effector_pose_tracking_mpc import EndEffectorPoseTrackingMpc

import numpy as np

class Controller_ocs2():
    def __init__(
        self
    ) -> None:
        self.mpc_interface = None

    def get_controller(self, robots, num_envs):
        """Sets the Franka control mode to "velocity" and tests the MPC."""
        # Add MPC
        config = {
            "urdf_path": "data/ur10/urdf/ur10.urdf",
            "lib_folder": "/tmp/ocs2/auto_generated/ur10",
            "mpc_config_path": "data/ur10/mpc/task.info",
        }

        self.mpc_interface = [EndEffectorPoseTrackingMpc(config["mpc_config_path"], config["lib_folder"], config["urdf_path"]) 
                              for i in range(num_envs)]
        # Receive the number of arm dimensions
        self.arm_num_dof = self.mpc_interface[0].state_dim
        # print info about MPC
        print(self.mpc_interface)

        # Set control mode - part of ArticulationView class
        robots.switch_control_mode("velocity")
        # self.disable_gravity()     

        # Now we are ready!
        print("[INFO]: Setup complete...")   

        # Define simulation stepping
        self.dt = 0.01
        self.sim_time = 0.0
        # Define goals for the arm
        self.ee_goal_index = 0
        self.ee_goals = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ]
        # Define a goal for the arm
        self.ee_goal_pose = np.array(self.ee_goals[self.ee_goal_index])
        # Obtain measurements
        arm_joint_pos = robots.get_joint_positions()[:self.arm_num_dof]
        ee_curr_pose = robots.end_effector.get_world_pose()
        ee_curr_pose = np.concatenate((ee_curr_pose[0], ee_curr_pose[1]), axis=0)
   
        # Define target trajectory
        [self.mpc_interface[i].set_target_trajectory(
            time_traj=[self.sim_time, self.sim_time + 2], state_traj=[ee_curr_pose, self.ee_goal_pose], input_traj=[None, None])
            for i in range(len(self.mpc_interface))]
        
        # Reset the MPC
        [self.mpc_interface[i].reset(self.sim_time, arm_joint_pos) for i in range(len(self.mpc_interface))]     

        return self.mpc_interface  

    def step_controller(self, robots):
        # obtain current measurements
        arm_joint_pos = robots.get_joint_positions()[:self.arm_num_dof]
        # compute arm's optimal control command
        arm_cmd = self.mpc_interface.advance(self.sim_time, arm_joint_pos)
        # print mpc cost
        # perform actions
        action = ArticulationAction(joint_velocities=arm_cmd, joint_indices=[range(self.arm_num_dof)])
        self.apply_action(action)
        # update sim-time
        self.sim_time += self.dt
        # obtain new measurements
        ee_curr_pose = self.end_effector.get_world_pose()
        ee_curr_pose = np.concatenate((ee_curr_pose[0], ee_curr_pose[1]), axis=0)
        # compute the waypoint error
        error = np.linalg.norm(ee_curr_pose[:3] - ee_goal_pose[:3])
        # get next waypoint
        if error < 0.014:
            # print goal state
            print(
                f"\tMPC cost: { self.mpc_interface.get_current_cost()}\n",
                f"\tCurrent EE state:\n"
                f"\t\tI_r_IE    : {ee_curr_pose[:3]} \n"
                f"\t\tq_IE      : {ee_curr_pose[3:]} \n"
                f"\tGoal EE state:\n"
                f"\t\tI_r_IE_des: {self.ee_goals[ee_goal_index][:3]} \n"
                f"\t\tq_IE_des  : {self.ee_goals[ee_goal_index][3:]} \n"
                "----------------------------------------------",
            )
            # next goal
            ee_goal_index += 1
            if ee_goal_index >= len(self.ee_goals):
                ee_goal_index = 0
            # Define a goal for the arm
            ee_goal_pose = np.array(self.ee_goals[ee_goal_index])
            # Update prims
            # self.goal_vis_prim.set_world_pose(ee_goal_pose[:3], ee_goal_pose[3:])
            # Define target trajectory
            self.mpc_interface.set_target_trajectory(
                time_traj=[self.sim_time, self.sim_time + 2], state_traj=[ee_curr_pose, ee_goal_pose], input_traj=[None, None]
            ) 