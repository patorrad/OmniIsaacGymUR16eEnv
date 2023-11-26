# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import random

import carb
import numpy as np
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.universal_robots import UR10


class BinFilling(BaseTask):
    """Task using UR10 robot to fill a bin with screws and showcase the surface gripper torque/ force limits.

    Args:
        name (str, optional): Task name identifier. Should be unique if added to the World. Defaults to "bin_filling".
    """

    def __init__(self, name: str = "bin_filling") -> None:
        BaseTask.__init__(self, name=name, offset=None)
        self._ur10_robot = None
        self._packing_bin = None
        self._assets_root_path = get_assets_root_path()
        if self._assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder")
            return
        self._ur10_asset_path = self._assets_root_path + "/Isaac/Samples/Leonardo/Stage/ur10_bin_filling.usd"
        self._screw_asset_paths = [
            self._assets_root_path + "/Isaac/Props/Flip_Stack/large_corner_bracket_physics.usd",
            self._assets_root_path + "/Isaac/Props/Flip_Stack/screw_95_physics.usd",
            self._assets_root_path + "/Isaac/Props/Flip_Stack/screw_99_physics.usd",
            self._assets_root_path + "/Isaac/Props/Flip_Stack/small_corner_bracket_physics.usd",
            self._assets_root_path + "/Isaac/Props/Flip_Stack/t_connector_physics.usd",
        ]
        self._screws = []
        self._max_screws = 100
        self._screws_to_add = 0
        self._pipe_position = np.array([0, 0.85, 1.2]) / get_stage_units()
        self._target_position = np.array([0, 0.90, -0.44]) / get_stage_units()
        self._bin_initial_position = np.array([0.35, 0.15, -0.40]) / get_stage_units()
        self._bin_size = np.array([0.25, 0.35, 0.20]) / get_stage_units()
        return

    def get_current_num_of_screws_to_add(self) -> int:
        """
        Returns:
            int: Number of screws left to drop from the pipe
        """
        return self._screws_to_add

    def set_up_scene(self, scene: Scene) -> None:
        """Loads the stage USD and adds the robot and packing bin to the World's scene.

        Args:
            scene (Scene): The world's scene.
        """
        super().set_up_scene(scene)
        add_reference_to_stage(usd_path=self._ur10_asset_path, prim_path="/World/Scene")
        self._ur10_robot = scene.add(
            UR10(prim_path="/World/Scene/ur10", name="my_ur10", gripper_usd=None, attach_gripper=True)
        )
        self._ur10_robot.gripper.set_translate(value=0.162)
        self._ur10_robot.set_joints_default_state(
            positions=np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0])
        )
        self._ur10_robot.gripper.set_direction(value="x")
        self._ur10_robot.gripper.set_force_limit(value=8.0e1)
        self._ur10_robot.gripper.set_torque_limit(value=10.0e0)
        self._packing_bin = scene.add(
            RigidPrim(
                prim_path="/World/Scene/bin",
                name="packing_bin",
                position=self._bin_initial_position,
                orientation=euler_angles_to_quat(np.array([0, 0, np.pi / 2])),
            )
        )
        return

    def get_observations(self) -> dict:
        """Returns current observations from the task needed for the behavioral layer at each time step.

           Observations:
            - packing_bin
                - position
                - orientation
                - target_position
                - size
            - my_ur10:
                - joint_positions
                - end_effector_position
                - end_effector_orientation

        Returns:
            dict: [description]
        """
        joints_state = self._ur10_robot.get_joints_state()
        bin_position, bin_orientation = self._packing_bin.get_world_pose()
        end_effector_position, end_effector_orientation = self._ur10_robot.end_effector.get_world_pose()
        # TODO: change values with USD
        return {
            "packing_bin": {
                "position": bin_position,
                "orientation": bin_orientation,
                "target_position": self._target_position,
                "size": self._bin_size,
            },
            "my_ur10": {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
                "end_effector_orientation": end_effector_orientation,
            },
        }

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """Executed before the physics step.

        Args:
            time_step_index (int): Current time step index
            simulation_time (float): Current simulation time.
        """
        BaseTask.pre_step(self, time_step_index=time_step_index, simulation_time=simulation_time)
        self._ur10_robot.gripper.update()
        if self._screws_to_add > 0 and len(self._screws) < self._max_screws and time_step_index % 30 == 0:
            self._add_screw()
        return

    def post_reset(self) -> None:
        """Executed after reseting the scene"""
        self._screws_to_add = 0
        self._screws = []
        return

    def add_screws(self, screws_number: int = 10) -> None:
        """Adds number of screws to be added by the pipe

        Args:
            screws_number (int, optional): number of screws to be added by the pipe. Defaults to 10.
        """
        self._screws_to_add += screws_number
        return

    def _add_screw(self):
        asset_path = self._screw_asset_paths[random.randint(0, len(self._screw_asset_paths) - 1)]
        prim_path = "/World/objects/object_{}".format(len(self._screws))
        add_reference_to_stage(usd_path=asset_path, prim_path=prim_path)
        self._screws.append(
            self.scene.add(
                XFormPrim(prim_path=prim_path, name="screw_{}".format(len(self._screws)), position=self._pipe_position)
            )
        )
        self._screws_to_add -= 1
        return

    def cleanup(self) -> None:
        """Removed the added screws when resetting."""
        for i in range(len(self._screws)):
            self.scene.remove_object(self._screws[i].name)
        self._screws = []
        return

    def get_params(self) -> dict:
        """Task parameters are
            - bin_name
            - robot_name

        Returns:
            dict: defined parameters of the task.
        """
        params_representation = dict()
        params_representation["bin_name"] = {"value": self._packing_bin.name, "modifiable": False}
        params_representation["robot_name"] = {"value": self._ur10_robot.name, "modifiable": False}
        return params_representation
