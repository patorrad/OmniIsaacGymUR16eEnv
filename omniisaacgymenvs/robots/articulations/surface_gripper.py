# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from omni.isaac.dynamic_control import _dynamic_control
from omni.isaac.manipulators.grippers.gripper import Gripper
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper
from omni.isaac.surface_gripper._surface_gripper import Surface_Gripper_Properties
from omni.isaac.core.utils.types import ArticulationAction
import numpy as np
import carb
import omni.kit.app


class SurfaceGripper(Gripper):
    """Provides high level functions to set/ get properties and actions of a surface gripper 
        (a suction cup for example).

        Args:
            end_effector_prim_path (str): prim path of the Prim that corresponds to the gripper root/ end effector.
            translate (float, optional): _description_. Defaults to 0.
            direction (str, optional): _description_. Defaults to "x".
            grip_threshold (float, optional): _description_. Defaults to 0.01.
            force_limit (float, optional): _description_. Defaults to 1.0e6.
            torque_limit (float, optional): _description_. Defaults to 1.0e4.
            bend_angle (float, optional): _description_. Defaults to np.pi/24.
            kp (float, optional): _description_. Defaults to 1.0e2.
            kd (float, optional): _description_. Defaults to 1.0e2.
            disable_gravity (bool, optional): _description_. Defaults to True.
        """

    def __init__(
        self,
        end_effector_prim_path: str,
        translate: float = 0,
        direction: str = "x",
        grip_threshold: float = 0.01,
        force_limit: float = 1.0e6,
        torque_limit: float = 1.0e4,
        bend_angle: float = np.pi / 24,
        kp: float = 1.0e2,
        kd: float = 1.0e2,
        disable_gravity: bool = True,
    ) -> None:
        Gripper.__init__(self, end_effector_prim_path=end_effector_prim_path)
        self._dc_interface = _dynamic_control.acquire_dynamic_control_interface()
        self._translate = translate
        self._direction = direction
        self._grip_threshold = grip_threshold
        self._force_limit = force_limit
        self._torque_limit = torque_limit
        self._bend_angle = bend_angle
        self._kp = kp
        self._kd = kd
        self._disable_gravity = disable_gravity
        self._virtual_gripper = None
        self._articulation_num_dofs = None


        
        return

    def initialize(
        self, physics_sim_view: omni.physics.tensors.SimulationView = None, articulation_num_dofs: int = None
    ) -> None:
        """Create a physics simulation view if not passed and creates a rigid prim view using physX tensor api.
            This needs to be called after each hard reset (i.e stop + play on the timeline) before interacting with any
            of the functions of this class.

        Args:
            physics_sim_view (omni.physics.tensors.SimulationView, optional): current physics simulation view. Defaults to None
            articulation_num_dofs (int, optional): num of dofs of the Articulation. Defaults to None.
        """
        Gripper.initialize(self, physics_sim_view=physics_sim_view)
        self._articulation_num_dofs = articulation_num_dofs
        virtual_gripper_props = Surface_Gripper_Properties()
        virtual_gripper_props.parentPath = self._end_effector_prim_path
        virtual_gripper_props.d6JointPath = virtual_gripper_props.parentPath + "/d6FixedJoint"
        virtual_gripper_props.gripThreshold = self._grip_threshold
        virtual_gripper_props.forceLimit = self._force_limit
        virtual_gripper_props.torqueLimit = self._torque_limit
        virtual_gripper_props.bendAngle = self._bend_angle
        virtual_gripper_props.stiffness = self._kp
        virtual_gripper_props.damping = self._kd
        virtual_gripper_props.disableGravity = self._disable_gravity
        tr = _dynamic_control.Transform()
        if self._direction == "x":
            tr.p.x = self._translate
        elif self._direction == "y":
            tr.p.y = self._translate
        elif self._direction == "z":
            tr.p.z = self._translate
        
        else:
            carb.log_error("Direction specified for the surface gripper doesn't exist")
        
       
        virtual_gripper_props.offset = tr
        virtual_gripper = Surface_Gripper(self._dc_interface)
        
    
        virtual_gripper.initialize(virtual_gripper_props)
        self._virtual_gripper = virtual_gripper
        if self._default_state is None:
            self._default_state = not self.is_closed()
        
        return

    def close(self) -> None:
        """Applies actions to the articulation that closes the gripper (ex: to hold an object).
        """
        if not self.is_closed():
            self._virtual_gripper.close()
        if not self.is_closed():
            carb.log_warn("gripper didn't close successfully")
        return

    def open(self) -> None:
        """Applies actions to the articulation that opens the gripper (ex: to release an object held).
        """
        result = self._virtual_gripper.open()
        if not result:
            carb.log_warn("gripper didn't open successfully")

        return

    def update(self) -> None:
        self._virtual_gripper.update()
        return

    def is_closed(self) -> bool:
        return self._virtual_gripper.is_closed()

    def set_translate(self, value: float) -> None:
        self._translate = value
        return

    def set_direction(self, value: float) -> None:
        self._direction = value
        return

    def set_force_limit(self, value: float) -> None:
        self._force_limit = value
        return

    def set_torque_limit(self, value: float) -> None:
        self._torque_limit = value
        return

    def set_default_state(self, opened: bool):
        """Sets the default state of the gripper

        Args:
            opened (bool): True if the surface gripper should start in an opened state. False otherwise.
        """
        self._default_state = opened
        return

    def get_default_state(self) -> dict:
        """Gets the default state of the gripper

        Returns:
            dict: key is "opened" and value would be true if the surface gripper should start in an opened state. False otherwise.
        """
        return {"opened": self._default_state}

    def post_reset(self):
        Gripper.post_reset(self)
        if self._default_state:  # means opened is true
            self.open()
        else:
            self.close()
        return

    def forward(self, action: str) -> ArticulationAction:
        """calculates the ArticulationAction for all of the articulation joints that corresponds to "open"
           or "close" actions.

        Args:
            action (str): "open" or "close" as an abstract action.

        Raises:
            Exception: _description_

        Returns:
            ArticulationAction: articulation action to be passed to the articulation itself
                                (includes all joints of the articulation).
        """
        if self._articulation_num_dofs is None:
            raise Exception(
                "Num of dofs of the articulation needs to be passed to initialize in order to use this method"
            )
        if action == "open":
            self.open()
        elif action == "close":
            self.close()
        else:
            raise Exception("action {} is not defined for SurfaceGripper".format(action))
        
        return ArticulationAction(joint_positions=[None] * self._articulation_num_dofs)