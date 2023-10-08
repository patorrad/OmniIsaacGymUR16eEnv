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


from omniisaacgymenvs.robots.articulations.ant import Ant
from omniisaacgymenvs.tasks.shared.locomotion import LocomotionTask
from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path

from pxr import PhysxSchema

import numpy as np
import torch
import math


from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.urdf import _urdf
import omni.kit

from typing import Optional
import numpy as np
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.manipulators.grippers.surface_gripper import SurfaceGripper
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.nucleus import get_assets_root_path
import carb



class UR10(Robot):
    """[summary]

        Args:
            prim_path (str): [description]
            name (str, optional): [description]. Defaults to "ur10_robot".
            usd_path (Optional[str], optional): [description]. Defaults to None.
            position (Optional[np.ndarray], optional): [description]. Defaults to None.
            orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
            end_effector_prim_name (Optional[str], optional): [description]. Defaults to None.
            attach_gripper (bool, optional): [description]. Defaults to False.
            gripper_usd (Optional[str], optional): [description]. Defaults to "default".

        Raises:
            NotImplementedError: [description]
        """

    def __init__(
        self,
        prim_path: str,
        name: str = "ur10_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        attach_gripper: bool = False,
        gripper_usd: Optional[str] = "default",
    ) -> None:
        prim = get_prim_at_path(prim_path)
        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        if not prim.IsValid():
            add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/ee_link"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        else:
            # TODO: change this
            if self._end_effector_prim_name is None:
                self._end_effector_prim_path = prim_path + "/ee_link"
            else:
                self._end_effector_prim_path = prim_path + "/" + end_effector_prim_name
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        self._gripper_usd = gripper_usd
        if attach_gripper:
            if gripper_usd == "default":
                assets_root_path = get_assets_root_path()
                if assets_root_path is None:
                    carb.log_error("Could not find Isaac Sim assets folder")
                    return
                gripper_usd = assets_root_path + "/Isaac/Robots/UR10/Props/short_gripper.usd"
                add_reference_to_stage(usd_path=gripper_usd, prim_path=self._end_effector_prim_path)
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="x"
                )
            elif gripper_usd is None:
                carb.log_warn("Not adding a gripper usd, the gripper already exists in the ur10 asset")
                self._gripper = SurfaceGripper(
                    end_effector_prim_path=self._end_effector_prim_path, translate=0.1611, direction="x"
                )
            else:
                raise NotImplementedError
        self._attach_gripper = attach_gripper
        return

    @property
    def attach_gripper(self) -> bool:
        """[summary]

        Returns:
            bool: [description]
        """
        return self._attach_gripper

    @property
    def end_effector(self) -> RigidPrim:
        """[summary]

        Returns:
            RigidPrim: [description]
        """
        return self._end_effector

    @property
    def gripper(self) -> SurfaceGripper:
        """[summary]

        Returns:
            SurfaceGripper: [description]
        """
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        """[summary]
        """
        super().initialize(physics_sim_view)
        if self._attach_gripper:
            self._gripper.initialize(physics_sim_view=physics_sim_view, articulation_num_dofs=self.num_dof)
        self._end_effector = RigidPrim(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self.disable_gravity()
        self._end_effector.initialize(physics_sim_view)
        return

    def post_reset(self) -> None:
        """[summary]
        """
        Robot.post_reset(self)
        self._end_effector.post_reset()
        self._gripper.post_reset()
        return



class AntLocomotionTask(LocomotionTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
    
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._num_observations = 60
        self._num_actions = 8
        self._ant_positions = torch.tensor([0, 0, 0.5])

        LocomotionTask.__init__(self, name=name, env=env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_ant()
        urdf_interface = _urdf.acquire_urdf_interface()
        # # Set the settings in the import config
        # import_config = _urdf.ImportConfig()
        # import_config.merge_fixed_joints = False
        # import_config.convex_decomp = False
        # import_config.import_inertia_tensor = True
        # import_config.fix_base = True
        # import_config.make_default_prim = True
        # import_config.self_collision = False
        # import_config.create_physics_scene = True
        # import_config.import_inertia_tensor = False
        # import_config.default_drive_strength = 1047.19751
        # import_config.default_position_drive_damping = 52.35988
        # import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        # import_config.distance_scale = 1
        # import_config.density = 0.0
        # # Get the urdf file path
        # extension_path = get_extension_path_from_name("omni.isaac.urdf")
        # root_path = "/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e"
        # file_name = "ur16e.urdf"
        # # root_path = extension_path + "/data/urdf/robots/franka_description/robots"
        # # file_name = "panda_arm_hand.urdf"
        # # Finally import the robot
        # result, prim_path = omni.kit.commands.execute( "URDFParseAndImportFile", urdf_path="{}/{}".format(root_path, file_name),
        #                                               import_config=import_config,)

        self._ur10_positions = torch.tensor([0.0, 0.0, 1.5])
        self._ur10_rotations = torch.tensor([0.0, 0.0, 1.0, 0.0])
        self._ur10_dof_target = torch.tensor([0.06, -2.5, 2.03, 0.58, 1.67, 1.74], device = self._device) 
        self._ur10_dof_targets = self._ur10_dof_target.repeat(self._num_envs, 1) 
        ur10 = UR10(prim_path="/World/Fancy_Robot"
                    ,usd_path="/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e/ur16e.usd", 
                    name="ur10", position=self._ur10_positions, orientation=self._ur10_rotations, attach_gripper=False)
        # self.ur10 = UR10(prim_path=self.default_zero_env_path + "/ur10", name="ur10", position=self._ur10_positions, orientation=self._ur10_rotations, attach_gripper=True)

        RLTask.set_up_scene(self, scene)

     
        self._ants = ArticulationView(prim_paths_expr="/World/envs/.*/Ant/torso", name="ant_view", reset_xform_properties=False)
        scene.add(self._ants)
        return

    def get_ant(self):
        ant = Ant(prim_path=self.default_zero_env_path + "/Ant", name="Ant", translation=self._ant_positions)
        self._sim_config.apply_articulation_settings("Ant", get_prim_at_path(ant.prim_path), self._sim_config.parse_actor_config("Ant"))

    def get_robot(self):
        return self._ants

    def post_reset(self):
        self.joint_gears = torch.tensor([15, 15, 15, 15, 15, 15, 15, 15], dtype=torch.float32, device=self._device)
        dof_limits = self._ants.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self._device)

        LocomotionTask.post_reset(self)

    def get_dof_at_limit_cost(self):
        return get_dof_at_limit_cost(self.obs_buf, self._ants.num_dof)


@torch.jit.script
def get_dof_at_limit_cost(obs_buf, num_dof):
    # type: (Tensor, int) -> Tensor
    return torch.sum(obs_buf[:, 12:12+num_dof] > 0.99, dim=-1)