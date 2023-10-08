
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.urdf import _urdf
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.franka.tasks import FollowTarget
import omni.kit.commands
import omni.usd
from omniisaacgymenvs.tasks.base.rl_task import RLTask
import numpy as np
import torch
import math
import trimesh
class TofSensorTask(RLTask):
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

        self._device = self._cfg["rl_device"]
       

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0.0, 0.0, 2.0])
        self._ur10_positions = torch.tensor([0.0, 0.0, 1.5])
        self._ur10_rotations = torch.tensor([0.0, 0.0, 1.0, 0.0])
        self._ur10_dof_target = torch.tensor([0.06, -2.5, 2.03, 0.58, 1.67, 1.74], device = self._device) 
        self._ur10_dof_targets = self._ur10_dof_target.repeat(self._num_envs, 1) 
        # self._target_object_positions = torch.tensor([-0.4, 0.0, 0.9])
        self._target_object_positions = [torch.tensor([-0.6, 0.0, 0.9]), torch.tensor([-0.6, -0.25, 0.9])]
        # self.debug_draw = _debug_draw.acquire_debug_draw_interface()

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        self._end_effector_link = "ee_link"

        self._step = 0

        RLTask.__init__(self, name, env)
        self.setup_scene()
        return

    
    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        # Acquire the URDF extension interface
        urdf_interface = _urdf.acquire_urdf_interface()
        # Set the settings in the import config
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.self_collision = False
        import_config.create_physics_scene = True
        import_config.import_inertia_tensor = False
        import_config.default_drive_strength = 1047.19751
        import_config.default_position_drive_damping = 52.35988
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        import_config.distance_scale = 1
        import_config.density = 0.0
        # Get the urdf file path
        extension_path = get_extension_path_from_name("omni.isaac.urdf")
        root_path = "/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e"
        file_name = "ur16e.urdf"
        # root_path = extension_path + "/data/urdf/robots/franka_description/robots"
        # file_name = "panda_arm_hand.urdf"
        # Finally import the robot
        result, prim_path = omni.kit.commands.execute( "URDFParseAndImportFile", urdf_path="{}/{}".format(root_path, file_name),
                                                      import_config=import_config,)
       
        # Optionally, you could also provide a `dest_path` parameter stage path to URDFParseAndImportFile,
        # which would import the robot on a new stage, in which case you'd need to add it to current stage as a reference:
        #   dest_path = "/path/to/dest.usd
        #   result, prim_path = omni.kit.commands.execute( "URDFParseAndImportFile", urdf_path="{}/{}".format(root_path, file_name),
        # import_config=import_config,dest_path = dest_path)
        #   prim_path = omni.usd.get_stage_next_free_path(
        #       self.world.scene.stage, str(current_stage.GetDefaultPrim().GetPath()) + prim_path, False
        #   )
        #   robot_prim = self.world.scene.stage.OverridePrim(prim_path)
        #   robot_prim.GetReferences().AddReference(dest_path)
        # This is required for robot assets that contain texture, otherwise texture won't be loaded.

        # Now lets use it with one of the tasks defined under omni.isaac.franka
        # Similar to what was covered in Tutorial 6 Adding a Manipulator in the Required Tutorials
        return 
    #     my_task = FollowTarget(name="follow_target_task", franka_prim_path=prim_path,
    #                            franka_robot_name="fancy_franka", target_name="target")
    #     world.add_task(my_task)

    #     return

    # async def setup_post_load(self):
    #     self._world = self.get_world()
    #     self._franka = self._world.scene.get_object("fancy_franka")
    #     self._controller = RMPFlowController(name="target_follower_controller", robot_articulation=self._franka)
    #     self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
    #     await self._world.play_async()
    #     return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # world = self.get_world()
        # observations = world.get_observations()
        # actions = self._controller.forward(
        #     target_end_effector_position=observations["target"]["position"],
        #     target_end_effector_orientation=observations["target"]["orientation"],
        # )
        # self._franka.apply_action(actions)
        return

