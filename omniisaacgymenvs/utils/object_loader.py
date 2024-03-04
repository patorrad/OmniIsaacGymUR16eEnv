import torch
import numpy as np

from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import FixedCuboid
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.prims import RigidPrimView

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf
from pxr import UsdPhysics
from omni.physx.scripts import utils
from omni.physx import acquire_physx_interface

import omni
import carb

from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, is_prim_path_valid

import os


class Object:

    def __init__(self, _sim_config, num_envs, device,
                 default_zero_env_path) -> None:
        self._sim_config = _sim_config
        self.num_envs = num_envs
        self.device = device
        self.default_zero_env_path = default_zero_env_path

    def load_cube(self, scale,num_object=1):
        self.scale_size = torch.as_tensor(scale).repeat(self.num_envs,
                                                        1).to(self.device)
        for i in range(self.num_envs):
            
            for j in range(1,num_object+1):

                target_object_1 = DynamicCuboid(
                    prim_path=f"/World/envs/env_{i}/manipulated_object_{j}",
                    name=f"manipulated_object_{j}",
                    position=[0, 0, 2.02],
                    # size=0.2,
                    scale=np.array(
                        scale),  #self._task_cfg["sim"]["Object"]["scale"]
                    color=torch.tensor([0, 169 / 255, 1]))

                self._sim_config.apply_articulation_settings(
                    "table", get_prim_at_path(target_object_1.prim_path),
                    self._sim_config.parse_actor_config(f"manipulated_object_{j}"))
        return self.scale_size

    def load_manipulated_object(self):

        object_dir = self.current_directory + "/omniisaacgymenvs/assests/objects/shapenet_nomat/" + self._task_cfg[
            'sim']["Object"]["category"]
        object_list = os.listdir(object_dir)

        for i in range(self.num_envs):
            object_name = object_list[i]  # np.random.choice(object_list)

            object_path = object_dir + "/" + object_name + "/model_normalized_nomat.usd"
            self.load_object(usd_path=object_path, env_index=i, object_index=1)
            self.object_prim_path.append(object_path)

            # object_path = object_dir + "/" + np.random.choice(object_list) + "/model_normalized_nomat.usd"
            # self.load_object(usd_path=object_path,env_index=i,object_index=2)

    def load_table(self, position, orientation,scale):
        table_translation = np.array(
            position)  #  self._task_cfg['sim']["Table"]["position"]
        table_orientation = np.array(
            orientation)  # self._task_cfg['sim']["Table"]["quaternion"]

        table = FixedCuboid(
            prim_path=self.default_zero_env_path + "/table",
            name="table",
            translation=table_translation,
            orientation=table_orientation,
            scale=scale,
            size=1.0,
            color=np.array([1, 197 / 255, 197 / 255]),
        )
        import omni.isaac.core.utils.nucleus as nucleus_utils
        table_usd_path = f"{nucleus_utils.get_assets_root_path()}/NVIDIA/Assets/ArchVis/Residential/Furniture/Tables/Whittershins.usd"
        # fix table base
        # table = prim_utils.create_prim(self.default_zero_env_path + "/table",
        #                                usd_path=table_usd_path,
        #                                translation=table_translation,
        #                                scale=(0.005, 0.005, 0.0202))
        table_prim = get_prim_at_path(self.default_zero_env_path + "/table")

        self._sim_config.apply_rigid_body_settings(
            "table",
            table_prim,
            self._sim_config.parse_actor_config("table"),
            is_articulation=False)

    def load_object(self,
                    usd_path,
                    env_index,
                    object_index,
                    translaton=[-0.69, 0.1, 1.3],
                    orientation=[0, 0, 0.707, 0.707],
                    scale=[0.4, 0.4, 0.4]):

        # ================================= load object ========================================
        prim_utils.create_prim(f"/World/envs/env_{env_index}" +
                               f"/manipulated_object_{object_index}",
                               usd_path=usd_path,
                               translation=translaton,
                               orientation=orientation,
                               scale=scale)

        stage = omni.usd.get_context().get_stage()
        object_prim = stage.GetPrimAtPath(
            f"/World/envs/env_{env_index}" +
            f"/manipulated_object_{object_index}")

        # ================================= set property ========================================
        # Make it a rigid body
        # utils.setRigidBody(object_prim, "convexHull", True)
        # # mass_api = UsdPhysics.MassAPI.Apply(object_prim)
        # # mass_api.CreateMassAttr(10)
        # # # Alternatively set the density
        # # mass_api.CreateDensityAttr(1000)
        # UsdPhysics.CollisionAPI.Apply(object_prim)

        # self._sim_config.apply_rigid_body_settings("Object", object_prim.GetPrim(),self._sim_config.parse_actor_config("Object"),is_articulation=False)
        # Make it a rigid body with kinematic
        # utils.setRigidBody(object_prim, "convexMeshSimplification", True)

        # mass_api = UsdPhysics.MassAPI.Apply(object_prim)
        # mass_api.CreateMassAttr(10)
        # # Alternatively set the density
        # mass_api.CreateDensityAttr(1000)
        # UsdPhysics.CollisionAPI.Apply(object_prim)

        # ================================= set property ========================================
        # Make it a rigid body with kinematic
        # utils.setRigidBody(object_prim, "convexMeshSimplification", True)

        # mass_api = UsdPhysics.MassAPI.Apply(object_prim)
        # mass_api.CreateMassAttr(10)
        # # Alternatively set the density
        # mass_api.CreateDensityAttr(1000)
        UsdPhysics.CollisionAPI.Apply(object_prim)
        self._sim_config.apply_rigid_body_settings(
            "Object",
            object_prim.GetPrim(),
            self._sim_config.parse_actor_config("Object"),
            is_articulation=False)

        # ================================= add texture ========================================
        # Change the server to your Nucleus install, default is set to localhost in omni.isaac.sim.base.kit
        default_server = carb.settings.get_settings().get(
            "/persistent/isaac/asset_root/default")
        mtl_created_list = []
        # Create a new material using OmniPBR.mdl
        omni.kit.commands.execute(
            "CreateAndBindMdlMaterialFromLibrary",
            mdl_name="OmniPBR.mdl",
            mtl_name="OmniPBR",
            mtl_created_list=mtl_created_list,
        )
        stage = omni.usd.get_context().get_stage()
        mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])
        # Set material inputs, these can be determined by looking at the .mdl file
        # or by selecting the Shader attached to the Material in the stage window and looking at the details panel
        omni.usd.create_material_input(
            mtl_prim,
            "diffuse_texture",
            default_server +
            "/Isaac/Samples/DR/Materials/Textures/marble_tile.png",
            Sdf.ValueTypeNames.Asset,
        )

        # Bind the material to the prim
        cube_mat_shade = UsdShade.Material(mtl_prim)
        UsdShade.MaterialBindingAPI(object_prim).Bind(
            cube_mat_shade, UsdShade.Tokens.strongerThanDescendants)

    def load_sphere(self):

        target = DynamicSphere(prim_path=self.default_zero_env_path +
                               "/target",
                               name="target",
                               radius=0.025,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings(
            "target", get_prim_at_path(target.prim_path),
            self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)

    def add_scene(self, scene, prim_paths_expr, name):

        object = RigidPrimView(
            prim_paths_expr=
            prim_paths_expr,  #"/World/envs/.*/manipulated_object_1"
            name=name,  #manipulated_object_view
            reset_xform_properties=False)
        scene.add(object)

        return object
