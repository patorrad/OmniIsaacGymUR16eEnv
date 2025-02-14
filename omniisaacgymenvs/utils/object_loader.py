import torch
import numpy as np

from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import DynamicCylinder
from omni.isaac.core.objects import FixedCuboid
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.prims import RigidPrimView

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf, Tf
from pxr import UsdPhysics, PhysxSchema
from omni.physx.scripts import utils
from omni.physx import acquire_physx_interface

import omni
import carb
import omni.isaac.core.utils.nucleus as nucleus_utils

from omni.isaac.core.utils.prims import get_prim_at_path, delete_prim, is_prim_path_valid
import os
import random
from cprint import *

class Object:

    def __init__(self, _sim_config, num_envs, device,
                 default_zero_env_path) -> None:
        self. _sim_config = _sim_config
        self.num_envs = num_envs
        self.device = device
        self.default_zero_env_path = default_zero_env_path

    def load_cube(self, scale, num_object=1 , objects=['DyanmicCylinder', 'DynamicCuboid']):
        self.scale_size = torch.as_tensor(scale).to(self.device)
        #self.scale_size = torch.as_tensor(scale).repeat(self.num_envs,1).to(self.device)

        # Cylinder scale
        # Cylinder scale:  0.04278853596915767 0.10075032265668002
        # Cylinder scale:  0.030985669961447092 0.09100043020890668
        # radius=random.uniform(0.03, 0.05) #orig shakti 0.0381, # opt range 0.04 to 0.85
        # height=random.uniform(0.1, 0.13) #orig shakti 0.0889, # opt range 0.08 to 0.17      

        for i in range(self.num_envs):
            
            for j in range(1,num_object+1):
                
                if objects[j-1] == 'DynamicCuboid':
                    # Randomize scale
                    rand = torch.rand((3), device='cuda:0')
                    min = [0.05, 0.05, 0.08]
                    max = [0.1, 0.1, 0.12]
                    rand[0] = min[0] + (max[0] - min[0]) * rand[0]
                    rand[1] = min[1] + (max[1] - min[1]) * rand[1]
                    rand[2] = min[2] + (max[2] - min[2]) * rand[2]
                    # import pdb; pdb.set_trace()
                    target = DynamicCuboid(
                        prim_path=f"/World/envs/env_{i}/manipulated_object_{j}",
                        name=f"manipulated_object_{j}",
                        position=[0, 0, 2.02],
                        scale = rand,
                        # scale=np.array(scale[j-1]),
                        color=torch.tensor([0, 169 / 255, 1]))
                else:
                    # Randomize scale
                    radius=random.uniform(0.03, 0.05) #orig shakti 0.0381, # opt range 0.04 to 0.85
                    height=random.uniform(0.1, 0.13) #orig shakti 0.0889, # opt range 0.08 to 0.17
                    cprint.ok("Cylinder scale: ", radius, height)
                    target = DynamicCylinder(
                        prim_path=f"/World/envs/env_{i}/manipulated_object_{j}",
                        name=f"manipulated_object_{j}",
                        position=[0, 0, 2.02],
                        # scale=np.array(scale[j-1]),
                        radius=radius, #orig shakti 0.0381, # opt range 0.04 to 0.85
                        height=height, #orig shakti 0.0889, # opt range 0.08 to 0.17
                        color=torch.tensor([1, 0, 0]))
                    # target = DynamicSphere(
                    #     prim_path=f"/World/envs/env_{i}/manipulated_object_{j}",
                    #     name=f"manipulated_object_{j}",
                    #     position=[0, 0, 2.02],
                    #     #scale=np.array(scale[j-1]),
                    #     radius=0.0381,
                    #     color=torch.tensor([1, 0, 0]))

                self._sim_config.apply_articulation_settings(
                    f"manipulated_object_{j}", get_prim_at_path(target.prim_path), # TABLE
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

    def load_table(self, position, orientation, scale, name="table"):
        table_translation = np.array(
            position)  #  self._task_cfg['sim']["Table"]["position"]
        table_orientation = np.array(
            orientation)  # self._task_cfg['sim']["Table"]["quaternion"]

        table = FixedCuboid(
            prim_path=self.default_zero_env_path + f"/{name}",
            name=name,
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
        table_prim = get_prim_at_path(table.prim_path) #get_prim_at_path(self.default_zero_env_path + f"/{name}")

        rigid_api = UsdPhysics.RigidBodyAPI.Apply(table_prim)   
        rigid_api.CreateKinematicEnabledAttr(True)

        # # # apply rigid body API and schema
        # # # physicsAPI = UsdPhysics.RigidBodyAPI.Apply(prim)
        # # PhysxSchema.PhysxRigidBodyAPI.Apply(table_prim)
        # attr = table_prim.GetAttribute("physics:kinematicEnabled")
        # # print(attr)

        self._sim_config.apply_rigid_body_settings(
            name,
            table_prim,
            self._sim_config.parse_actor_config(name),
            is_articulation=False)
        

    def load_pod(self, position, orientation, scale):
        prim_utils.create_prim(self.default_zero_env_path + "/pod",
                               usd_path="/home/shaktis/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/pod/pod.usd",
                               translation=position,
                               orientation=orientation,
                               scale=scale)

        stage = omni.usd.get_context().get_stage()
        pod_prim = stage.GetPrimAtPath(self.default_zero_env_path + "/pod")

        # self._sim_config.apply_rigid_body_settings(
        #     "pod",
        #     pod_prim,
        #     self._sim_config.parse_actor_config("pod"),
        #     is_articulation=False)
        
        # pod = RigidPrim(
        #     prim_path=self.default_zero_env_path + "/pod",
        #     name="pod",
        #     translation=position,
        #     orientation=orientation,
        #     scale=scale
        # )
        # pod.disable_rigid_body_physics()
        
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
            prim_paths_expr=prim_paths_expr,  #"/World/envs/.*/manipulated_object_1"
            name=name,  #manipulated_object_view
            reset_xform_properties=False)

        scene.add(object)

        return object
