#Important installation: # install pycuda first, https://github.com/inducer/pycuda and

# from tkinter import E
from typing import List
import torch
import numpy as np
import cv2
from typing import List
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import time
from typing import Union
# datastructures
from pytorch3d.structures import Meshes

# 3D transformations functions
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, matrix_to_quaternion, quaternion_multiply

# rendering components
from pytorch3d.renderer import (RasterizationSettings, MeshRasterizer,
                                OpenGLPerspectiveCameras)

from pytorch3d.structures.meshes import (
    join_meshes_as_batch,
    join_meshes_as_scene,
    Meshes,
)

#=====================================faster ===========================
# import os
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# import OpenGL.EGL
# from pytorch3d.renderer.opengl import MeshRasterizerOpenGL
#=====================================faster ===========================
from pytorch3d.ops import interpolate_face_attributes
import pdb


import warnings
import math

CUDA_LAUNCH_BLOCKING = 1
warnings.filterwarnings('ignore')

#========================================================================================
# pytorch3d renderer
#========================================================================================


class Render:

    def __init__(self, device, num_envs,
                 objects_mesh, num_cameras, image_size):

        

        self.objects_mesh = objects_mesh

       
        self.num_cameras = num_cameras  # number of cameras in per robot

        self.device = device  #cuda or cpu(it is better to use cuda)

        self.batch_size = self.num_cameras * num_envs  # maximum number of cameras in the env

        self.image_size = image_size  #image_size

        # init camera
        self.init_camera()  # init camera parameter in the pyrender

   
        # init rasterizer
        self.init_rasterizer()  # init rasterizer for render

   

    #========================================================================================
    # init pytorch3d camera
    #========================================================================================
    def init_camera(self) -> None:

        # The transformation matrix between the pytorch3d coordinate and orignial digit cameras coordinate
        self.flip_R = torch.unsqueeze(torch.as_tensor(
            [[6.1232e-17, 0.0000e+00, 1.0000e+00],
             [1.0000e+00, 6.1232e-17, -6.1232e-17],
             [-6.1232e-17, 1.0000e+00, 3.7494e-33]],
            device=self.device),
                                      dim=0)
        
      

        self.flip_quat = (matrix_to_quaternion(self.flip_R)
                          )  # rotation to quaternion
        
       

        # init the cameras postion
        # the digit cameras is at (0,0,0.015), the same location in the pytorch3d is (0,0,-0.015)
        init_camera_pos = torch.unsqueeze(torch.as_tensor([0., 0, -0.015],
                                                          device=self.device),
                                          dim=0)

        # prepare for transformation
        transform = Transform3d(device=self.device).rotate(self.flip_R)
        # the camera location in the gel perspective
        self.init_camera_pos = transform.transform_points(
            init_camera_pos.view(-1, 3))

        # init size information
        # self.image_size, self.height, self.width = self.conf.sensor.camera.image_size, self.conf.sensor.camera.height, self.conf.sensor.camera.width
        self.height = self.image_size
        self.width = self.image_size
        # self.cameras = OpenGLPerspectiveCameras(device=self.device,
        #                                         znear=0.001,
        #                                         aspect_ratio=3 / 4,
        #                                         fov=60,
        #                                         R=self.flip_R,
        #                                         T=self.init_camera_pos)

        # self.cameras_center = self.cameras.get_camera_center()

        # self.rasterizer = MeshRasterizer(cameras=self.cameras,
        #                                  raster_settings=self.raster_settings)

    #========================================================================================
    # init rasterizer setting
    #========================================================================================
    def init_rasterizer(self) -> None:
        '''
        the params are free to be changed
        '''

        self.raster_settings = RasterizationSettings(
            image_size=self.image_size,
            # max_faces_per_bin=3000
            # faces_per_pixel=1,
        )

  
   


   
    #========================================================================================
    # update the postion of cameras
    #========================================================================================
    def update_camera(self, cameras_pos: torch.Tensor,
                      cameras_quat: torch.Tensor):
        '''
        very tricky transformation
        '''

        # add flip R to the rotation to flip the camera
        self.R_cameras = torch.repeat_interleave(self.flip_R,
                                                 len(cameras_pos),
                                                 dim=0)

        # obtain T_cameras
        self.T_cameras = torch.repeat_interleave(self.init_camera_pos,
                                                 len(cameras_pos),
                                                 dim=0)

        # pdb.set_trace()

        # self.cameras = OpenGLPerspectiveCameras(device=self.device,
        #                                         znear=0.001,
        #                                         aspect_ratio=3 / 4,
        #                                         fov=60,
        #                                         R=self.R_cameras,
        #                                         T=self.T_cameras)

        # self.cameras_center = self.cameras.get_camera_center()

        # self.rasterizer = MeshRasterizer(cameras=self.cameras,
        #                                  raster_settings=self.raster_settings)

    #========================================================================================
    # update the vertices of object
    #========================================================================================
    def update_obj_vertices(self, cameras_pos: torch.Tensor,
                            cameras_quat: torch.Tensor, obj_pos: torch.Tensor,
                            obj_quat: torch.Tensor) -> torch.Tensor:
        '''
        To simplify the calculation, assume the location and orientation of digit
        as unchanged and the relative transformation matrix between the digit and object
        need to be calculated
        
        '''
        
        obj_mesh, obj_face = self.update_single_link_vertices(
            cameras_pos, cameras_quat, obj_pos, obj_quat)

    

        return obj_mesh, obj_face

    def update_transform(self, cameras_pos: torch.Tensor,
                         cameras_quat: torch.Tensor, obj_pos: torch.Tensor,
                         obj_quat: torch.Tensor) -> torch.Tensor:

        #obtain quaternion for the relative transformation
       
        quaternion = quaternion_multiply(quaternion_invert(cameras_quat),
                                         obj_quat)
        rotation = quaternion_to_matrix(quaternion_invert(quaternion))

        #obtain translations for the relative transformation
        transform = Transform3d(device=self.device).rotate(
            quaternion_to_matrix((cameras_quat)))
        translations = transform.transform_points(
            (obj_pos - cameras_pos).view(-1, 1, 3))[:, 0]

        # apply transformations to vertices
        transform = Transform3d(
            device=self.device).rotate(rotation).translate(translations)

        return transform

    def update_single_link_vertices(self, cameras_pos: torch.Tensor,
                                    cameras_quat: torch.Tensor,
                                    obj_pos: torch.Tensor,
                                    obj_quat: torch.Tensor) -> torch.Tensor:

      
        transform = self.update_transform(cameras_pos, cameras_quat, obj_pos,
                                          obj_quat)


        obj_mesh = []
        obj_face = []

      

          
        scales = self.obj_scales

        scale_vertices = self.objects_mesh[0] * (scales)
       

        transform_obj_mesh = transform.transform_points(scale_vertices.float().view(1, -1, 3))

        obj_mesh = transform_obj_mesh

        obj_face = self.objects_mesh[1]

            # obj_mesh.append(transform_obj_mesh)

            # obj_face.append(self.objects_mesh[obj_index][1])
        # ax = plt.axes(projection='3d')
        # # points_pixel = pixel_coords[0].cpu().view(-1, 1, 3)
        # # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        # #            points_pixel[:, :, 1].reshape(-1),
        # #            points_pixel[:, :, 2].reshape(-1),
        # #            c=points_pixel[:, :, 0].reshape(-1))

        # points_pixel = objects_mesh[0].cpu().view(-1, 1, 3)
        # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        #            points_pixel[:, :, 1].reshape(-1),
        #            points_pixel[:, :, 2].reshape(-1),
        #            c='r')
        # points_pixel = self.digit_vertices.cpu().view(-1, 1, 3)
        # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        #            points_pixel[:, :, 1].reshape(-1),
        #            points_pixel[:, :, 2].reshape(-1),
        #            c='b')
        # plt.show()

        return obj_mesh, obj_face

    

    #========================================================================================
    # compute the coord and normal for pixel
    #========================================================================================
    def compute_pixel_normal_coord(
            self, mesh, pix_to_face: torch.Tensor,
            bary_coords: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        '''
        compute the normal vector and coordinate for each pixel in the image
        '''

        # compute pixel normal
        faces = mesh.faces_packed()  # (F, 3)
        vertex_normals = mesh.verts_normals_packed()  # (V, 3)
        faces_normals = vertex_normals[faces]
        pixel_normals = interpolate_face_attributes(pix_to_face, bary_coords,
                                                    faces_normals)

        # compute pixel coord
        verts_packed = mesh.verts_packed()
        faces_verts = verts_packed[faces]
        pixel_coords = interpolate_face_attributes(pix_to_face, bary_coords,
                                                   faces_verts)
        return pixel_normals, pixel_coords

    #========================================================================================
    # render mesh as batch
    #========================================================================================

    def render_mesh_batch(
            self, verts_list: List, faces_list: List
    ) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        render the mesh as batch to reduce GPU storage
        
        Input:  verts_list,
                faces_list,
                start(the starter index for the vertices)
                end(the end index for the vertices)
        '''
   
        mesh = Meshes(verts=verts_list, faces=faces_list[None], textures=None)
        self.cameras = OpenGLPerspectiveCameras(device=self.device,
                                                znear=0.001,
                                                aspect_ratio=3 / 4,
                                                fov=60,
                                                R=self.R_cameras,
                                                T=self.T_cameras)

        # Init rasterizer

        self.rasterizer = MeshRasterizer(cameras=self.cameras,
                                         raster_settings=self.raster_settings)

        # render

        fragments = self.rasterizer(mesh)

        pix_to_face, depth, bary_coords = fragments.pix_to_face, fragments.zbuf, fragments.bary_coords

        # obtain normnals and coords for each pixel in the image
        pixel_normals, pixel_coords = self.compute_pixel_normal_coord(
            mesh, pix_to_face, bary_coords)

        return pixel_normals, pixel_coords, depth

    def update(
            self, obj_pos: torch.Tensor, obj_quat: torch.Tensor,
            cameras_pos: torch.Tensor,
            cameras_quat: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        '''
        update light, cameras, mesh
        '''
        
        # update cameras location
        self.update_camera(cameras_pos, cameras_quat)

        #update the mesh vertices of the objects
        obj_mesh, obj_face = self.update_obj_vertices(cameras_pos,
                                                      cameras_quat, obj_pos,
                                                      obj_quat)
       
        return obj_mesh, obj_face

    #========================================================================================
    # render mesh
    #========================================================================================
    def render_mesh(
        self, obj_pos: torch.Tensor, obj_quat: torch.Tensor,
        cameras_pos: torch.Tensor, cameras_quat: torch.Tensor
    ) -> Union[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        render mesh 
        '''

        obj_mesh, obj_face = self.update(obj_pos, obj_quat,
                                                   cameras_pos, cameras_quat)
        #joint mesh into one scene
        # verts_list, faces_list = self.join_mesh_scene(obj_mesh, obj_face,
        #                                               gel_mesh)

        # pixel_normals, pixel_coords, depth = self.render_mesh_batch(
        #     verts_list, faces_list)

        pixel_normals, pixel_coords, depth = self.render_mesh_batch(
            obj_mesh, obj_face)

        # ax = plt.axes(projection='3d')
        # # points_pixel = pixel_coords[0].cpu().view(-1, 1, 3)
        # # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        # #            points_pixel[:, :, 1].reshape(-1),
        # #            points_pixel[:, :, 2].reshape(-1),
        # #            c=points_pixel[:, :, 0].reshape(-1))
        # points_pixel = obj_mesh[-1].cpu().view(-1, 1, 3)
        # ax.scatter(points_pixel[:, :, 0].reshape(-1),
        #            points_pixel[:, :, 1].reshape(-1),
        #            points_pixel[:, :, 2].reshape(-1),
        #            c='r')
        # ax.scatter(cameras_pos[:, 0].cpu().reshape(-1),
        #            cameras_pos[:, 1].cpu().reshape(-1),
        #            cameras_pos[:, 2].cpu().reshape(-1),
        #            c='b')
       
        # plt.show()
       
        return pixel_normals, pixel_coords, depth

   

    #========================================================================================
    # render
    #========================================================================================
    def render(self, obj_pos: torch.Tensor, obj_quat: torch.Tensor,
               cameras_pos: torch.Tensor, cameras_quat: torch.Tensor,
               obj_scales: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:

        # ajust the camera pos by force

        self.batch_size = len(obj_pos)
      

        self.obj_scales = obj_scales

        # if (obj_pos is not None) and (
        #         self.conf.sensor.force.enable):  # only if we need to do this
        #     cameras_pos = self.adjust_with_force(obj_pos, cameras_pos,
        #                                          robo_force)

        # render  batch

        pixel_normals, pixel_coords, depth = self.render_mesh(
            obj_pos, obj_quat, cameras_pos, cameras_quat)

        # normalize pixel normal
        pixel_normals = pixel_normals.view(self.batch_size, self.height,
                                           self.width, 3)
        pixel_normals = pixel_normals / \
            torch.norm(pixel_normals, dim=3, keepdim=True)

        # resize pixel coord
        pixel_coords = pixel_coords.view(self.batch_size, self.height,
                                         self.width, 3)

    

        return depth


######################################################################################################################################
    
import trimesh
mesh = trimesh.load("/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/cube.stl")
objects_mesh = [torch.as_tensor(mesh.vertices,dtype=torch.float32).to("cuda:0"),torch.as_tensor(mesh.faces).to("cuda:0")]


renderer = Render(device="cuda:0", num_envs=1,
                 objects_mesh=objects_mesh, num_cameras=1, image_size=64)

camera_data = np.load("/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/outputs/2024-01-22/13-38-23/ee.npy")
object_data = np.load("/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/outputs/2024-01-22/13-38-23/object.npy")
obj_pos = torch.as_tensor(object_data[0,:3][None],device="cuda:0")
obj_quat = torch.as_tensor(object_data[0,3:][None],device="cuda:0")

cameras_pos = torch.as_tensor(camera_data[0,:3][None],device="cuda:0")
cameras_quat = torch.as_tensor(camera_data[0,3:][None],device="cuda:0")

obj_scales = torch.as_tensor(np.array([0.1, 0.1, 0.3])[None],device="cuda:0")

depth = renderer.render(obj_pos, obj_quat,
               cameras_pos, cameras_quat,
               obj_scales) 
import cv2
import pdb
# pdb.set_trace()
plt.figure(figsize=(10, 10))
plt.imshow(depth[0].cpu().numpy())
plt.axis("off")
plt.show()