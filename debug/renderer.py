import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

# rendering components
from pytorch3d.renderer import (RasterizationSettings, MeshRasterizer,
                                OpenGLPerspectiveCameras)


# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# # Set paths
DATA_DIR = "./data"
obj_filename = "/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/cube.stl"

# Set paths
# DATA_DIR = "./data"
# obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

import trimesh
mesh=trimesh.load(obj_filename)

vertices = torch.as_tensor(mesh.vertices).to(device)[None].float()
faces = torch.as_tensor(mesh.faces).to(device)[None]

import numpy as np

camera_data = np.load("/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/outputs/2024-01-22/14-40-07/ee.npy")
object_data = np.load("/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/outputs/2024-01-22/14-40-07/object.npy")
obj_pos = torch.as_tensor(object_data[0,:3][None],device="cuda:0")
obj_quat = torch.as_tensor(object_data[0,3:][None],device="cuda:0")


cameras_pos = torch.as_tensor(camera_data[0,:3][None],device="cuda:0")
cameras_quat = torch.as_tensor(camera_data[0,3:][None],device="cuda:0")

from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion


vertices = vertices[0]@quaternion_to_matrix(obj_quat)[0]+obj_pos[0]

vertices = vertices*0.1

mesh = Meshes(verts=vertices[None], faces=faces, textures=None)
# Load obj file

# ax = plt.axes(projection='3d')
# ax.scatter(vertices[ :, 0].reshape(-1).cpu(),
#                    vertices[:, 1].reshape(-1).cpu(),
#                    vertices[:, 2].reshape(-1).cpu(),
#                    c='r')

# ax.scatter(cameras_pos[:,0].cpu(),cameras_pos[:,1].cpu(),cameras_pos[:,2].cpu(),c="b")
# plt.show()



# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
R, T = look_at_view_transform(2.7, 0, 180) 

R = quaternion_to_matrix(quaternion_invert(cameras_quat))
T = cameras_pos
cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that 
# the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for 
# explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of 
# the difference between naive and coarse-to-fine rasterization. 
raster_settings = RasterizationSettings(
    image_size=64, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)

# Place a point light in front of the object. As mentioned above, the front of the cow is facing the 
# -z direction. 
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
# apply the Phong lighting model
rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    )


fragments = rasterizer(mesh)

pix_to_face, depth, bary_coords = fragments.pix_to_face, fragments.zbuf, fragments.bary_coords

plt.figure(figsize=(10, 10))
plt.imshow(depth[0].cpu().numpy())
plt.axis("off")
plt.show()

from pytorch3d.ops import interpolate_face_attributes
def compute_pixel_normal_coord(
            mesh, pix_to_face: torch.Tensor,
            bary_coords: torch.Tensor):
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
    
pixel_normals, pixel_coords = compute_pixel_normal_coord(
            mesh, pix_to_face, bary_coords)


ax = plt.axes(projection='3d')
# import pdb
# pdb.set_trace()
points_pixel = pixel_coords[0].cpu().view(-1, 1, 3)
points_pixel = vertices[None].cpu().view(-1, 1, 3)
ax.scatter(points_pixel[:, :, 0].reshape(-1),
           points_pixel[:, :, 1].reshape(-1),
           points_pixel[:, :, 2].reshape(-1),
           c=points_pixel[:, :, 0].reshape(-1))
plt.show()