# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#############################################################################
# Example Ray Cast
#
# Shows how to use the built-in wp.Mesh data structure and wp.mesh_query_ray()
# function to implement a basic ray-tracer.
#
##############################################################################

import matplotlib.pyplot as plt
from pxr import Usd, UsdGeom, UsdSkel, Gf

import warp as wp
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
import os

import trimesh

DEVICE = 'cpu'
# DEVICE = 'cuda:0'
wp.init()
# wp.config.mode = "debug"
wp.set_device(DEVICE)

@wp.kernel
def draw(mesh: wp.uint64, cam_pos: wp.vec3, cam_dir: wp.vec4, width: int, height: int, pixels: wp.array(dtype=wp.vec3), 
         t_out: wp.array(dtype=wp.float32), ray_dir: wp.array(dtype=wp.vec3), rng_seed: wp.int32):
    # Warp quaternion is x, y, z, w
    q2 = wp.quat(cam_dir[1], cam_dir[2], cam_dir[3], cam_dir[0])
    # q2 = wp.quat(0., 1., 0., 0.)
    tid = wp.tid()

    pi = 3.14159265359
    y = tid // height
    z = tid % width

    # For 25 degree cone
    EMITTER_DIAMETER = wp.tan(12.5 * pi / 180.) * 2.

    # For inner edge of noise cone
    NO_NOISE_DIAMETER = wp.tan(11.486 * pi / 180.) * 2.

    sy = EMITTER_DIAMETER / (float(height) - 1.) * float(y) - float(EMITTER_DIAMETER) / 2.
    sz = EMITTER_DIAMETER / (float(width) - 1.) * float(z) - float(EMITTER_DIAMETER) / 2.

    # compute view ray
    ro = cam_pos
    # rd = wp.normalize(output)
    grid_vec = wp.vec3(1.0, sy, sz)
    rd = wp.quat_rotate(q2, grid_vec)
    # rd = wp.normalize(wp.vec3(0., 0., -1.0))
    # print(rd)
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    color = wp.vec3(0.0, 0.0, 0.0)

    if wp.abs(wp.sqrt(sz * sz + sy * sy)) < (EMITTER_DIAMETER / 2.):
        if wp.mesh_query_ray(mesh, ro, rd, 1.2, t, u, v, sign, n, f):
            color = n * 0.5 + wp.vec3(0.5, 0.5, 0.5)
            
            # ignore this ray if it wouldn't reflect back to the receiver
            ray_dot_product = wp.dot(rd, n)
            if ray_dot_product < -0.996 or ray_dot_product > -0.866:
                t = 0.
            # else:
            #     print(ray_dot_product)
            # if distance between [u,v] and ro is in the noise part of the cone
            if wp.abs(wp.sqrt(sz * sz + sy * sy)) > (NO_NOISE_DIAMETER) / 2.:
                # use random function to determine whether we should give the reading t or 0
                # from experiment: there were 9 out-of-range readings out of the 34 total for a given distance
                rng_state = wp.rand_init(rng_seed, tid)
                if wp.randf(rng_state) <= 9./34.:
                    t = float(0.)
                    # t = float(1.)
    
    pixels[tid] = color
    t_out[tid] = t
    ray_dir[tid] = rd


class Raycast:
    def __init__(self):
        self.width = 16 #1024
        self.height = 16 #1024
        self.cam_pos = (0.0, 1.5, 2.5)
        # self.cam_pos = (0.0, 1.50, 1)
        self.step = 0
        self.result = np.zeros((self.height, self.width, 3))

    def set_geom(self, mesh):
        # # asset_stage = Usd.Stage.Open("/home/aurmr/workspaces/paolo_ws/src/stage_test_1.usd") 
        # # cube_geom = asset_stage.GetPrimAtPath("/World/Cube")
        
        # points = np.array(geom.GetPointsAttr().Get())
        # indices = np.array(geom.GetFaceVertexIndicesAttr().Get())
        # # print(points.shape)
        # # print(indices)
        # # print(mesh_geom.GetFaceVertexCountsAttr().Get())
        # # print(mesh_geom.GetFaceVertexIndicesAttr())
        # # indices_test = np.array([0, 1, 3, 1, 3, 2])
        # indices_test = np.concatenate((np.delete(indices, np.arange(3, indices.size, 4)), np.delete(indices, np.arange(1, indices.size, 4))))
        # points_test = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=float)
        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)

        # # create wp mesh
        # self.mesh = wp.Mesh(
        #     points=wp.array(points, dtype=wp.vec3), velocities=None, indices=wp.array(indices_test, dtype=int)
        # )

        self.mesh = mesh
        self.ray_hit = wp.zeros(self.width * self.height, dtype=wp.float32)
        self.ray_dir = wp.zeros(self.width * self.height, dtype=wp.vec3)

    def update(self):
        pass

    def render(self, rng_seed = 42, cam_pos = (0.0, 1.5, 2.5), cam_dir = np.array([1, 0, 0, 0]), is_live=False):
        # print('cam_pose', cam_pos)
        # print('quaternion', cam_dir)
        # cam_dir = get_forward_direction_vector(cam_dir)
        # print('cam_dir', cam_dir)
        # with wp.ScopedTimer("render"):
        wp.launch(
            kernel=draw,
            dim=self.width * self.height,
            inputs=[self.mesh.id, cam_pos, cam_dir, self.width, self.height, self.pixels, self.ray_hit, self.ray_dir, rng_seed]
        )


        wp.synchronize_device()
        
        plt.imshow(
            self.ray_hit.numpy().reshape((self.height, self.width)), origin="lower", interpolation="antialiased"
        )
        # plt.colorbar(label="Distance", orientation="horizontal")
        # plt.show()
        # if self.step % 100 == 0:
        #     plt.savefig("/home/aurmr/Pictures/raycast_cube_1.png",
        #     bbox_inches ="tight",
        #     pad_inches = 1,
        #     transparent = True,
        #     facecolor ="g",
        #     edgecolor ='w',
        #     orientation ='landscape')
        
        # self.step += 1
        # print("raytracer", self.ray_hit.numpy().shape)
        # print("raytracer", self.ray_hit)
        return self.ray_hit, self.ray_dir

    def save(self):
        for i in self.result.shape[0]:
            plt.imshow(
                self.result.shape[i].numpy().reshape((self.height, self.width, 3)), origin="lower", interpolation="antialiased"
            )
            plt.savefig("/home/aurmr/Pictures/raycast_cube_{}.png".format(i),
                bbox_inches ="tight",
                pad_inches = 1,
                transparent = True,
                facecolor ="g",
                edgecolor ='w',
                orientation ='landscape')


def warp_from_trimesh(trimesh: trimesh.Trimesh, device):
    mesh = wp.Mesh(
        points=wp.array(trimesh.vertices, dtype=wp.vec3, device=device),
        indices=wp.array(trimesh.faces.flatten(), dtype=int, device=device))
    return mesh


def get_support_surfaces_trimesh(mesh: trimesh.Trimesh, for_normal=None, threshold=None):
    # No caching at the moment so don't put this in any loops
    facet_centroids = []
    if for_normal:
        scores = mesh.facets_normal.dot(for_normal)
        support_mask = scores < threshold
    else:
        support_mask = np.ones((len(mesh.facets)))
    facets = []
    for facet, total_area, is_support in zip(mesh.facets, mesh.facets_area, support_mask):
        if not is_support:
            continue
        facets.append(facet)
        weighted_centroid = 0
        for tri_index in facet:
            weighted_centroid += mesh.area_faces[tri_index] * mesh.triangles_center[tri_index]
        facet_centroids.append(weighted_centroid / total_area)
    return facets, mesh.facets_area[support_mask], np.array(facet_centroids), mesh.facets_normal[support_mask]


def geom_to_trimesh(geom):
    if isinstance(geom, UsdGeom.Mesh):
        trimesh = load_trimesh_from_usdgeom(geom)
    elif isinstance(geom, UsdGeom.Cube):
        trimesh = get_trimesh_for_cube(geom)
    elif isinstance(geom, UsdGeom.Cylinder):
        trimesh = get_trimesh_for_cylinder(geom)
    elif isinstance(geom, UsdGeom.Cone):
        trimesh = get_trimesh_for_cone(geom)
    elif isinstance(geom, UsdGeom.Sphere):
        trimesh = get_trimesh_for_sphere(geom)
    else:
        raise Exception("No mesh representation for obj" + str(geom))
    return trimesh


def get_trimesh_for_cube(cube: UsdGeom.Cube):
    transform = cube.GetLocalTransformation()
    translate, rotation, scale = UsdSkel.DecomposeTransform(transform)
    # maybe we need to incorporate translate into the transform matrix
    # transform = Gf.Matrix4d(Gf.Vec4d(scale[0], scale[1], scale[2], 1))
    # transform = trimesh.transformations.rotation_matrix(3.14/2, [1, 0, 0], translate)
    transform = trimesh.transformations.translation_matrix(translate)
    # transform = UsdSkel.MakeTransform(translate, Gf.Quatf(1, 0, 0, 0), scale)
    size = cube.GetSizeAttr().Get()
    baked_trimesh = trimesh.creation.box(extents=(size, size, size))
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


def get_trimesh_for_cylinder(cylinder: UsdGeom.Cylinder):
    transform = cylinder.GetLocalTransformation()
    translate, rotation, scale = UsdSkel.DecomposeTransform(transform)
    # transform = Gf.Matrix4d(Gf.Vec4d(scale[0], scale[1], scale[2], 1))
    transform = trimesh.transformations.translation_matrix(translate)
    baked_trimesh = trimesh.creation.cylinder(radius=cylinder.GetRadiusAttr().Get(), height=cylinder.GetHeightAttr().Get())
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


def get_trimesh_for_cone(cone: UsdGeom.Cone):
    baked_trimesh = trimesh.creation.cone(radius=cone.GetRadiusAttr().Get(), height=cone.GetHeightAttr().Get())
    baked_trimesh.apply_transform(trimesh.transformations.translation_matrix([0,0,-cone.GetHeightAttr().Get() / 2]))
    return baked_trimesh


def get_trimesh_for_sphere(shpere: UsdGeom.Sphere):
    transform = shpere.GetLocalTransformation()
    baked_trimesh = trimesh.creation.icosphere(radius=shpere.GetRadiusAttr().Get())
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


def load_trimesh_from_usdgeom(mesh: UsdGeom.Mesh):
    transform = mesh.GetLocalTransformation()
    baked_trimesh = trimesh.Trimesh(vertices=mesh.GetPointsAttr().Get(), faces=np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3))
    baked_trimesh.apply_transform(transform)
    return baked_trimesh

# Method to get direction vector from quaternion (forward direction)
# Quaternion = w, x, y, z
def get_forward_direction_vector(q: np.array) -> np.array:
    # return np.array([2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])])
    # return np.array([1 - 2 * (q[1] * q[1] + q[2] * q[2]), 2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1])])
    return np.array([-1,0,0])

if __name__ == "__main__":
    example = Raycast()
    example.render()