import matplotlib.pyplot as plt
from pxr import Usd, UsdGeom, UsdSkel, Gf

import warp as wp
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)
import os

import trimesh
from trimesh import transformations
from cprint import *

# DEVICE = 'cpu'
DEVICE = 'cuda:0'
wp.init()
# wp.config.mode = "debug"
wp.set_device(DEVICE)
wp.config.print_launches = False

MAX_DIST = 1.2  # meters


@wp.kernel
def draw(mesh_id: wp.uint64, cam_pos: wp.vec3, cam_dir: wp.vec4, width: int,
         height: int, pixels: wp.array(dtype=wp.vec3),
         ray_dist: wp.array(dtype=wp.float32),
         ray_dir: wp.array(dtype=wp.vec3),
         normal_vec: wp.array(dtype=wp.vec3)):
    # Warp quaternion is x, y, z, w
    q2 = wp.quat(cam_dir[1], cam_dir[2], cam_dir[3], cam_dir[0])
    tid = wp.tid()

    pi = 3.14159265359
    y = tid // height
    z = tid % width

    # For 25 degree cone
    EMITTER_DIAMETER = wp.tan(12.5 * pi / 180.) * 2.

    # For inner edge of noise cone
    NO_NOISE_DIAMETER = wp.tan(11.486 * pi / 180.) * 2.

    sy = EMITTER_DIAMETER / (float(height) -
                             1.) * float(y) - float(EMITTER_DIAMETER) / 2.
    sz = EMITTER_DIAMETER / (float(width) -
                             1.) * float(z) - float(EMITTER_DIAMETER) / 2.

    # compute view ray
    start = cam_pos
    # rd = wp.normalize(output)
    grid_vec = wp.vec3(1.0, sy, sz)
    dir = wp.quat_rotate(q2, grid_vec)
    # rd = wp.normalize(wp.vec3(0., 0., -1.0))
    # print(rd)
    t = float(0.0)
    bary_u = float(0.0)
    bary_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face = int(0)

    color = wp.vec3(0.0, 0.0, 0.0)

    if wp.abs(wp.sqrt(sz * sz + sy * sy)) < (EMITTER_DIAMETER / 2.):
        if wp.mesh_query_ray(mesh_id, start, dir, MAX_DIST, t, bary_u, bary_v,
                             sign, normal, face):
            color = normal * 0.5 + wp.vec3(0.5, 0.5, 0.5)

            # # ignore this ray if it wouldn't reflect back to the receiver
            # ray_dot_product = wp.dot(dir, normal)
            # if ray_dot_product < -0.996 or ray_dot_product > -0.866:
            #     t = 0.
            # # else:
            # #     print(ray_dot_product)
            # # if distance between [u,v] and ro is in the noise part of the cone
            # if wp.abs(wp.sqrt(sz * sz + sy * sy)) > (NO_NOISE_DIAMETER) / 2.:
            #     # use random function to determine whether we should give the reading t or 0
            #     # from experiment: there were 9 out-of-range readings out of the 34 total for a given distance
            #     rng_state = wp.rand_init(rng_seed, tid)
            #     if wp.randf(rng_state) <= 9./34.:
            #         t = float(0.)
            #         # t = float(1.)

    pixels[tid] = color
    ray_dist[tid] = t
    ray_dir[tid] = dir
    normal_vec[tid] = normal


class Raycast:

    def __init__(self, width, height, vertices, faces):
        self.width = width  #1024
        self.height = height  #1024
        self.cam_pos = (0.0, 1.5, 2.5)
        # self.cam_pos = (0.0, 1.50, 1)
        self.step = 0
        self.result = np.zeros((self.height, self.width, 3))

        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.ray_dist = wp.zeros(self.width * self.height, dtype=wp.float32)
        self.ray_dir = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.normal_vec = wp.zeros(self.width * self.height, dtype=wp.vec3)

        self.init_buffer(vertices, faces)

    def init_buffer(self, vertices, faces):
        self.warp_mesh_list = []
        for i, vert in enumerate(vertices):
            warp_mesh = wp.Mesh(points=wp.empty(shape=vert.shape,
                                                dtype=wp.vec3),
                                indices=wp.from_torch(
                                    faces[i].flatten(),
                                    dtype=wp.int32,
                                ))
            self.warp_mesh_list.append(warp_mesh)

    def set_geom(self, vertices, mesh_index):

        wp.copy(self.warp_mesh_list[mesh_index].points, vertices)
        self.warp_mesh_list[mesh_index].refit()

        self.mesh = self.warp_mesh_list[mesh_index]

        # empty buffer
        self.pixels.zero_()
        self.ray_dist.zero_()
        self.ray_dir.zero_()
        self.normal_vec.zero_()

    def update(self):
        pass

    def render(self,
               rng_seed=42,
               cam_pos=(0.0, 1.5, 2.5),
               cam_dir=np.array([1, 0, 0, 0]),
               is_live=False):
        

        wp.launch(kernel=draw,
                  dim=self.width * self.height,
                  inputs=[
                      self.mesh.id, cam_pos, cam_dir, self.width, self.height,
                      self.pixels, self.ray_dist, self.ray_dir, self.normal_vec
                  ])

        wp.synchronize_device()

        plt.imshow(self.ray_dist.numpy().reshape((self.height, self.width)),
                   origin="lower",
                   interpolation="antialiased")
        return self.ray_dist, self.ray_dir, self.normal_vec

    def save(self):
        for i in self.result.shape[0]:
            plt.imshow(self.result.shape[i].numpy().reshape(
                (self.height, self.width, 3)),
                       origin="lower",
                       interpolation="antialiased")
            plt.savefig("/home/aurmr/Pictures/raycast_cube_{}.png".format(i),
                        bbox_inches="tight",
                        pad_inches=1,
                        transparent=True,
                        facecolor="g",
                        edgecolor='w',
                        orientation='landscape')


def warp_from_trimesh(trimesh: trimesh.Trimesh, device):
    mesh = wp.Mesh(points=wp.array(trimesh.vertices,
                                   dtype=wp.vec3,
                                   device=device),
                   indices=wp.array(trimesh.faces.flatten(),
                                    dtype=int,
                                    device=device))
    return mesh


def get_support_surfaces_trimesh(mesh: trimesh.Trimesh,
                                 for_normal=None,
                                 threshold=None):
    # No caching at the moment so don't put this in any loops
    facet_centroids = []
    if for_normal:
        scores = mesh.facets_normal.dot(for_normal)
        support_mask = scores < threshold
    else:
        support_mask = np.ones((len(mesh.facets)))
    facets = []
    for facet, total_area, is_support in zip(mesh.facets, mesh.facets_area,
                                             support_mask):
        if not is_support:
            continue
        facets.append(facet)
        weighted_centroid = 0
        for tri_index in facet:
            weighted_centroid += mesh.area_faces[
                tri_index] * mesh.triangles_center[tri_index]
        facet_centroids.append(weighted_centroid / total_area)
    return facets, mesh.facets_area[support_mask], np.array(
        facet_centroids), mesh.facets_normal[support_mask]


# relative_pos is the position of the object that this object's position should be relative to.
# For example, this can be the position of the first object added to the Isaac Sim environment
def geom_to_trimesh(geom, relative_pos, relative_rot):
    if isinstance(geom, UsdGeom.Mesh):
        trimesh = load_trimesh_from_usdgeom(geom)
    elif isinstance(geom, UsdGeom.Cube):
        trimesh = get_trimesh_for_cube(geom, relative_pos, relative_rot)
    elif isinstance(geom, UsdGeom.Cylinder):
        trimesh = get_trimesh_for_cylinder(geom, relative_pos)
    elif isinstance(geom, UsdGeom.Cone):
        trimesh = get_trimesh_for_cone(geom)
    elif isinstance(geom, UsdGeom.Sphere):
        trimesh = get_trimesh_for_sphere(geom)
    else:
        raise Exception("No mesh representation for obj" + str(geom))
    return trimesh


# relative_pos is the position of the object that this object's position should be relative to.
# For example, this can be the position of the first object added to the Isaac Sim environment
def get_trimesh_for_cube(cube: UsdGeom.Cube, relative_pos, relative_rot):
    size = cube.GetSizeAttr().Get()
    transform = transformations.compose_matrix(angles=relative_rot,
                                               translate=relative_pos)
    baked_trimesh = trimesh.creation.box(extents=(size, size, size))
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


# relative_pos is the position of the object that this object's position should be relative to.
# For example, this can be the position of the first object added to the Isaac Sim environment
def get_trimesh_for_cylinder(cylinder: UsdGeom.Cylinder, relative_pos):
    transform = cylinder.GetLocalTransformation()
    translate, rotation, scale = UsdSkel.DecomposeTransform(transform)
    # transform = Gf.Matrix4d(Gf.Vec4d(scale[0], scale[1], scale[2], 1))
    transform = trimesh.transformations.translation_matrix([
        translate[0] - relative_pos[0], translate[1] - relative_pos[1],
        translate[2] - relative_pos[2]
    ])
    baked_trimesh = trimesh.creation.cylinder(
        radius=cylinder.GetRadiusAttr().Get(),
        height=cylinder.GetHeightAttr().Get())
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


def get_trimesh_for_cone(cone: UsdGeom.Cone):
    baked_trimesh = trimesh.creation.cone(radius=cone.GetRadiusAttr().Get(),
                                          height=cone.GetHeightAttr().Get())
    baked_trimesh.apply_transform(
        trimesh.transformations.translation_matrix(
            [0, 0, -cone.GetHeightAttr().Get() / 2]))
    return baked_trimesh


def get_trimesh_for_sphere(shpere: UsdGeom.Sphere):
    transform = shpere.GetLocalTransformation()
    baked_trimesh = trimesh.creation.icosphere(
        radius=shpere.GetRadiusAttr().Get())
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


def load_trimesh_from_usdgeom(mesh: UsdGeom.Mesh):
    transform = mesh.GetLocalTransformation()
    baked_trimesh = trimesh.Trimesh(
        vertices=mesh.GetPointsAttr().Get(),
        faces=np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3))
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


if __name__ == "__main__":
    example = Raycast()
    example.render()
