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
import time
from pytorch3d.transforms import quaternion_to_matrix, Transform3d, quaternion_invert, quaternion_to_axis_angle, quaternion_multiply, axis_angle_to_quaternion
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.core.utils.prims import get_prim_at_path
import cv2
import torch
# DEVICE = 'cpu'
DEVICE = 'cuda:0'
wp.init()
# wp.config.mode = "debug"
wp.set_device(DEVICE)
wp.config.print_launches = False

MAX_DIST = 1.2  # meters


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


def circle_points(radius, centers, normals, num_points):
    """
    Generate points on a batch of circles in 3D space.

    Args:
    radius (float): The radius of the circles.
    centers (torch.Tensor): a tensor of shape (batch_size, 3) representing the centers of the circles.
    normals (torch.Tensor): a tensor of shape (batch_size, 3) representing the normals to the planes of the circles.
    num_points (int): The number of points to generate on each circle.

    Returns:
    torch.Tensor: a tensor of shape (batch_size, num_points, 3) representing the points on the circles.
    """
    batch_size = centers.shape[0]

    # Normalize the normal vectors
    normals = normals / torch.norm(normals, dim=-1, keepdim=True)

    # Generate random vectors not in the same direction as the normals
    not_normals = torch.rand(batch_size, 3, device='cuda:0')
    while (normals * not_normals).sum(
            dim=-1).max() > 0.99:  # Ensure they're not too similar
        not_normals = torch.rand(batch_size, 3, device='cuda:0')

    # Compute the basis of the planes
    basis1 = torch.cross(normals, not_normals)
    basis1 /= torch.norm(basis1, dim=-1, keepdim=True)
    basis2 = torch.cross(normals, basis1)
    basis2 /= torch.norm(basis2, dim=-1, keepdim=True)

    # Generate points on the circles
    t = torch.arange(0,
                     2 * torch.pi,
                     step=2 * torch.pi / num_points,
                     device='cuda:0')

    circles = centers[:, None, :] + radius[:, None, :] * (
        basis1[:, None, :] * torch.cos(t)[None, :, None] +
        basis2[:, None, :] * torch.sin(t)[None, :, None])
   
    return circles


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a batch of quaternions to rotation matrices.

    Args:
    quaternion (torch.Tensor): a tensor of shape (batch_size, 4) representing the quaternions.

    Returns:
    torch.Tensor: a tensor of shape (batch_size, 3, 3) representing the rotation matrices.
    """
    w, x, y, z = quaternion.unbind(dim=-1)

    batch_size = quaternion.shape[0]

    rotation_matrix = torch.empty((batch_size, 3, 3), device='cuda:0')

    rotation_matrix[:, 0, 0] = 1 - 2 * y**2 - 2 * z**2
    rotation_matrix[:, 0, 1] = 2 * x * y - 2 * z * w
    rotation_matrix[:, 0, 2] = 2 * x * z + 2 * y * w
    rotation_matrix[:, 1, 0] = 2 * x * y + 2 * z * w
    rotation_matrix[:, 1, 1] = 1 - 2 * x**2 - 2 * z**2
    rotation_matrix[:, 1, 2] = 2 * y * z - 2 * x * w
    rotation_matrix[:, 2, 0] = 2 * x * z - 2 * y * w
    rotation_matrix[:, 2, 1] = 2 * y * z + 2 * x * w
    rotation_matrix[:, 2, 2] = 1 - 2 * x**2 - 2 * y**2

    return rotation_matrix


def find_plane_normal(num_env, quaternions):
    """
    Find the normal to a plane defined by a batch of points and rotations.

    Args:
    num_env: 
    quaternions (torch.Tensor): a tensor of shape (batch_size, 4) representing the rotations.

    Returns:
    torch.Tensor: a tensor of shape (batch_size, 3) representing the normals to the planes.
    """
    # Convert the quaternions to rotation matrices
    rotation_matrices = quaternion_to_rotation_matrix(quaternions)
    normals = torch.tensor([1.0, 0.0, 0.0],
                           device='cuda:0').expand(num_env, -1)
    normals = normals.view(num_env, 3, 1)
    rotated_normals = torch.bmm(rotation_matrices, normals)
    return rotated_normals.view(num_env, 3)


def draw_raytrace(debug_draw, debug_sensor_ray_pos_list,
                  debug_ray_hit_points_list, debug_ray_colors, debug_ray_sizes,
                  debug_end_point_colors, debug_point_sizes,
                  debug_start_point_colors, debug_circle):
    debug_draw.clear_lines()
    debug_draw.clear_points()

    debug_sensor_ray_pos_list = np.concatenate(debug_sensor_ray_pos_list,
                                               axis=0)
    debug_ray_hit_points_list = np.concatenate(debug_ray_hit_points_list,
                                               axis=0)
    debug_ray_colors = np.concatenate(debug_ray_colors, axis=0)
    debug_ray_sizes = np.concatenate(debug_ray_sizes, axis=0)
    debug_end_point_colors = np.concatenate(debug_end_point_colors, axis=0)
    debug_point_sizes = np.concatenate(debug_point_sizes, axis=0)
    debug_start_point_colors = np.concatenate(debug_start_point_colors, axis=0)
    debug_circle = np.concatenate(debug_circle, axis=0)

    debug_draw.draw_lines(debug_sensor_ray_pos_list, debug_ray_hit_points_list,
                          debug_ray_colors, debug_ray_sizes)
    debug_draw.draw_points(debug_ray_hit_points_list, debug_end_point_colors,
                           debug_point_sizes)
    debug_draw.draw_points(debug_sensor_ray_pos_list, debug_start_point_colors,
                           debug_point_sizes)
    # Debug draw the gripper pose
    debug_draw.draw_points(debug_circle, [(1, 0, 0, 1)], [10])


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
    EMITTER_DIAMETER = wp.tan(60.0 * pi / 180.) * 4.

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

    # if wp.abs(wp.sqrt(sz * sz + sy * sy)) < (EMITTER_DIAMETER / 2.):
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
    
    



class Renderer:

    def __init__(self, width, height, object_prime_path, _task_cfg, _cfg,
                 num_envs, device, default_sensor_radius):
        self.width = width  #1024
        self.height = height  #1024
        self.cam_pos = (0.0, 1.5, 2.5)
        # self.cam_pos = (0.0, 1.50, 1)
        self.step = 0
        self.result = np.zeros((self.height, self.width, 3))

        self.object_prime_path = object_prime_path
        self._task_cfg = _task_cfg
        self._cfg = _cfg
        self.num_envs = num_envs
        # draw info
        self.debug_draw = _debug_draw.acquire_debug_draw_interface()
        self.device = device

        self.default_sensor_radius = default_sensor_radius

        # init raycasting buffer
        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.ray_dist = wp.zeros(self.width * self.height, dtype=wp.float32)
        self.ray_dir = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.normal_vec = wp.zeros(self.width * self.height, dtype=wp.vec3)

        self.init_mesh()
        self.init_buffer([self.mesh_vertices[0]], [self.mesh_faces[0]])

    def init_mesh(self):
        from pxr import Usd, UsdGeom

        cube = UsdGeom.Cube(get_prim_at_path(self.object_prime_path))

        size = cube.GetSizeAttr().Get()
        cube = trimesh.creation.box(extents=(1, 1, 1))

        self.mesh_faces = torch.as_tensor(cube.faces[None, :, :],
                                          dtype=torch.int32).repeat(
                                              (self.num_envs, 1,
                                               1)).to(self.device)
        self.mesh_vertices = torch.as_tensor(cube.vertices[None, :, :],
                                             dtype=torch.float32).repeat(
                                                 (self.num_envs, 1,
                                                  1)).to(self.device)

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

    def transform_mesh(self, cur_object_pose, cur_object_rot, scale_size,
                       mesh_vertices):

        transform = Transform3d(device=self.device).scale(scale_size).rotate(
            quaternion_to_matrix(
                quaternion_invert(cur_object_rot))).translate(cur_object_pose)

        transformed_vertices = transform.transform_points(
            mesh_vertices.clone().to(self.device))

        max_xyz = torch.max(transformed_vertices, dim=1).values
        min_xyz = torch.min(transformed_vertices, dim=1).values
        bboxes = torch.hstack([min_xyz, max_xyz])
        center_points = (max_xyz + min_xyz) / 2

        return bboxes, center_points, transformed_vertices

    def set_geom(self, vertices, mesh_index):
        wp.build.clear_kernel_cache()

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

        return self.ray_dist, self.ray_dir, self.normal_vec

    def raytrace_step(self, gripper_pose, gripper_rot, cur_object_pose,
                      cur_object_rot, scale_size, sensor_radius) -> None:

        _, _, transformed_vertices = self.transform_mesh(
            cur_object_pose, cur_object_rot, scale_size, self.mesh_vertices)

        normals = find_plane_normal(self.num_envs, gripper_rot)
        raycast_circle = circle_points(
            sensor_radius, gripper_pose, normals,
            self._task_cfg['sim']["URRobot"]['num_sensors'])

        # for draw point
        if self._cfg["debug"]:
            debug_sensor_ray_pos_list = []
            debug_ray_hit_points_list = []
            debug_ray_colors = []
            debug_ray_sizes = []
            debug_point_sizes = []
            debug_end_point_colors = []
            debug_start_point_colors = []
            debug_circle = []

        self.raycast_reading = torch.zeros(
            (self.num_envs,
             self._cfg["raycast_width"] * self._cfg["raycast_height"] *
             self._task_cfg['sim']["URRobot"]['num_sensors'])).to(
                 self.device) - 1

        num_pixel = self._cfg["raycast_width"] * self._cfg["raycast_height"]
        # ray average distance
        self.raytrace_dist = torch.zeros((self.num_envs, 2)).to(self.device)
        # ray tracing reading
        self.raytrace_reading = torch.zeros(
            (self.num_envs,
             self._cfg["raycast_width"] * self._cfg["raycast_height"],
             2)).to(self.device)
        # ray trace coverage
        self.raytrace_cover_range = torch.zeros(
            (self.num_envs, 2)).to(self.device)
        # ray trace max min dist
        self.raytrace_dev = torch.zeros((self.num_envs, 2)).to(self.device)
        
        for i, env in zip(
                torch.arange(
                    self._task_cfg['sim']["URRobot"]['num_sensors']).repeat(
                        self.num_envs),
                torch.arange(self.num_envs).repeat_interleave(
                    self._task_cfg['sim']["URRobot"]['num_sensors'])):

            self.set_geom(wp.from_torch(transformed_vertices[env]),
                          mesh_index=0)
           
            # import time 
            # start = time.time()
          
            ray_t, ray_dir, normal = self.render(raycast_circle[env][i].clone(),gripper_rot[env])
        # import pdb
        # pdb.set_trace()
        
        
        #     # replace the zero value

        #     ray_t_copy = ray_t.clone()
         

            if self._cfg["debug"]:

                sensor_ray_pos_np = raycast_circle[env][i].cpu().numpy()
                sensor_ray_pos_tuple = (sensor_ray_pos_np[0],
                                        sensor_ray_pos_np[1],
                                        sensor_ray_pos_np[2])

        
                ray_dir = ray_dir.numpy()
                ray_t = ray_t.numpy()

                line_vec = np.transpose(
                    np.multiply(np.transpose(ray_dir), ray_t))

                # Get rid of ray misses (0 values)
                line_vec = line_vec[np.any(line_vec, axis=1)]
                ray_hit_points_list = line_vec + np.array(sensor_ray_pos_tuple)
                hits_len = len(ray_hit_points_list)

                if hits_len > 0:
                    sensor_ray_pos_list = [
                        sensor_ray_pos_tuple for _ in range(hits_len)
                    ]
                    ray_colors = [(1, i, 0, 1) for _ in range(hits_len)]
                    ray_sizes = [2 for _ in range(hits_len)]
                    point_sizes = [7 for _ in range(hits_len)]
                    start_point_colors = [
                        (0, 0.75, 0, 1) for _ in range(hits_len)
                    ]  # start (camera) points: green
                    end_point_colors = [(1, i, 1, 1) for _ in range(hits_len)]
                    
                    if self._cfg["debug"]:
                        debug_sensor_ray_pos_list.append(sensor_ray_pos_list)
                        debug_ray_hit_points_list.append(ray_hit_points_list)
                        debug_ray_colors.append(ray_colors)
                        debug_ray_sizes.append(ray_sizes)

                        debug_end_point_colors.append(end_point_colors)
                        debug_point_sizes.append(point_sizes)
                        debug_start_point_colors.append(start_point_colors)

                        debug_circle.append([raycast_circle[env][i].cpu().numpy()])

        if self._cfg["debug"]:

            if len(debug_sensor_ray_pos_list) > 0:

                draw_raytrace(self.debug_draw, debug_sensor_ray_pos_list,
                              debug_ray_hit_points_list, debug_ray_colors,
                              debug_ray_sizes, debug_end_point_colors,
                              debug_point_sizes, debug_start_point_colors,
                              debug_circle)

        # return self.raycast_reading, self.raytrace_cover_range, self.raytrace_dev

    def update_params(self, actions):
        action = torch.clip(actions, -1, 1)
       
        cur_sensor_radius = self.default_sensor_radius + action * 0.02
        print(cur_sensor_radius)
        return cur_sensor_radius


# if __name__ == "__main__":
#     example = Raycast()
#     example.render()
