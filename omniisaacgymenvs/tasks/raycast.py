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
# wp.set_device(DEVICE)


@wp.kernel
def draw(mesh: wp.uint64, radius: wp.float32, cam_pos: wp.vec3, cam_dir: wp.vec4, width: int, height: int, pixels: wp.array(dtype=wp.vec3), 
         t_out: wp.array(dtype=wp.float32), u_out: wp.array(dtype=wp.float32), v_out: wp.array(dtype=wp.float32), 
         cam_dir_array: wp.array(dtype=wp.vec3), x_out: wp.array(dtype=wp.int32), y_out: wp.array(dtype=wp.int32)):
    pi = 3.141592653589793
    EMITTER_DIAMETER = wp.tan(17.5*pi/180.)*2.
    # Warp quaternion is x, y, z, w
    q2 = wp.quat(cam_dir[1], cam_dir[2], cam_dir[3], cam_dir[0])
    q = wp.quat(0.0,-0.707,0.0,0.707)
    x_test=wp.vec3(1.0,0.0,0.0)
    # output = wp.quat_rotate(q2, x_test)

    tid = wp.tid()
    # print('tid')
    # print(tid)
    z = tid % width
    y = tid // height
    # print('x')
    # print(z)
    # print('y')
    # print(y)
    # sz = 2.0 * float(z) / float(height) - 1. 
    # sy = 2.0 * float(y) / float(height) - 1. 
    sz = EMITTER_DIAMETER / (float(width) - 1.) * float(z) - float(EMITTER_DIAMETER) / 2.
    sy = EMITTER_DIAMETER / (float(height) - 1.) * float(y) - float(EMITTER_DIAMETER) / 2.
    # hypotenuse = wp.sqrt(wp.pow(sz, 2) + wp.pow(wp.pow(sy, 2)))
    # print('sx')
    # print(sz)
    # print('sy')
    # print(sy)
    # compute view ray
    ro = cam_pos
    # rd = wp.normalize(output)
    grid_vec = wp.vec3(1.0, sy, sz)
    # rd = wp.normalize(wp.quat_rotate(q2, grid_vec))
    rd = wp.quat_rotate(q2, grid_vec)
    # print('ro')
    # print(ro)
    # print('rd')
    # print(rd)
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    color = wp.vec3(0.0, 0.0, 0.0)

    if wp.mesh_query_ray(mesh, ro, rd, 1.0e6, t, u, v, sign, n, f):
        # if distance between [u,v] and ro is less than radius
        if wp.abs(wp.sqrt(u * u + v * v)) < radius:
            color = n * 0.5 + wp.vec3(0.5, 0.5, 0.5)

    pixels[tid] = color
    t_out[tid] = t
    u_out[tid] = u
    v_out[tid] = v
    if wp.abs(wp.sqrt(sz * sz + sy * sy)) < (EMITTER_DIAMETER / 2.):
        cam_dir_array[tid] = rd
    x_out[tid] = z
    y_out[tid] = y


class Raycast:
    def __init__(self, width = 1024, height = 1024, debug = False):
        self.debug = debug
        self.width = 10
        self.height = 10
        self.cam_pos = (0.0, 1.5, 2.5)
        self.step = 0
        self.result = np.zeros((self.height, self.width, 3))

    def set_geom(self, mesh):
        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.mesh = mesh
        self.ray_hit = wp.zeros(self.width * self.height, dtype=wp.float32)
        # if self.debug:
        self.u = wp.zeros(self.width * self.height, dtype=wp.float32)
        self.v = wp.zeros(self.width * self.height, dtype=wp.float32)
        self.cam_dir = wp.zeros(self.width * self.height, dtype=wp.vec3)
        self.x = wp.zeros(self.width * self.height, dtype=wp.int32)
        self.y = wp.zeros(self.width * self.height, dtype=wp.int32)

    def update(self):
        pass

    def render(self, cam_pos = (0.0, 1.5, 2.5), cam_dir = np.array([1, 0, 0, 0]), is_live=False):
        radius = 1
        with wp.ScopedTimer("render"):
            wp.launch(
                kernel=draw,
                dim=self.width * self.height,
                inputs=[self.mesh.id, radius, cam_pos, cam_dir, self.width, self.height, self.pixels, self.ray_hit, self.u, self.v, self.cam_dir, self.x, self.y]
            )

            wp.synchronize_device()
        
        self.step += 1
        # print("raytracer", self.ray_hit.numpy().shape)
        print("raytracer", self.ray_hit)
        return self.ray_hit, self.u, self.v

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
    transform = Gf.Matrix4d(Gf.Vec4d(scale[0], scale[1], scale[2], 1))
    #transform = UsdSkel.MakeTransform(translate, Gf.Quatf(1, 0, 0, 0), scale)
    size = cube.GetSizeAttr().Get()
    baked_trimesh = trimesh.creation.box(extents=(size, size, size))
    baked_trimesh.apply_transform(transform)
    return baked_trimesh


def get_trimesh_for_cylinder(cylinder: UsdGeom.Cylinder):
    transform = cylinder.GetLocalTransformation()
    translate, rotation, scale = UsdSkel.DecomposeTransform(transform)
    transform = Gf.Matrix4d(Gf.Vec4d(scale[0], scale[1], scale[2], 1))
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

if __name__ == "__main__":
    print("I AM IN MAIN!!!!!!!!")
    # Dimensions of image plane
    width = 64
    height = 64
    # Initialize Raycast for warp
    raycast = Raycast(width, height, debug = True)
    cam_pos = (0.0, 0.0, 0.0)
    # quaternion w, x, y, z
    # cam_dir = np.array([0.707, 0, -0.707, 0])
    cam_dir = np.array([1, 0, 0, 0])
    # Import usd into stage
    stage = Usd.Stage.Open('./cube.usd')
    prim = stage.GetPrimAtPath('/World/Cube')
    print(prim.GetName()) # prints "Prim"
    print(prim.GetPrimPath()) # prints "/Prim"
    # Convert stage prim to trimesh
    mesh = geom_to_trimesh(UsdGeom.Cube(prim))
    assert mesh.is_watertight
    mesh.vertices -= mesh.center_mass
    mesh.split
    # Run warp kernel for raycasting
    raycast.set_geom(warp_from_trimesh(mesh, DEVICE))
    raycast.render(cam_pos, cam_dir)

    ## Back to trimesh
    # Create axis for visualization
    axis_origins = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]])
    axis_directions = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    # stack axis rays into line segments for visualization as Path3D
    axis_visualize = trimesh.load_path(np.hstack((
        axis_origins,
        axis_origins + axis_directions)).reshape(-1, 2, 3), colors=np.array([[0, 0, 255, 255], [0, 255, 0, 255], [255, 0, 0, 255]]))

    # create rays for visualization
    ray_origins = np.zeros((width * height, 3))
    ray_directions = raycast.cam_dir.numpy()
    print('ray_origins')
    print(ray_origins)
    print('ray_directions')
    print(ray_directions)
    print('x')
    print(raycast.x)
    print('y')
    print(raycast.y)

    # run the mesh- ray test
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins,
        ray_directions=ray_directions)

    # stack rays into line segments for visualization as Path3D
    ray_visualize = trimesh.load_path(np.hstack((
        ray_origins,
        ray_origins + ray_directions)).reshape(-1, 2, 3))

    # make mesh transparent- ish
    mesh.visual.face_colors = [100, 100, 100, 100]
    blue = [0, 0, 255, 255]
    red = [255, 0, 0, 255]
    print('locations')
    print(locations)
    # create a visualization scene with rays, hits, and mesh
    scene = trimesh.Scene([
        mesh,
        ray_visualize,
        axis_visualize,
        trimesh.points.PointCloud(locations),
        trimesh.points.PointCloud(ray_origins, colors=blue),
        trimesh.points.PointCloud(ray_directions, colors=red)])

    # display the scene
    scene.show()