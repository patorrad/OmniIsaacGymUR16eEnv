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
from pxr import Usd, UsdGeom

import warp as wp
import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)

import os

wp.init()


@wp.kernel
def draw(mesh: wp.uint64, cam_pos: wp.vec3, width: int, height: int, pixels: wp.array(dtype=wp.vec3), t_out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    #print(tid)
    x = tid % width
    y = tid // width
    #print(x)
    #print(y)
    sx = 2.0 * float(x) / float(height) - 1.0
    #print(sx)
    sy = 2.0 * float(y) / float(height) - 1.0
    #print(sy)
    # compute view ray
    ro = cam_pos
    rd = wp.normalize(wp.vec3(sx, sy, -1.0))
    # rd = wp.normalize(wp.vec3(0., 0., -1.0))
    # print(rd)
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(0)

    color = wp.vec3(0.0, 0.0, 0.0)

    if wp.mesh_query_ray(mesh, ro, rd, 1.0e6, t, u, v, sign, n, f):
        color = n * 0.5 + wp.vec3(0.5, 0.5, 0.5)

    pixels[tid] = color
    t_out[tid] = t


class Raycast:
    def __init__(self):
        self.width = 1024
        self.height = 1024
        # self.cam_pos = (0.0, 1.5, 2.5)
        self.cam_pos = (0.0, 1.50, 1)

        asset_stage = Usd.Stage.Open("/home/aurmr/workspaces/paolo_ws/src/stage_test_1.usd") 
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/World/Cube"))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())
        # print(points.shape)
        # print(indices)
        # print(mesh_geom.GetFaceVertexCountsAttr().Get())
        # print(mesh_geom.GetFaceVertexIndicesAttr())
        # indices_test = np.array([0, 1, 3, 1, 3, 2])
        indices_test = np.concatenate((np.delete(indices, np.arange(3, indices.size, 4)), np.delete(indices, np.arange(1, indices.size, 4))))
        points_test = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=float)
        self.pixels = wp.zeros(self.width * self.height, dtype=wp.vec3)

        # create wp mesh
        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3), velocities=None, indices=wp.array(indices_test, dtype=int)
        )

        self.ray_hit = wp.zeros(self.width * self.height, dtype=wp.float32)

    def update(self):
        pass

    def render(self, is_live=False):
        with wp.ScopedTimer("render"):
            wp.launch(
                kernel=draw,
                dim=self.width * self.height,
                inputs=[self.mesh.id, self.cam_pos, self.width, self.height, self.pixels, self.ray_hit],
            )

            wp.synchronize_device()

        # plt.imshow(
        #     self.pixels.numpy().reshape((self.height, self.width, 3)), origin="lower", interpolation="antialiased"
        # )
        # plt.show()

        # plt.imshow(
        #     self.ray_hit.numpy().reshape((self.height, self.width)), origin="lower", interpolation="antialiased"
        # )
        # plt.show()
        print("raytracer", self.ray_hit.numpy().shape)
        print("raytracer", self.ray_hit)


if __name__ == "__main__":
    example = Raycast()
    example.render()
