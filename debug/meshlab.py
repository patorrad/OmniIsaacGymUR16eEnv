import trimesh #[0.1, 0.1, 0.3]
mesh = trimesh.creation.box()
mesh.export("cube.stl")

import pymeshlab
def add_vertices(ms):
    ms.meshing_remove_duplicate_faces()
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_surface_subdivision_midpoint(iterations=10)

    return ms
ms = pymeshlab.MeshSet()
ms.load_new_mesh("cube.stl")
new_ms = add_vertices(ms)
new_ms.save_current_mesh("cube.stl")