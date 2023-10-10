# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``omni.isaac.urdf_importer``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_urdf.html


positional arguments:
  input               The path to the input URDF file.
  output              The path to store the USD file.

optional arguments:
  -h, --help          show this help message and exit
  --headless          Force display off at all times. (default: False)
  --merge_joints, -m  Consolidate links that are connected by fixed joints. (default: False)
  --fix_base, -f      Fix the base to where it is imported. (default: False)
  --gym, -g           Make the asset instanceable for efficient cloning. (default: False)

"""

"""Launch Isaac Sim Simulator first."""


import argparse
import contextlib

# omni-isaac-orbit
from omni.isaac.kit import SimulationApp



# add argparse arguments
parser = argparse.ArgumentParser("Utility to convert a URDF into USD format.")
parser.add_argument("--input", type=str, help="The path to the input URDF file.")
parser.add_argument("--output", type=str, help="The path to store the USD file.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument(
    "--merge_joints",
    "-m",
    action="store_true",
    default=False,
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument(
    "--fix_base", "-f", action="store_true", default=True, help="Fix the base to where it is imported."
)
parser.add_argument(
    "--gym", "-g", action="store_true", default=False, help="Make the asset instanceable for efficient cloning."
)
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""



_DRIVE_TYPE = {
    "none": 0,
    "position": 1,
    "velocity": 2,
}
"""Mapping from drive name to URDF importer drive number."""

_NORMALS_DIVISION = {
    "catmullClark": 0,
    "loop": 1,
    "bilinear": 2,
    "none": 3,
}
"""Mapping from normals division name to URDF importer normals division number."""

import os

import carb
import omni.isaac.core.utils.stage as stage_utils
import omni.kit.commands
from omni.isaac.core.simulation_context import SimulationContext

from assets import check_file_path

from omni.isaac.urdf import _urdf

# ./orbit.sh -p omniisaacgymenvs/utils/tools/convert_urdf.py --input="/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e/ur16e.urdf" --output="/home/aurmr/Documents/Entong/OmniIsaacGymUR16eEnv/omniisaacgymenvs/assests/robots/ur16e/ur16e.usd"
def main():
    # check valid file path
    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)
    if os.path.exists(dest_path):
        carb.log_warn(f"Destination file already exists: {dest_path}. Overwriting...")
    if not os.path.exists(os.path.dirname(dest_path)):
        os.makedirs(os.path.dirname(dest_path))

    # Import URDF config
    _, urdf_config = omni.kit.commands.execute("URDFCreateImportConfig")

    # Set URDF config
    # -- stage settings -- dont need to change these.
    urdf_interface = _urdf.acquire_urdf_interface()
        # Set the settings in the import config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.convex_decomp = False
    import_config.import_inertia_tensor = True
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751
    import_config.default_position_drive_damping = 52.35988
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print(f"Saving USD file: {dest_path}")
    print("URDF importer config:")
    for key in dir(urdf_config):
        if not key.startswith("__"):
            try:
                # get attribute
                attr = getattr(urdf_config, key)
                # check if attribute is a function
                if callable(attr):
                    continue
                # print attribute
                print(f"\t{key}: {attr}")
            except TypeError:
                # this is only the case for subdivison scheme
                pass
    print("-" * 80)
    print("-" * 80)

    # Import URDF file
    omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=urdf_path, import_config=import_config, dest_path=dest_path
    )
    print('==================================================================')

    # Simulate scene (if not headless)
    if not args_cli.headless:
        # Open the stage with USD
        stage_utils.open_stage(dest_path)
        # Load kit helper
        sim = SimulationContext()
        # stage_utils.add_reference_to_stage(dest_path, "/Robot")
        # Reinitialize the simulation
        # Run simulation
        with contextlib.suppress(KeyboardInterrupt):
            while True:
                # perform step
                sim.step()


if __name__ == "__main__":
    # Run cloning example
    main()
    # Close the simulator
    simulation_app.close()
