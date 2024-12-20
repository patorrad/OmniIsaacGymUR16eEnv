import sapien.core as sapien
from sapien.utils.viewer import Viewer
import numpy as np

def demo(fix_root_link, balance_passive_force):
    engine = sapien.Engine()
    renderer = sapien.VulkanRenderer()
    engine.set_renderer(renderer)

    scene_config = sapien.SceneConfig()
    scene = engine.create_scene(scene_config)
    scene.set_timestep(1 / 240.0)
    scene.add_ground(0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    viewer.set_camera_xyz(x=-2, y=0, z=1)
    viewer.set_camera_rpy(r=0, p=-0.3, y=0)

    # Load URDF
    loader: sapien.URDFLoader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot: sapien.Articulation = loader.load("omniisaacgymenvs/assests/robots/ur16e/ur16e.urdf")
    

    # Set initial joint positions

    target_joint_positions = np.zeros(6)
    target_joint_positions[0] = 0
    target_joint_positions[1] = -1.57
    target_joint_positions[2] = 1.57
    target_joint_positions[3] = 0
    target_joint_positions[4] = 0
    arm_init_qpos = np.zeros(6)
   
    # arm_init_qpos[0] = 1.57
   
    robot.set_qpos(target_joint_positions)

    pinocchio_model = robot.create_pinocchio_model()

  

    links = robot.get_links()
    names = [link.get_name() for link in links]
    robot.set_root_pose(sapien.Pose([0, 0, 0]))
    index = 0
    while not viewer.closed:
        for _ in range(1):  # render every 4 steps
            # if balance_passive_force:
            qf = robot.compute_passive_force(
                    external=False, coriolis_and_centrifugal=False)
            robot.set_qf(qf)
            #robot.set_qpos(target_joint_positions)
            scene.step()
        
        if index%200==0:
            robot.set_qpos(target_joint_positions)
            qf = robot.compute_passive_force(
                    external=False, coriolis_and_centrifugal=False)
            robot.set_qf(qf)
            #robot.set_qpos(target_joint_positions)
            scene.step()
            
        
        # result, success, error = pinocchio_model.compute_inverse_kinematics(
        #     ee_link.get_index(), target_pose, robot.get_qpos(),
        #     [1] * 6 + [0] * 6)
        scene.update_render()
        viewer.render()
        index+=1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix-root-link', action='store_true')
    parser.add_argument('--balance-passive-force', action='store_true')
    args = parser.parse_args()

    demo(fix_root_link=args.fix_root_link,
         balance_passive_force=args.balance_passive_force)


if __name__ == '__main__':
    main()
