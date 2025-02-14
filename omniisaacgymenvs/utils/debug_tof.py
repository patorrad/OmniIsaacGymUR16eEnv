from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.ros_bridge")

import omni.isaac.core as core
from omni.isaac.debug_draw import _debug_draw
import numpy as np
import time
import os
import sys
import torch

# Import necessary modules and set up the simulation world
from omni.isaac.core.world import World
from omni.isaac.core.objects.ground_plane import GroundPlane

import rospy
from std_msgs.msg import Float32MultiArray, UInt16MultiArray, Int32MultiArray

from omniisaacgymenvs.utils.tof_to_pcd import Tof_to_pcd

# from pynput import keyboard
import importlib

import pointnet2_sem_seg

import numpy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

class PointNet:
    def __init__(self):
        # self.classes = ['cylinder','box','floor', 'back', 'ceiling', 'side']
        self.classes = ['cyl', 'box', 'bottom', 'back', 'left', 'right', 'top']
        self.NUM_CLASSES = 7

        '''MODEL LOADING'''       
        self.classifier = pointnet2_sem_seg.get_model(self.NUM_CLASSES).cuda()
        checkpoint = torch.load('/home/paolo/Documents/gripper_pointnet/log/sem_seg/pointnet2_sem_seg/checkpoints/12-8-24_best_model_big_model.pth') #torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier = self.classifier.eval()

    def run_inference(self, points):
        points = torch.Tensor(points)
        points = points.float().cuda()
        points = points.transpose(2, 1)
        seg_pred, trans_feat = self.classifier(points)
        pred_val = seg_pred.contiguous().cpu().data.numpy()
        seg_pred = seg_pred.contiguous().view(-1, self.NUM_CLASSES)
        pred_val = np.argmax(pred_val, 2)
        return pred_val


class VisualizeToF:
    
    def __init__(self):
        self.debug_draw = _debug_draw.acquire_debug_draw_interface()
        self.tof_to_pcd = Tof_to_pcd()
        self.listener()
    
    def listener(self):
        
        rospy.init_node('listener', anonymous=True)
        # rospy.Subscriber("/sensor/filtered_data", UInt16MultiArray, self.callback)
        rospy.Subscriber("/sensor/data", Int32MultiArray, self.callback)
        # rospy.spin()
    
    def callback(self, msg):
        self.sensor_data = np.array(msg.data)
    
    def transform_points(self):
        R = [self.tof_to_pcd.quaternion_to_rotation_matrix(q) for q in self.tof_to_pcd.Q]
        R2 = [self.tof_to_pcd.quaternion_to_rotation_matrix(q) for q in self.tof_to_pcd.Q2]
        sensor_resolution = [8, 8]
        fov_h = 45
        fov_v = 45

        transformed_points = np.empty((8, 8, 3))
        for i in range(4):
            # distance = self.tof_to_pcd.filtered_data[int(0 + i*64) : int(64 + i*64)].reshape((8,8)) #/ 1000 # millimeters to meters
            distance = self.sensor_data[int(0 + i*64) : int(64 + i*64)].reshape((8,8)) #/ 1000 # millimeters to meters
            
            points = self.tof_to_pcd.get_tof_angles(sensor_resolution, fov_h, fov_v, distance)
            points = points.reshape((64, 3))

            for j in range(64):
                points[j, :] = self.tof_to_pcd.transform(points[j, :], R[i], self.tof_to_pcd.T[i], R2[i])
            points = points.reshape((8, 8, 3))

            transformed_points = np.concatenate((transformed_points, points))

        if not np.isnan(transformed_points).any() or not np.isinf(transformed_points).any():
            self.flattened_points = transformed_points.reshape(-1, 3) / 1000
        else:
            print("Nan or Inf values in transformed points")
            self.flattened_points = np.zeros((320, 3))

    def draw_points(self, pred=None):
        # flattened_points = np.random.rand(32, 8, 3).reshape(-1, 3)
        # import pdb; pdb.set_trace()
        # print(flattened_points[64:128,:].shape)
        # sensor_1_color=(1, 0, 0, 1) #, (0, 1, 0, 1), (0, 0, 1, 1), (1, 0, 1, 1)]
        # sensor_2_color=(0, 1, 1, 1)
        # sensor_3_color=(0, 0, 1, 1)
        # sensor_4_color=(0, 1, 0, 1)
        # colors = [sensor_1_color] * self.flattened_points[64:128,:].shape[0] + [sensor_2_color] * self.flattened_points[128:192,:].shape[0] + [sensor_3_color] * self.flattened_points[192:256,:].shape[0] + [sensor_4_color] * self.flattened_points[256:320,:].shape[0]
        # point_size=5
        # self.debug_draw.clear_points()
        # self.debug_draw.draw_points(self.flattened_points[64:], colors, [point_size] * self.flattened_points[64:].shape[0])

        # Define a mapping from values to colors
        # 0: red,   1: green, 2: blue, 3: yellow, 4: magenta, 5: cyan, 6: white
        # ['cylinder','box',    'bottom', 'back',   'left',     'right', 'top']
        value_to_color = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1), (0, 0, 0, 1)]
        colors = [value_to_color[i] for i in pred[0]]
        # value_to_color = {0: (1, 0, 0, 1), 1: (0, 1, 0, 1), 2: (0, 0, 1, 1), 3: (1, 1, 0, 1), 4: (1, 0, 1, 1), 5: (0, 1, 1, 1), }
        # Map each value in the array to its corresponding color
        # colors = np.vectorize(value_to_color.get)(input_array)
        # colors = [sensor_1_color] * self.flattened_points[64:128,:].shape[0] + [sensor_2_color] * self.flattened_points[128:192,:].shape[0] + [sensor_3_color] * self.flattened_points[192:256,:].shape[0] + [sensor_4_color] * self.flattened_points[256:320,:].shape[0]
        point_size=5
        self.debug_draw.clear_points()
        self.debug_draw.draw_points(self.flattened_points[64:], colors, [point_size] * self.flattened_points[64:].shape[0])

class Affordance:
    def __init__(self):
        self.affordance = None


    def get_affordance(self):
        return self.affordance

    def set_affordance(self, affordance):
        self.affordance = affordance

    def reset_affordance(self):
        self.affordance = None



if __name__ == '__main__':

    # # Create the world instance
    world = World(stage_units_in_meters=1.0)
    # stage = world.stage

    # # Add a ground plane
    world.scene.add_default_ground_plane(z_position=-1.0)

    visualizeToF = VisualizeToF()
    pointnet = PointNet()

    # # Initialize the DebugDraw interface
    # debug_draw_instance = _debug_draw.acquire_debug_draw_interface()

    # # Your function for drawing points (already defined in your script)
    # def draw_points(debug_draw, points_array, point_color=(0, 1, 0, 1), point_size=5):
    #     flattened_points = points_array.reshape(-1, 3)
    #     debug_draw.clear_points()
    #     debug_draw.draw_points(flattened_points, [point_color] * flattened_points.shape[0], [point_size] * flattened_points.shape[0])

    # # Generate sample points
    # points_array = np.random.rand(32, 8, 3)  # Replace with actual data
    # print(points_array)
    # # Call the function to draw points
    # draw_points(debug_draw_instance, points_array)

    ## Code below is to save live data
    # visualizeToF.data = None
    # def on_press(key):
    #     try:
    #         print(f"Key {key.char} pressed")
    #         if key.char == 'f':
    #             np.save('data.npy', test.data)
    #             print("Data saved")
    #         elif key.char == 's':
    #             if visualizeToF.data is None:
    #                 visualizeToF.data = visualizeToF.flattened_points
    #             else:
    #                 visualizeToF.data = np.concatenate((visualizeToF.data, visualizeToF.flattened_points))
    #     except AttributeError:
    #         print(f"Special key {key} pressed")

    # def on_release(key):
    #     if key == keyboard.Key.esc:  # Stop listener with 'esc' key
    #         print("Exiting...")
    #         return False

    # # Start the listener in non-blocking mode
    # listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    # listener.start()

    # Run the simulation loop
    while simulation_app.is_running():
        world.step(render=True)
        visualizeToF.transform_points()
        points = np.expand_dims(visualizeToF.flattened_points[64:,:], axis=0)
        pred = pointnet.run_inference(points)
        visualizeToF.draw_points(pred)
        time.sleep(0.016)  # Sleep to limit the loop to about 60 FPS

