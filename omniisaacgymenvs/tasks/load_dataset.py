import csv
import numpy as np
import pandas as pd

def remove_new_line(text):
    return text.replace('\n', ' ')

# Specify your csv file path
csv_file_path = '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-03-29/16-01-53/dataset.csv'

# Load the csv file into a numpy array
df = pd.read_table(csv_file_path, sep=',', engine='python', converters = {'Gripper_pose': remove_new_line})
                                                                        #   'Hits_Pose': remove_new_line,
                                                                        #   'Tof_reading': remove_new_line,
                                                                        #   'Object_hit': remove_new_line,
                                                                        #   'Object1_position': remove_new_line,
                                                                        #   'Object2_position': remove_new_line})

array = df.to_numpy()

print(array)

# create an array for each episode
episodes = np.split(array, np.where(np.diff(array[:, 0]))[0] + 1)

# "remove" objects that fall
# print(episodes[0][2][8])
# print(episodes[0][2][9])

# print(episodes[0][59][8])
# print(episodes[0][59][9])

# print(episodes[0])
