# import pickle
# import pandas as pd
# from sklearn.model_selection import train_test_split

# List of pickle files to combine
# pickle_files = ['/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-12_box_gripper_frame/12-56-45/dataset.pkl', 
#                 '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-13_cylinder_gripper_frame/15-09-49/dataset.pkl', 
#                 '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-14_thickcyl_gripper_frame/21-58-48/dataset.pkl',
#                 '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-14_thinbox_gripper_frame/15-36-39/dataset.pkl',
#                 ]

# pickle_files = ['/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-19_empty_bin/14-37-20_noise_50/dataset.pkl',
#                 '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-19_empty_bin/13-53-27_noise_20/dataset.pkl',
#                 '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-20_cuboid_30deg/11-55-04/dataset.pkl',
#                 '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-20_cuboid_30deg/12-35-49/dataset.pkl',
#                 '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-20_cuboid_30deg/13-53-08/dataset.pkl',
#                 '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-20_cuboid_30deg/14-37-31/dataset.pkl']

# pickle_files = ['/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-13_cylinder_gripper_frame/15-09-49/dataset.pkl']

# pickle_files =['/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-27/07-35-35/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-27/06-40-36/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/22-16-24/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/21-49-15/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/21-14-19/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/20-51-05/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/18-24-34/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/18-04-31/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/15-17-33/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/14-55-28/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/14-10-25/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-26/13-44-02/dataset cuboid cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/22-38-01/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/20-55-28/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/19-19-08/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/16-02-25/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/15-37-09/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/14-41-22/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/14-01-31/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/12-32-49/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/12-06-53/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/11-50-53/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/11-30-24/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/10-47-49/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/10-28-50/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/10-03-14/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/09-06-24/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/00-47-15/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/00-31-47/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/00-15-51/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/23-55-30/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/23-21-52/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/22-48-22/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/21-13-50/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/20-49-36/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/18-06-47/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-19_empty_bin/14-37-20_noise_50/dataset.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-19_empty_bin/13-53-27_noise_20/dataset.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-20_cuboid_30deg/11-55-04/dataset.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-20_cuboid_30deg/12-35-49/dataset.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-20_cuboid_30deg/13-53-08/dataset.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-20_cuboid_30deg/14-37-31/dataset.pkl']

# pickle_files = [
#     # '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/20-55-28/dataset cuboid 20.492834980723547 noise50.pkl',
#     #            '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/19-19-08/dataset cuboid 20.492834980723547 noise50.pkl',
#     #            '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/16-02-25/dataset cuboid 20.492834980723547 noise50.pkl',
#     #            '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/15-37-09/dataset cuboid 20.492834980723547 noise50.pkl',
#     #            '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/14-41-22/dataset cuboid 20.492834980723547 noise50.pkl',
#     #            '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/14-01-31/dataset cuboid 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/10-47-49/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/10-28-50/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/10-03-14/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/09-06-24/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/00-47-15/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/00-31-47/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-23/00-15-51/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/23-55-30/dataset cyl 20.492834980723547 noise50.pkl',
#                '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/23-21-52/dataset cyl 20.492834980723547 noise50.pkl',
#             #    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/22-48-22/dataset cyl 20.492834980723547 noise50.pkl',
#             #    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/21-13-50/dataset cyl 20.492834980723547 noise50.pkl',
#             #    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/20-49-36/dataset cyl 20.492834980723547 noise50.pkl',
#             #    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-22/18-06-47/dataset cyl 20.492834980723547 noise50.pkl',
#             #    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-19_empty_bin/14-37-20_noise_50/dataset.pkl',
#             #    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-19_empty_bin/13-53-27_noise_20/dataset.pkl',
#             ]

# pickle_files = ['/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-12-04/17-38-17/dataset cyl 12.5 noise20.pkl',
#                 '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-12-04/17-26-08/dataset cyl 12.5 noise20.pkl']

# # Load and combine DataFrames
# dataframes = []
# for file in pickle_files:
#     with open(file, 'rb') as f:
#         df = pickle.load(f)  # Assuming each file contains a DataFrame
#         dataframes.append(df)

# combined_df = pd.concat(dataframes)
# # import pdb; pdb.set_trace()
# # import sys
# # import numpy as np
# # np.set_printoptions(threshold=sys.maxsize)
# # combined_df['Object_hit'].iloc[0] 
# shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
# train_df, test_df = train_test_split(shuffled_df, test_size=0.2, random_state=42)
# # Save the combined DataFrame to a new pickle file
# with open('/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/combined_empty_cuboid_train.pkl', 'wb') as f:
#     pickle.dump(train_df, f)
# with open('/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/combined_empty_cuboid_test.pkl', 'wb') as f:
#     pickle.dump(test_df, f)

# print("Combined pickle file saved as 'combined_empty_cuboid_cyl.pkl'")

import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the root directories to search for pickle files
root_dirs = [
    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-19_empty_bin',                                                                                                                                                                                                                                                                                                                                     
    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-12-04_moving_gripper_1',
    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-12-05_moving_gripper_1',
    '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-12-10_stacked'
]

# Find all pickle files in the specified directories
pickle_files = []
for root_dir in root_dirs:
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.pkl'):
                print("Appended file: ", os.path.join(subdir, file))
                pickle_files.append(os.path.join(subdir, file))

# Load and combine DataFrames
dataframes = []
for file in pickle_files:
    with open(file, 'rb') as f:
        df = pickle.load(f)  # Assuming each file contains a DataFrame
        dataframes.append(df)

# Combine all DataFrames
combined_df = pd.concat(dataframes)

# Shuffle and split into train and test sets
shuffled_df = combined_df.sample(frac=1).reset_index(drop=True)
train_df, test_df = train_test_split(shuffled_df, test_size=0.2, random_state=42)

# Save the combined DataFrame to new pickle files
# output_dir = '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs'
output_dir = '/Documents_ext/ToF_Dataset'
with open(os.path.join(output_dir, 'combined_empty_cuboid_cyl_stacked_train.pkl'), 'wb') as f:
    pickle.dump(train_df, f)
with open(os.path.join(output_dir, 'combined_empty_cuboid_cyl_stacked_test.pkl'), 'wb') as f:
    pickle.dump(test_df, f)

print(f"Combined pickle files saved as 'combined_stacked_train.pkl' and 'combined_stacked_test.pkl'")

