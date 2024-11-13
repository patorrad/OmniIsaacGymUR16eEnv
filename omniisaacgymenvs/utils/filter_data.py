import numpy as np
import pandas as pd


path = '/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-12/12-44-35/dataset.pkl'
df = pd.read_pickle(path)

# get only environment 0 
env_0 = df[df['Env'] == 0]

# FOR DEBUGGING
#env_0.to_csv(r'testing.csv')

group_num = 0
groupies = []

# Group the dataframe by 'Episode' and select 'Object1_position'
grouped = env_0.groupby('Episode')['Object1_position']

print('Episodes pre-processing: ' + str(grouped.ngroups))

# Create a list to store the indices of the groups where the object does not move
keep_indices = []

# movement threshold
max = 0.0001 #0.015 #0.03 #0.015 #0.6 inches
import pdb; pdb.set_trace()
# Iterate over the groups
for name, group in grouped:
    group_num += 1

    positions = group.to_numpy()
    positions = np.concatenate(positions)
    positions = np.reshape(positions,(-1,3))
    positions = positions[5:]
 
    range = np.ptp(positions[5:], axis = 0)
    
    if np.max(range) <= max and group_num != len(grouped)- 1:
        keep_indices.extend(group.index.tolist())
        groupies.append(group_num)

print('Episodes post-processing: ' + str(len(groupies)))

# Create a new dataframe that only includes the groups where the object does not move
env_0_no_move = env_0.loc[keep_indices]

hit_poses = env_0_no_move.loc[:,'Hits_Pose'].to_numpy()
objects_hit = env_0_no_move.loc[:,'Object_hit'].to_numpy()

# each entry is a step contains array of [hit_poses, object_hit] (58 steps make up an episode)
data_as_numpy = np.stack((hit_poses, objects_hit), axis=1) 

np.save('empty_bin.npy', data_as_numpy)        
