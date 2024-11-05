import numpy as np
import pandas as pd


path = '/home/shaktis/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-10-30/17-32-20/dataset.pkl'
# path = '/home/shaktis/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-10-31/11-00-42/dataset.pkl'
df = pd.read_pickle(path)

# First sample no top higher off the table rectagular object target
data = [228, 231, 170, 104, 106, 104, 105, 107, 235, 234, 162, 108, 108, 108, 108, 105, 241, 233, 162, 108, 107, 110, 109, 109, 243, 235, 158, 108, 107, 110, 110, 111, 245, 238, 162, 110, 109, 109, 110, 113, 246, 241, 174, 111, 109, 112, 110, 112, 247, 242, 201, 111, 112, 112, 111, 113, 218, 224, 182, 113, 109, 109, 111, 113, 100, 101, 102, 102, 101, 100, 101, 104, 100, 102, 103, 105, 103, 102, 104, 105, 103, 102, 102, 104, 103, 106, 104, 107, 102, 103, 103, 102, 105, 107, 108, 105, 103, 102, 103, 104, 103, 106, 106, 107, 103, 102, 106, 105, 105, 104, 106, 106, 105, 105, 106, 105, 107, 110, 106, 106, 118, 109, 111, 111, 110, 110, 120, 126, 166, 160, 135, 113, 111, 112, 106, 109, 168, 161, 137, 112, 111, 109, 109, 107, 2459, 160, 138, 114, 110, 109, 108, 106, 164, 159, 138, 111, 109, 110, 110, 104, 1099, 158, 138, 109, 108, 107, 109, 106, 164, 158, 142, 110, 108, 108, 107, 106, 2401, 157, 146, 110, 111, 106, 105, 104, 215, 166, 164, 112, 109, 106, 106, 104, 113, 107, 109, 108, 108, 106, 103, 103, 111, 109, 110, 110, 108, 107, 107, 105, 114, 110, 110, 109, 110, 107, 109, 107, 113, 112, 109, 111, 109, 107, 109, 105, 114, 113, 113, 115, 110, 113, 111, 109, 113, 116, 112, 113, 110, 109, 109, 108, 116, 115, 117, 114, 114, 112, 110, 114, 124, 122, 119, 119, 117, 118, 119, 123]
filtered_data = [110, 110, 110, 105, 105, 107, 104, 106, 110, 110, 110, 107, 106, 108, 108, 106, 110, 110, 110, 107, 108, 107, 108, 108, 110, 110, 110, 108, 108, 108, 109, 111, 110, 110, 110, 112, 109, 111, 110, 112, 110, 110, 110, 112, 110, 110, 112, 109, 110, 110, 110, 113, 111, 112, 113, 111, 110, 110, 110, 113, 112, 110, 112, 113, 100, 99, 101, 101, 101, 100, 100, 102, 101, 101, 102, 103, 105, 103, 104, 103, 102, 102, 102, 103, 102, 104, 105, 105, 102, 103, 102, 102, 105, 105, 107, 104, 102, 102, 104, 104, 103, 102, 105, 105, 102, 103, 104, 105, 104, 102, 105, 106, 103, 106, 104, 105, 105, 105, 108, 107, 117, 110, 109, 110, 110, 110, 121, 126, 110, 110, 134, 112, 110, 110, 106, 108, 110, 110, 110, 113, 112, 110, 108, 107, 110, 110, 110, 114, 110, 108, 109, 105, 110, 110, 123, 112, 109, 109, 109, 106, 110, 110, 110, 112, 109, 108, 109, 106, 110, 110, 110, 112, 107, 107, 105, 105, 110, 110, 110, 109, 108, 107, 106, 106, 110, 110, 110, 111, 107, 105, 105, 104, 110, 107, 108, 107, 106, 105, 104, 103, 110, 108, 109, 108, 108, 105, 105, 106, 113, 111, 110, 110, 109, 107, 108, 105, 113, 111, 111, 109, 108, 107, 106, 107, 113, 113, 113, 112, 110, 110, 108, 107, 114, 113, 110, 110, 111, 110, 109, 108, 116, 115, 112, 114, 112, 112, 110, 111, 124, 123, 118, 118, 119, 117, 118, 121]
#Second sample no top close to the table rectagular object target
data = [211, 214, 121, 96, 97, 96, 96, 95, 220, 218, 117, 98, 96, 97, 97, 97, 228, 216, 116, 97, 96, 100, 99, 100, 230, 219, 120, 98, 97, 99, 102, 102, 235, 221, 119, 103, 100, 102, 103, 105, 215, 203, 116, 104, 103, 103, 103, 104, 168, 164, 119, 105, 103, 105, 107, 106, 130, 131, 118, 104, 105, 104, 106, 105, 94, 93, 95, 94, 94, 97, 97, 97, 94, 93, 94, 97, 98, 100, 98, 98, 95, 97, 96, 94, 95, 98, 98, 102, 94, 96, 96, 96, 98, 98, 100, 101, 96, 94, 98, 97, 98, 98, 103, 102, 95, 95, 97, 99, 97, 97, 101, 102, 95, 97, 96, 99, 100, 99, 102, 104, 95, 96, 96, 98, 98, 100, 104, 106, 116, 118, 113, 104, 102, 101, 100, 99, 1891, 144, 126, 105, 102, 103, 103, 98, 152, 152, 130, 106, 102, 99, 97, 95, 1055, 146, 126, 103, 103, 101, 104, 97, 149, 146, 124, 102, 100, 99, 100, 95, 2051, 147, 130, 100, 100, 98, 92, 95, 149, 145, 130, 102, 98, 97, 96, 94, 3459, 145, 3479, 102, 97, 95, 96, 94, 67, 78, 94, 98, 98, 98, 93, 95, 67, 80, 98, 101, 101, 98, 96, 98, 67, 80, 98, 101, 103, 100, 99, 99, 64, 82, 98, 103, 102, 100, 100, 99, 65, 86, 96, 104, 105, 105, 99, 102, 60, 81, 99, 102, 103, 101, 101, 98, 63, 81, 100, 108, 103, 105, 101, 100, 60, 78, 97, 101, 105, 104, 105, 104]
filtered_data = [102, 102, 119, 95, 96, 97, 94, 95, 102, 102, 116, 96, 96, 96, 96, 97, 102, 102, 116, 98, 98, 97, 99, 100, 102, 102, 119, 99, 97, 99, 100, 101, 102, 102, 120, 102, 102, 103, 101, 103, 102, 178, 118, 100, 101, 102, 102, 102, 149, 146, 110, 106, 104, 105, 105, 104, 125, 116, 117, 104, 103, 102, 104, 102, 92, 92, 94, 93, 95, 96, 96, 96, 93, 94, 96, 96, 97, 97, 98, 98, 94, 95, 95, 95, 96, 100, 98, 99, 94, 95, 95, 95, 97, 99, 99, 99, 94, 95, 96, 97, 97, 97, 100, 100, 94, 94, 96, 99, 96, 97, 99, 100, 95, 97, 98, 98, 99, 100, 102, 104, 95, 95, 94, 99, 96, 98, 104, 107, 115, 117, 111, 102, 101, 103, 99, 99, 129, 133, 117, 103, 102, 101, 100, 98, 139, 138, 122, 105, 102, 97, 100, 96, 138, 135, 118, 102, 100, 100, 99, 98, 138, 136, 118, 102, 100, 98, 98, 95, 136, 134, 121, 100, 98, 96, 94, 94, 136, 133, 122, 101, 98, 97, 96, 95, 112, 133, 129, 101, 95, 94, 94, 94, 74, 81, 96, 97, 98, 97, 93, 94, 74, 81, 97, 99, 100, 98, 95, 96, 73, 82, 98, 101, 100, 98, 97, 98, 72, 81, 97, 103, 101, 100, 97, 100, 73, 83, 99, 103, 107, 103, 98, 99, 71, 80, 98, 102, 101, 101, 101, 97, 73, 82, 98, 104, 103, 104, 102, 101, 71, 84, 99, 104, 104, 103, 104, 102]
# Third sample with top and higher off the table rectagular object target
data = [253, 249, 197, 113, 114, 114, 113, 114, 250, 251, 191, 116, 113, 115, 115, 114, 255, 250, 178, 117, 115, 118, 118, 119, 259, 251, 173, 118, 116, 121, 120, 121, 260, 255, 183, 122, 121, 119, 122, 121, 262, 257, 205, 125, 120, 120, 122, 123, 255, 257, 207, 125, 121, 122, 122, 124, 220, 223, 196, 124, 120, 124, 120, 122, 109, 109, 110, 110, 113, 111, 112, 113, 108, 111, 112, 113, 115, 114, 115, 115, 112, 112, 112, 113, 112, 115, 115, 119, 112, 114, 113, 113, 113, 117, 121, 116, 112, 114, 116, 115, 115, 116, 118, 119, 113, 113, 116, 116, 117, 115, 115, 121, 114, 116, 114, 117, 116, 120, 122, 123, 157, 158, 159, 164, 166, 189, 224, 223, 178, 173, 155, 123, 121, 120, 116, 120, 179, 174, 157, 122, 118, 120, 117, 115, 178, 173, 153, 124, 119, 119, 117, 114, 181, 171, 155, 120, 116, 118, 116, 115, 180, 170, 152, 118, 115, 117, 115, 114, 178, 169, 160, 120, 114, 113, 113, 113, 182, 165, 162, 119, 113, 113, 113, 112, 226, 174, 173, 120, 114, 111, 113, 111, 124, 118, 118, 116, 119, 113, 112, 112, 123, 121, 121, 118, 118, 116, 113, 111, 122, 122, 120, 121, 118, 116, 115, 115, 124, 124, 122, 119, 116, 118, 117, 114, 124, 122, 123, 122, 121, 118, 118, 117, 125, 124, 123, 122, 123, 117, 118, 116, 127, 128, 122, 124, 122, 120, 120, 119, 135, 146, 147, 143, 143, 140, 142, 141]
filtered_data = [121, 121, 121, 113, 112, 114, 112, 113, 121, 121, 121, 113, 115, 114, 115, 112, 121, 121, 121, 116, 115, 115, 114, 117, 121, 121, 121, 117, 114, 117, 118, 118, 121, 121, 121, 121, 117, 120, 118, 120, 121, 121, 121, 121, 119, 119, 120, 120, 121, 121, 121, 123, 122, 122, 121, 122, 121, 121, 121, 123, 123, 120, 122, 123, 109, 109, 110, 111, 111, 111, 112, 115, 109, 109, 112, 112, 113, 113, 114, 114, 112, 112, 113, 113, 113, 117, 114, 116, 112, 113, 114, 114, 112, 115, 118, 117, 113, 114, 114, 115, 115, 115, 118, 118, 113, 114, 116, 116, 115, 114, 119, 119, 114, 115, 116, 117, 117, 119, 120, 124, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 122, 120, 119, 116, 117, 121, 121, 121, 121, 120, 119, 115, 115, 121, 121, 144, 123, 117, 118, 115, 113, 121, 121, 121, 121, 116, 116, 116, 115, 121, 121, 147, 118, 115, 116, 115, 111, 121, 121, 121, 118, 115, 113, 111, 112, 121, 121, 121, 118, 114, 113, 114, 109, 121, 121, 121, 119, 112, 111, 110, 110, 123, 117, 117, 117, 114, 112, 112, 111, 122, 119, 118, 119, 119, 115, 112, 112, 123, 123, 121, 120, 116, 116, 115, 114, 125, 121, 120, 119, 117, 117, 115, 115, 124, 122, 121, 122, 120, 118, 116, 116, 124, 123, 123, 121, 118, 119, 117, 116, 126, 125, 123, 123, 122, 121, 118, 117, 137, 147, 145, 143, 139, 138, 139, 140]
# Forth sample with top and higher off the table and circular object
data = [100, 2263, 224, 227, 130, 113, 103, 105, 98, 125, 217, 223, 135, 115, 103, 99, 100, 1588, 218, 222, 139, 117, 106, 101, 100, 137, 217, 220, 140, 116, 104, 102, 100, 1045, 214, 222, 137, 120, 109, 106, 100, 125, 222, 228, 141, 119, 107, 103, 102, 121, 218, 230, 142, 121, 110, 103, 105, 114, 195, 210, 137, 115, 108, 106, 233, 220, 140, 109, 110, 109, 108, 114, 200, 110, 107, 97, 102, 101, 102, 99, 138, 105, 103, 97, 96, 101, 97, 100, 145, 104, 104, 97, 98, 98, 105, 101, 188, 113, 112, 108, 105, 107, 108, 108, 240, 183, 142, 112, 118, 120, 123, 120, 241, 223, 189, 139, 153, 442, 158, 157, 227, 170, 164, 155, 160, 156, 149, 151, 207, 222, 225, 166, 118, 108, 101, 102, 218, 252, 257, 197, 120, 111, 101, 97, 224, 260, 260, 214, 120, 111, 103, 99, 225, 256, 259, 220, 119, 110, 102, 99, 226, 259, 256, 220, 124, 110, 104, 97, 223, 254, 253, 228, 122, 109, 100, 95, 214, 251, 254, 228, 124, 113, 105, 94, 210, 249, 252, 242, 139, 111, 98, 95, 140, 148, 139, 134, 140, 150, 153, 140, 118, 111, 111, 113, 110, 112, 108, 106, 109, 110, 108, 106, 104, 104, 101, 101, 105, 107, 105, 102, 99, 98, 96, 100, 109, 108, 107, 106, 107, 100, 103, 101, 117, 115, 116, 113, 112, 115, 110, 108, 143, 136, 135, 138, 139, 134, 136, 142, 156, 201, 242, 249, 248, 248, 245, 243]
filtered_data = [98, 119, 121, 121, 129, 111, 103, 99, 96, 126, 121, 121, 135, 114, 103, 98, 99, 131, 121, 121, 138, 115, 104, 100, 97, 132, 121, 121, 140, 115, 106, 101, 99, 133, 121, 121, 138, 118, 109, 103, 99, 126, 121, 121, 139, 118, 109, 103, 102, 119, 121, 121, 141, 117, 110, 104, 105, 113, 121, 121, 136, 116, 108, 104, 121, 121, 140, 106, 110, 109, 109, 114, 121, 110, 107, 100, 103, 102, 100, 100, 136, 103, 104, 95, 97, 100, 99, 99, 143, 104, 105, 98, 97, 99, 102, 101, 121, 111, 111, 103, 105, 104, 108, 106, 121, 121, 141, 113, 115, 118, 120, 120, 121, 121, 160, 134, 151, 161, 160, 157, 121, 159, 155, 158, 157, 159, 147, 149, 121, 121, 121, 133, 115, 109, 101, 102, 121, 121, 121, 121, 118, 110, 101, 97, 121, 121, 121, 121, 119, 113, 101, 100, 121, 121, 121, 121, 120, 110, 102, 99, 121, 121, 121, 121, 121, 110, 101, 96, 121, 121, 121, 121, 123, 108, 101, 94, 121, 121, 121, 121, 125, 110, 103, 96, 121, 121, 121, 121, 137, 109, 99, 95, 138, 149, 136, 135, 139, 148, 150, 141, 119, 111, 111, 112, 113, 110, 107, 106, 107, 107, 107, 106, 104, 101, 102, 100, 106, 105, 106, 102, 97, 98, 99, 99, 108, 108, 108, 106, 105, 102, 103, 102, 115, 113, 114, 115, 113, 113, 111, 108, 140, 134, 135, 139, 136, 134, 134, 141, 157, 121, 121, 121, 121, 121, 121, 121]



hit_poses = df.loc[:,'Hits_Pose'].to_numpy() # 8, 64, 3

objects_hit = df.loc[:,'Object_hit'].to_numpy() # 512

corrected_hit_poses = []
corrected_objects_hit = []
indices = []
for i, sub_array in enumerate(hit_poses, 0):
    if sub_array.shape == (8, 64, 3):
        corrected_hit_poses.append(sub_array)
        corrected_objects_hit.append(objects_hit[i])
        indices.append(i)

corrected_hit_poses = np.stack(corrected_hit_poses) #.reshape(256, 3)
corrected_hit_poses = corrected_hit_poses.reshape(corrected_hit_poses.shape[0]*2, 256, 3)
corrected_objects_hit = np.stack(corrected_objects_hit)
corrected_objects_hit = corrected_objects_hit.reshape(corrected_objects_hit.shape[0]*2, 256)
import pdb; pdb.set_trace()
# each entry is a step contains array of [hit_poses, object_hit] (58 steps make up an episode)
data_as_numpy = np.stack((hit_poses, objects_hit), axis=1) 

np.save('empty_bin.npy', data_as_numpy)        
