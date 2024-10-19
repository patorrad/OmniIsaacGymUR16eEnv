import pickle as pkl
import pandas as pd
with open("/home/shaktis/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/outputs/2024-05-06/16-37-50/dataset.pkl", "rb") as f:
    object = pkl.load(f)
    
df = pd.DataFrame(object)
df.to_csv('test_data.csv')