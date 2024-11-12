import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Load the .pkl file
with open('/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-08/16-07-16/dataset.pkl', 'rb') as file:
    data = pickle.load(file)

# Ensure data is a numpy array if not already
data = np.array(data)

# Split the data into training and testing sets (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the training data to a new .pkl file
with open('/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-08/16-07-16/train_data.pkl', 'wb') as train_file:
    pickle.dump(train_data, train_file)

# Save the testing data to a new .pkl file
with open('/home/paolo/Documents/OmniIsaacGymUR16eEnv/omniisaacgymenvs/scripts/outputs/2024-11-08/16-07-16/test_data.pkl', 'wb') as test_file:
    pickle.dump(test_data, test_file)

print("Training and testing data saved as 'train_data.pkl' and 'test_data.pkl'.")
