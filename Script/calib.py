import pickle
import numpy as np

with open(r'cameraMatrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)

np.save('dist.npy', camera_matrix)

with open(r'dist.pkl', 'rb') as f:
    dist_matrix = pickle.load(f)

np.save('dist.npy', dist_matrix)

