import pickle
import numpy as np

with open(r'../calibration_parameters/cameraMatrix.pkl', 'rb') as f:
    camera_matrix = pickle.load(f)

np.save('../calibration_parameters/camera_matrix.npy', camera_matrix)

with open(r'../calibration_parameters/dist.pkl', 'rb') as f:
    dist_matrix = pickle.load(f)

np.save('../calibration_parameters/dist.npy', dist_matrix)

