import numpy as np
import os
import sys


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def hip_center_and_normalize(pl):
    data = pl.copy()
    coordinates = data[:, :, :, :3]
    visibility = data[:, :, :, 3:4]
    presence = data[:, :, :, 4:5]

    left_hip = coordinates[:, :, 23, :]
    right_hip = coordinates[:, :, 24, :]
    hip_center = (left_hip + right_hip) / 2
    # hip_center_z=hip_center[:,:,2]

    # batch , n_frames, n_landmarks, 5

    centered_coords = coordinates.copy()

    hip_center = hip_center[:, :, None, :]
    centered_coords -= hip_center

    x_values = centered_coords[:, :, :, 0]
    y_values = centered_coords[:, :, :, 1]
    z_values = centered_coords[:, :, :, 2]

    x_min = np.min(x_values, keepdims=True, axis=2)
    x_max = np.max(x_values, keepdims=True, axis=2)
    x_size = np.abs(x_max - x_min)
    x_max_size = np.max(x_size, keepdims=True, axis=1)

    y_min = np.min(y_values, keepdims=True, axis=2)
    y_max = np.max(y_values, keepdims=True, axis=2)
    y_size = np.abs(y_max - y_min)
    y_max_size = np.max(y_size, keepdims=True, axis=1)

    z_min = np.min(z_values, keepdims=True, axis=2)
    z_max = np.max(z_values, keepdims=True, axis=2)
    z_size = np.abs(z_max - z_min)
    z_max_size = np.max(z_size, keepdims=True, axis=1)

    epsilon = 1e-8

    x_values /= (x_max_size + epsilon)
    y_values /= (y_max_size + epsilon)
    z_values /= (z_max_size + epsilon)

    centered_coords[:, :, :, 0] = x_values
    centered_coords[:, :, :, 1] = y_values
    centered_coords[:, :, :, 2] = z_values

    processed_data = np.concatenate([centered_coords, visibility, presence], axis=3)

    return processed_data