import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.asarray(T)
    points = np.asarray(points)
    points = points.reshape((-1, 3))
    N = len(points)
    points = np.hstack((points, np.ones((N, 1))))
    transformed_points = (T @ points.T).T

    return np.squeeze(transformed_points[:, :3])