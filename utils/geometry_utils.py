import numpy as np


def isRotationMatrix(R):
    """
    Checks if a matrix is a valid rotation matrix.    
    """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def get_bearings_from_phi_coords(phi_coords):
    """
    Returns 3D bearing vectors (on the unite sphere) from phi_coords 
    """
    W = phi_coords.__len__()
    u = np.linspace(0, W - 1, W)
    bearings_theta = (2 * np.pi * u / W) - np.pi
    bearings_y = -np.sin(phi_coords)
    bearings_x = np.cos(phi_coords) * np.sin(bearings_theta)
    bearings_z = np.cos(phi_coords) * np.cos(bearings_theta)
    return np.vstack((bearings_x, bearings_y, bearings_z))


def extend_array_to_homogeneous(array):
    """
    Returns the homogeneous form of a vector by attaching
    a unit vector as additional dimensions
    Parameters
    ----------
    array of (3, n) or (2, n)
    Returns (4, n) or (3, n)
    -------
    """
    try:
        assert array.shape[0] in (2, 3, 4)
        dim, samples = array.shape
        return np.vstack((array, np.ones((1, samples))))

    except:
        assert array.shape[1] in (2, 3, 4)
        array = array.T
        dim, samples = array.shape
        return np.vstack((array, np.ones((1, samples)))).T


def extend_vector_to_homogeneous_transf(vector):
    """
    Creates a homogeneous transformation (4, 4) given a vector R3
    :param vector: vector R3 (3, 1) or (4, 1)
    :return: Homogeneous transformation (4, 4)
    """
    T = np.eye(4)
    if vector.__class__.__name__ == "dict":
        T[0, 3] = vector["x"]
        T[1, 3] = vector["y"]
        T[2, 3] = vector["z"]
    elif type(vector) == np.array:
        T[0:3, 3] = vector[0:3, 0]
    else:
        T[0:3, 3] = vector[0:3]
    return T

    