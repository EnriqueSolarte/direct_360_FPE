import numpy as np
from scipy.spatial import cKDTree
import time
from utils.camera_models.camera_base import Camera


class Sphere(Camera):

    def define_k_matrix(self):
        """
        Defines the K & Kinv matrices (affine transformations) to project from pixel (u, v)
        to (theta, phi)
        """
        # ! Kinv (u,v) --> (theta, phi)
        # ! K (theta, phi) --> (u, v)
        self.Kinv = np.asarray(
            (2 * np.pi / self.width, 0, -np.pi,
             0, -np.pi / self.height, np.pi / 2,
             0, 0, 1)).reshape(3, 3)
        self.K = np.linalg.inv(self.Kinv)

    @property
    def shape(self):
        return self.__shape

    def get_shape(self):
        return self.shape

    @shape.setter
    def shape(self, value):
        assert len(value) == 2
        # print(value)
        self.__shape = value
        self.width = value[1]
        self.height = value[0]
        self.grid = self.equirectangular_grid(value)
        self.grid2 = np.squeeze(np.stack([self.grid[0].reshape(-1, 1), self.grid[1].reshape(-1, 1)], axis=1))
        self.vgrid = self.vector_grid(self.grid)
        self.px_grid = self.pixel_grid()
        self.tree = cKDTree(self.grid2, balanced_tree=False)
        self.define_k_matrix()
        self.define_default_bearings(value)

    @shape.deleter
    def shape(self):
        self.__shape = (512, 1024)
        self.width = self.__shape[1]
        self.height = self.__shape[0]

    def __init__(self, shape=(512, 1024)):
        super(Sphere, self).__init__()
        # self.camera_equirectangular = Equirectangular(shape=shape)
        self.sphere_ratio = 1
        self.shape = shape

    def define_default_bearings(self, shape):
        h, w = shape
        u = np.linspace(0, w - 1, w).astype(int)
        v = np.linspace(0, h - 1, h).astype(int)
        uu, vv = np.meshgrid(u, v)
        self.deafult_pixel = np.vstack((uu.flatten(), vv.flatten(), np.ones((w * h,)))).astype(np.int)
        self.default_spherical = self.Kinv.dot(self.deafult_pixel)
        self.default_bearings = self.sphere2bearing(self.default_spherical)

    def pixel2euclidean_space(self, pixels):
        """
        From a set of equirectangular pixel, this returns the normalized vector representation for those pixels.
        Such normalization corresponds to the vectors which lay in sphere ratio=1
        """
        assert pixels.shape[1] in (2, 3), "pixels parameter out of shape (n, 3) or (n, 2). We got {}".format(
            pixels.shape)
        # TODO This procedure is repeated either in Pinhole as UnifiedModel, and spherical too. Maybe we could centralize it into CameraBase
        # Getting of pixels as [u, v, 1]
        # ! We add 0.5 to compensate pixels digitization (quatization)
        pixel_coords = np.hstack([pixels[:, 0:2]+0.5, np.ones((len(pixels[:, 0]), 1))])

        sphere_coords = np.dot(pixel_coords, self.Kinv.T)
        return self.sphere2bearing(sphere_coords.T)

    def sphere2bearing(self, sphere_coords):
        sin_coords = np.sin(sphere_coords[0:2, :])
        cos_coords = np.cos(sphere_coords[0:2, :])

        return np.vstack(
            [
                sin_coords[0, :] * cos_coords[1, :],
                - sin_coords[1, :],
                cos_coords[0, :] * cos_coords[1, :]
            ])

    def equirectangular2sphere(self, image):
        """
        Projects an equirectangular image (h, w, c) onto a unit surface sphere
        :param image: Equirectangular image. image.shape == self.get_shape()
        :return: pcl, color: both in (n, 3)
        """
        if len(image.shape) > 2:
            h, w, c = image.shape
        else:
            h, w = image.shape

        assert (h, w) == self.get_shape()

        tilde_theta, tilde_phi = self.equirectangular_grid((h, w))

        y = -np.sin(tilde_phi) * self.sphere_ratio
        x = np.sin(tilde_theta) * np.cos(tilde_phi) * self.sphere_ratio
        z = np.cos(tilde_theta) * np.cos(tilde_phi) * self.sphere_ratio

        pcl = np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])

        if len(image.shape) > 2:
            color = image.reshape((-1, 3)).astype(np.float) / 255.0
        else:
            color = image.reshape((-1, 1)).astype(np.float) / 255.0
            color = np.concatenate([color, color, color], axis=1)

        return pcl, color

    def pixel_grid(self, shape=None):

        if shape is None:
            shape = self.shape

        phi = np.linspace(0, shape[0] - 1, shape[0]).astype(np.int)
        theta = np.linspace(0, shape[1] - 1, shape[1]).astype(np.int)
        tilde_theta, tilde_phi = np.meshgrid(theta, phi)

        return np.hstack([tilde_theta.reshape(-1, 1), tilde_phi.reshape(-1, 1)]).astype(np.int)

    def vector_grid(self, grid):
        tilde_theta = grid[0]
        tilde_phi = grid[1]

        y = -np.sin(tilde_phi) * self.sphere_ratio
        x = np.sin(tilde_theta) * np.cos(tilde_phi) * self.sphere_ratio
        z = np.cos(tilde_theta) * np.cos(tilde_phi) * self.sphere_ratio

        return np.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])

    def vector_grid2equirect_grid(self, vector_grid):

        # Check if vector_grid elements are unit vectors
        assert np.isclose(np.linalg.norm(vector_grid[np.random.randint(0, len(vector_grid[:, 1])), :]), 1)

        theta = np.arctan2(vector_grid[:, 0], vector_grid[:, 2])
        phi = np.arcsin(-vector_grid[:, 1])

        return np.stack([theta, phi], axis=1)

    def equirectangular_grid(self, shape=None):
        if shape is None:
            shape = self.shape

        phi_step = np.pi / shape[0]
        theta_step = 2 * np.pi / shape[1]
        phi = np.linspace(np.pi / 2, -np.pi / 2 + phi_step, shape[0]-1)
        theta = np.linspace(-np.pi, np.pi - theta_step, shape[1]-1)
        tilde_theta, tilde_phi = np.meshgrid(theta, phi)

        return tilde_theta, tilde_phi

    def project_pcl(self, color_map, depth_map, scaler=1):
        h, w = color_map.shape[0], color_map.shape[1]
        assert (h, w) == depth_map.shape, "color frame must be == to depth map"
        color_pixels = self.get_color_array(color_map=color_map) / 255
        mask = depth_map.flatten() > 0
        pcl = self.default_bearings[:, mask] * (
            1 / scaler) * depth_map.flatten()[mask]
        return [pcl, color_pixels[mask, :].T]

    def back_project_pcl(self, xyz):
        """
        Projects XYZ array into uv coord
        """
        assert xyz.shape[0] == 3
        
        xyz_n = xyz / np.linalg.norm(xyz, axis=0, keepdims=True)
        
        normXZ = np.linalg.norm(xyz[(0, 2), :], axis=0, keepdims=True)
        
        phi_coord = -np.arcsin(xyz_n[1, :])
        theta_coord = np.sign(xyz[0, :]) * np.arccos(xyz[2, :]/normXZ)
        
        #  theta = math.acos(xyz[2] / normXZ)

        u = np.nan_to_num(np.clip(np.round((0.5 * theta_coord/np.pi + 0.5) * self.shape[1]),0 , self.shape[1]-1))
        v = np.nan_to_num(np.clip(np.round(-(phi_coord/np.pi - 0.5) * self.shape[0]), 0, self.shape[0] -1))
        
        return np.vstack((u, v)).astype(int)