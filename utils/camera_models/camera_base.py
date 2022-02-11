import numpy as np
from abc import ABC, abstractmethod


class Camera(ABC):
    """
    This Camera(*) is the camera base for every camera model implemented later
    """

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        assert isinstance(value, tuple)
        assert len(value) == 2
        self.__shape = value
        self.width = value[1]
        self.height = value[0]

    @shape.deleter
    def shape(self):
        self.__shape = (512, 512)
        self.width = self.__shape[1]
        self.height = self.__shape[0]

    def __init__(self):
        self.__shape = None
        self.width = None
        self.height = None
        self.K = None
        self.Kinv = None
        self.distortion_coeffs = None

    def get_shape(self):
        return self.height, self.width

    @staticmethod
    def sphere_normalization(pcl):
        """
        Projects a PLC array (3, n) or (4, n) onto sphere-surface; |r| = 1
        :return: sphere-normalized PCL
        """
        assert pcl.shape[0] in [3, 4]

        norm = np.linalg.norm(pcl[0:3, :], axis=0)
        pts_on_sphere = np.divide(pcl[0:3, :], norm)
        return pts_on_sphere

    @staticmethod
    def homogeneous_normalization(pcl):
        """
        Projects a PCL array (n, 3) or (n, 4) into homogeneous coordinates [x, y ,z] on a image plane z=1
        This functions also returns a mask of all points in front of the plane projection
        :param pcl: numpy array
        :return: normalized PLC
        """
        assert pcl.shape[1] in [3, 4], "PLC shape does not match. Expected (n, 3) or (n, 4). We got {}".format(
            pcl.shape)

        z = pcl[:, 2][:, None]
        pts_hm = pcl[:, 0:3] / z
        mask = z > 0
        return pts_hm, mask

    @abstractmethod
    def pixel2euclidean_space(self, pixels):
        raise NotImplementedError


    @staticmethod
    def get_color_array(color_map):
        """
        returns an array (3, n) of the colors in a equirectangular image
        """
        # ! This is the same solution by flatten every channel
        # color = np.zeros((3, self.deaful_pixel.shape[1]))
        # for idx in range(self.shape[0] * self.shape[1]):
        #     u, v = self.deaful_pixel[0, idx], self.deaful_pixel[1, idx]
        #     color[0, idx] = color_map[v, u, 0]
        #     color[1, idx] = color_map[v, u, 1]
        #     color[2, idx] = color_map[v, u, 2]
        if len(color_map.shape) > 2:
            return np.vstack((color_map[:, :, 0].flatten(),
                              color_map[:, :, 1].flatten(),
                              color_map[:, :, 2].flatten())).T
        else:
            return np.vstack((color_map.flatten(),
                              color_map.flatten(),
                              color_map.flatten())).T
