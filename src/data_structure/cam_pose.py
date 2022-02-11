from utils.geometry_utils import isRotationMatrix
import numpy as np

class CamPose:
    def __init__(self, data_manager, pose):
        self.dt = data_manager
        self.SE3 = pose
        self.scale = self.dt.cfg["vo_scale"]
        self.idx = None

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, value):
        assert value > 0
        self.__scale = value
        # self.SE3[0:3, 3] = self.SE3[0:3, 3] * value

    @property
    def SE3(self):
        return self.__pose

    @SE3.setter
    def SE3(self, value):
        assert value.shape == (4, 4)
        self.__pose = value
        self.rot = value[0:3, 0:3]
        self.t = value[0:3, 3]

    @property
    def rot(self):
        return self.__rot

    @rot.setter
    def rot(self, value):
        assert isRotationMatrix(value)
        self.__rot = value

    @property
    def t(self):
        return self.__t * self.scale

    @t.setter
    def t(self, value):
        assert value.reshape(3,).shape == (3,)
        self.__t = value.reshape(3,)

    def SE3_scaled(self):
        m = np.eye(4)
        m[0:3, 0:3] = self.rot
        m[0:3, 3] = self.t
        return m

    def apply_gt_scale(self, scale):
        self.scale = self.scale * scale
