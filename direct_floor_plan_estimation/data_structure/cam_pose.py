import numpy as np
from utils.geometry_utils import isRotationMatrix


class CamPose:
    def __init__(self, data_manager, pose):
        self.dt = data_manager
        self.SE3 = pose
        self.vo_scale = self.dt.cfg.get("vo_scale", 1)
        self.gt_scale = self.dt.cfg.get("gt_scale", 1)
        self.idx = None

    @property
    def vo_scale(self):
        return self.__vo_scale

    @vo_scale.setter
    def vo_scale(self, value):
        assert value > 0
        self.__vo_scale = value

    @property
    def gt_scale(self):
        return self.__gt_scale

    @gt_scale.setter
    def gt_scale(self, value):
        assert value > 0
        self.__gt_scale = value

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
        return self.__t * self.vo_scale * self.gt_scale

    @t.setter
    def t(self, value):
        assert value.reshape(3, ).shape == (3, )
        self.__t = value.reshape(3, )

    def SE3_scaled(self):
        m = np.eye(4)
        m[0:3, 0:3] = self.rot
        m[0:3, 3] = self.t
        return m
