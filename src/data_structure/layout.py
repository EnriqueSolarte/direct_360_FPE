import numpy as np 
from utils.enum import CAM_REF
class Layout:

    def __init__(self, data_manager):
        self.dt = data_manager

        self.boundary = None
        self.bearings = None

        # self.corners_id = None
        # self.corner_coords = Non
        self.pose_gt = None
        self.pose_est = None
        self.idx = None

        # !Planes features defined in this LY
        self.list_pl = []
        self.list_corners = []

        # >> used in room identifier
        self.central_pose = None

        self.ly_data = None
        self.cam_ref = None
        self.height_ratio = 1

    def apply_vo_scale(self, scale):

        if self.cam_ref == CAM_REF.WC_SO3:
            self.boundary = self.boundary + (scale/self.pose_est.vo_scale) * np.ones_like(self.boundary) * self.pose_est.t.reshape(3, 1)
            self.cam_ref = CAM_REF.WC

        elif self.cam_ref == CAM_REF.WC:
            delta_scale = scale - self.pose_est.vo_scale
            self.boundary = self.boundary + (delta_scale/self.pose_est.vo_scale) * np.ones_like(self.boundary) * self.pose_est.t.reshape(3, 1)

        if self.list_pl is not None:
            if self.list_pl.__len__() > 0:
                [pl.apply_scale(scale) for pl in self.list_pl if pl is not None]

        if self.list_corners.__len__() > 0:
            [cr.apply_scale(scale) for cr in self.list_corners]

        self.pose_est.vo_scale = scale

        return True

    def apply_gt_scale(self, scale):
        self.boundary = self.boundary*scale
        self.pose_est.gt_scale = scale

    def estimate_height_ratio(self):
        """
        Estimates the height ratio that describes the distance ratio of camera-floor over the
        camera-ceiling distance. This information is important to recover the 3D
        structure of the predicted Layout
        """
        floor = np.abs(self.ly_data[1, :])
        ceiling = np.abs(self.ly_data[0, :])

        ceiling[ceiling > np.radians(80)] = np.radians(80)
        ceiling[ceiling < np.radians(5)] = np.radians(5)
        floor[floor > np.radians(80)] = np.radians(80)
        floor[floor < np.radians(5)] = np.radians(5)

        self.height_ratio = np.mean(np.tan(ceiling)/np.tan(floor))

