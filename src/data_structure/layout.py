import numpy as np
from utils.enum import CAM_REF
from utils.geometry_utils import extend_array_to_homogeneous
from .ocg_patch import Patch


class Layout:
    @property
    def boundary(self):
        return self.__boundary

    @boundary.setter
    def boundary(self, value):
        if value is None:
            return
        self.__boundary = value
        self.is_initialized = False

    def __init__(self, data_manager):
        self.dt = data_manager

        self.boundary = None
        self.cam2boundary = None
        self.cam2boundary_mask = None
        
        self.bearings = None

        self.pose_gt = None
        self.pose_est = None
        self.idx = None

        # !Planes features defined in this LY
        self.list_pl = []
        self.list_corners = []

        self.patch = Patch(self.dt)
        self.patch.layout = self

        self.ly_data = None
        self.cam_ref = CAM_REF.CC
        self.height_ratio = 1

        self.is_initialized = True

    def initialize(self):
        """
        Initialize this layout
        """
        self.patch.initialize()

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

    def compute_cam2boundary(self):
        """
        Computes the horizontal distance for every boundary point w.r.t camera pose
        """
        if self.boundary is None:
            return
        if self.cam_ref == CAM_REF.WC_SO3 or self.cam_ref == CAM_REF.CC:
            # ! Boundary reference still in camera reference
            self.cam2boundary = np.linalg.norm(self.boundary[(0, 2), :], axis=0)
            self.cam2boundary_mask = self.cam2boundary.reshape([-1,]) < self.dt.cfg["room_id.clipped_ratio"]
        else:
            assert self.cam_ref == CAM_REF.WC
            pcl = np.linalg.inv(self.pose_est.SE3_scaled())[:3, :] @ extend_array_to_homogeneous(self.boundary)
            self.cam2boundary = np.linalg.norm(pcl[(0, 2), :], axis=0)
            self.cam2boundary_mask = self.cam2boundary.reshape([-1,]) < self.dt.cfg["room_id.clipped_ratio"]
            

    def get_clipped_boundary(self):
        """
        Returns a clipped boundary
        """

        assert self.cam_ref == CAM_REF.WC

        clipped_boundary = np.linalg.inv(self.pose_est.SE3_scaled())[:3, :] @ extend_array_to_homogeneous(self.boundary)
        mask = self.cam2boundary > self.dt.cfg["room_id.clipped_ratio"]
        radius = np.linalg.norm(clipped_boundary[(0, 2), :], axis=0)
        clipped_boundary[:, mask] = self.dt.cfg["room_id.clipped_ratio"] * clipped_boundary[:, mask]/radius[mask]

        return self.pose_est.SE3_scaled()[:3, :] @ extend_array_to_homogeneous(clipped_boundary)
