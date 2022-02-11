import numpy as np


class Layout:
    def __init__(self, frame):
        self.frame = frame
        # * 3D points projected in Euclidean
        self.boundary_floor = None
        self.boundary_ceiling = None

        # * Spherical PHI coordinate for each 3D point
        self.bearings_floor = None
        self.bearings_ceiling = None

        self.corners_id = None
        self.corner_coords = None

        # * Data stored in npy file (estimated data)
        self.ly_data = None
        
        self.gt_pose = self.frame.gt_pose
        self.est_pose = self.frame.est_pose
        

    def apply_vo_scale(self, scale):
        """
        Applies vo-scale to the LAYOUT
        """
        if self.reference == "WC_SO3":
            self.boundary_floor = self.boundary_floor + (
                scale / self.est_pose.vo_scale) * np.ones_like(
                    self.boundary_floor) * self.est_pose.t.reshape(3, 1)
            self.boundary_ceiling = self.boundary_ceiling + (
                scale / self.est_pose.vo_scale) * np.ones_like(
                    self.boundary_ceiling) * self.est_pose.t.reshape(3, 1)
            self.reference = "WC"

        elif self.reference == "WC":
            delta_scale = scale - self.est_pose.vo_scale
            self.boundary_floor = self.boundary_floor + (
                delta_scale / self.est_pose.vo_scale) * np.ones_like(
                    self.boundary_floor) * self.est_pose.t.reshape(3, 1)
            self.boundary_ceiling = self.boundary_ceiling + (
                delta_scale / self.est_pose.vo_scale) * np.ones_like(
                    self.boundary_ceiling) * self.est_pose.t.reshape(3, 1)

        self.est_pose.vo_scale = scale
        return True

    def apply_gt_scale(self, scale):
        self.boundary_floor = self.boundary_floor * scale
        self.boundary_ceiling = self.boundary_ceiling * scale
        self.est_pose.gt_scale = scale
