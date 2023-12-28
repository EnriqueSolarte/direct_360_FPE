
import numpy as np
from utils.enum import CAM_REF

class Plane:
    def __init__(self, data_manager):
        self.dt = data_manager

        self.normal = None
        self.orientation = None
        self.distance = None
        self.line_dir = None

        self.boundary = None
        self.pose = None

        # ! quality features
        self.theta_error = np.inf
        self.theta_uncertainty = np.inf
        self.distance2cam = np.inf
        self.explained_ratio = None

    @classmethod
    def from_distance_and_orientation(cls, distance, theta, dt):
        pl = cls(dt)
        pl.distance = distance
        pl.set_orientation_from_theta(theta)
        return pl

    @classmethod
    def from_distance_and_normal(cls, dist, normal, dt):
        pl = cls(dt)
        pl.distance = dist
        pl.normal = normal / np.linalg.norm(normal)
        if pl.distance < 0:
            pl.distance *= -1
            pl.normal *= -1

        pl.set_orientation_and_line_dir()
        return pl
    
    
    def get_distance_wrt_room_pose_ref(self, room_center):
        return self.distance - room_center.dot(self.normal)

    def get_orientation_wrt_room_pose_ref(self, room_center):
        if self.get_distance_wrt_room_pose_ref(room_center) < 0:
            return self.orientation-np.pi
        else:
            return self.orientation

    def set_orientation_from_theta(self, theta):
        self.orientation = theta
        self.normal = np.array((np.sin(theta), 0, np.cos(theta))).reshape(-1,)
        self.line_dir = np.cross(np.array((0, 1, 0)), self.normal)

    def compute_position(self):
        self.position = np.median(self.boundary, axis=1)

    def compute_distance2cam(self):
        self.distance2cam = np.abs(self.distance - self.pose.t.dot(self.normal))

    def set_theta_estimation(self, theta_z):
        """
        Sets the QLT features for this PLANE based on an orientation (Gaussian estimation)
        Returns whether this plane is candidate or not
        """
        # ! Only if theta_z if off a theta-clearance of the orientation for this plane
        if np.abs(self.get_orientation_wrt_room_pose_ref() - theta_z.mean) > np.radians(self.dt.params.max_theta_clearance):
            return False

        self.theta_error = self.get_orientation_wrt_room_pose_ref() - theta_z.mean
        self.orientation -= self.theta_error
        self.theta_uncertainty = theta_z.sigma
        self.was_used_for_orientation = True

        if abs(self.theta_error) < np.radians(self.dt.params.max_theta_error_allowed):
            self.isCandidate = True
            return True
        else:
            self.isCandidate = False
            return False

    def set_orientation_and_line_dir(self):
        self.orientation = np.arctan2(self.normal[0], self.normal[2])
        self.line_dir = np.cross((0, 1, 0), self.normal)

    def apply_vo_scale(self, scale):
        raise NotImplementedError()

    def project_points_into_pl(self, points):
        """
        Projects perpendicularly any set of point/s (3, n) into the this plane
        """
        point_dot_normal = np.sum(points.reshape(3, -1) * self.normal.reshape(3, 1), axis=0)
        projected_points = points.reshape(3, -1) + (self.distance - point_dot_normal) * self.normal.reshape(3, 1)
        return projected_points

    def set_boundary(self, boundary):
        self.boundary = self.project_points_into_pl(boundary)
