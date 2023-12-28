import numpy as np
from config import *


class PlaneEstimator:
    def __init__(self, data_manager):
        self.dt = data_manager

    def estimate_plane(self, pcl_boundaries):
        self.number_samples = None
        self.best_model = None
        self.best_evaluation = np.inf
        self.best_inliers = None
        self.best_inliers_num = 0
        self.counter_trials = 0

        assert pcl_boundaries.shape[0] == 3
        if pcl_boundaries.shape[1] < 50:
            return -1, False
        self.number_samples = pcl_boundaries.shape[1]
        random_state = np.random.RandomState(1000)

        max_trials = self.dt.cfg["plane_estimation.max_ransac_trials"]
        for self.counter_trials in range(max_trials):
            initial_inliers = random_state.choice(self.number_samples, 2, replace=False)

            sample_pts = pcl_boundaries[:, initial_inliers]

            plane_hat = self.compute_plane(sample_pts)

            # * Estimation
            sample_residuals = self.compute_residuals(
                plane=plane_hat,
                pcl_boundary=pcl_boundaries
            )

            # * Evaluation
            sample_evaluation = np.sum(sample_residuals ** 2)
            # print(np.linalg.norm(pcl_boundaries[:, -1] -  pcl_boundaries[:, 0]), pcl_boundaries.shape)
            # print("ratio", pcl_boundaries.shape[1])
            # * Selection
            # factor = self.pose.ly_size * self.cfg.params.factor_size
            # factor = (1024*factor/pcl_boundaries.shape[1]) * self.cfg.params.factor_size
            # print("ratio", factor)

            sample_inliers = abs(sample_residuals) < self.dt.cfg['plane_estimation.min_ransac_residuals']
            sample_inliers_num = np.sum(sample_inliers)

            # * Loop Control
            lc_1 = sample_inliers_num > self.best_inliers_num
            lc_2 = sample_inliers_num == self.best_inliers_num
            lc_3 = sample_evaluation < self.best_evaluation
            if lc_1 or (lc_2 and lc_3):
                # + Update best performance
                self.best_model = plane_hat.copy()
                self.best_inliers_num = sample_inliers_num.copy()
                self.best_evaluation = sample_evaluation.copy()
                self.best_inliers = sample_inliers.copy()

            if self.counter_trials >= self._dynamic_max_trials():
                break

        explained_ratio = self.best_inliers_num / self.number_samples
        if explained_ratio < self.dt.cfg["plane_estimation.min_inliers_ratio"]:
            return -1, False

        distance = np.linalg.norm(self.best_model)
        normal = self.best_model / distance

        from ..data_structure.plane import Plane

        # * Defining a PL feature
        pl = Plane.from_distance_and_normal(dist=distance, normal=normal, dt=self.dt)

        pl.boundary = pcl_boundaries[:, self.best_inliers]
        # pl.boundary = pl.get_sampled_boundary()
        # pcl = np.linalg.inv(self.pose.SE3_scaled())[0:3, :] @ extend_array_to_homogeneous(pl.boundary)
        # mask = np.linalg.norm(pcl, axis=0) < self.cfg.params.min_distance_to_plane*self.pose.scale
        # if mask.shape[0] < 0.1 * pl.boundary.shape[1]:
        #     return -1, False

        # pl.boundary = pl.boundary[:, mask]

        pl.explained_ratio = explained_ratio
        pl.compute_position()

        return pl, True

    @staticmethod
    def compute_plane(pcl_samples):
        vect_line = pcl_samples[:, 1] - pcl_samples[:, 0]
        vect_line /= np.linalg.norm(vect_line)
        normal = np.cross(vect_line, (0, 1, 0))
        distance = normal.dot(pcl_samples[:, 0])
        if distance < 0:
            normal *= -1
            distance *= -1

        return normal*distance

    def compute_residuals(self, plane, pcl_boundary):
        normal = plane / np.linalg.norm(plane)
        error = np.sum(normal.reshape(-1, 1) * pcl_boundary, axis=0) - np.linalg.norm(plane)
        return error

    def _dynamic_max_trials(self):
        if self.best_inliers_num == 0:
            return np.inf

        nom = 1 - self.dt.cfg['plane_estimation.ransac_prob_success']
        if nom == 0:
            return np.inf

        inlier_ratio = self.best_inliers_num / float(self.number_samples)
        denom = 1 - inlier_ratio ** 2
        if denom == 0:
            return 1
        elif denom == 1:
            return np.inf

        nom = np.log(nom)
        denom = np.log(denom)
        if denom == 0:
            return 0

        return int(np.ceil(nom / denom))
