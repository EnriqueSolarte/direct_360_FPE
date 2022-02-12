import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm

from config import *


class GT_ScaleRecover:
    def __init__(self, data_manager):
        self.dt = data_manager
        self.gt_poses = None
        self.gt_scale = None

    def estimate_initial_guess(self):
        t = [
            np.linalg.norm(gt) / np.linalg.norm(st)
            for gt, st in zip(self.gt_poses, self.st_poses)
            if np.linalg.norm(st) > 0.01
        ]
        return np.nanmean(t)

    def scale_loss(self, scale):
        error = np.array([
            np.linalg.norm(gt) - scale * np.linalg.norm(st)
            for gt, st in zip(self.gt_poses, self.st_poses)
        ])

        return np.sum(error**2)

    def estimate(self, scale_recover):
        self.gt_poses = [
            gt[0:3, 3].reshape((3, 1))
            for gt in scale_recover.dataset.gt.kf_poses
        ]
        self.st_poses = [
            st[0:3, 3].reshape((3, 1))
            for st in scale_recover.dataset.estimated_poses
        ]

        assert self.gt_poses.__len__() == self.st_poses.__len__()

        initial_guess = self.estimate_initial_guess()

        res = minimize(self.scale_loss, initial_guess, method='nelder-mead')

        # plot_list_pcl((np.hstack(self.gt_poses), res.x[0]*np.hstack(self.st_poses)), (1, 1))

        self.gt_scale = res.x[0] / scale_recover.vo_scale
        return self.gt_scale
