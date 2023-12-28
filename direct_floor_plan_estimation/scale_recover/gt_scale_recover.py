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
        # ! Getting list of GT and estimated poses
        self.gt_poses = [
            ly.pose_gt.t for ly in scale_recover.dt.list_ly
        ]
        self.st_poses = [
            ly.pose.t for ly in scale_recover.dt.list_ly
        ]

        assert self.gt_poses.__len__() == self.st_poses.__len__()

        initial_guess = self.estimate_initial_guess()

        res = minimize(self.scale_loss, initial_guess, method='nelder-mead')

        # from utils.visualization.vispy_utils import plot_list_pcl
        # plot_list_pcl((np.vstack(self.gt_poses).T, res.x[0]*np.vstack(self.st_poses).T))

        self.gt_scale = res.x[0]
        return self.gt_scale
