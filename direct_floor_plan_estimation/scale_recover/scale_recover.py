import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from utils.ocg_utils import compute_uv_bins
from .vo_scale_recover import VO_ScaleRecover
from .gt_scale_recover import GT_ScaleRecover
import yaml


class ScaleRecover:
    def __init__(self, data_manager):
        self.dt = data_manager
        self.hist_escales = []
        self.hist_vo_scales = []

        self.internal_idx = 0
        self.vo_scale_recover = VO_ScaleRecover(self.dt)
        self.gt_scale_recover = GT_ScaleRecover(self.dt)
        self.vo_scale = 1
        self.gt_scale = 1
        self.list_ly = None

    def get_next_batch(self):
        batch = self.list_ly[self.internal_idx:self.internal_idx +
                             self.dt.cfg["scale_recover.sliding_windows"]]
        self.internal_idx = self.internal_idx + self.dt.cfg[
            "scale_recover.sliding_windows"]//2
        # print(f"Reading Batch LY.idx's: {batch[0].idx} - {batch[-1].idx}")
        return batch

    @staticmethod
    def apply_vo_scale(list_ly, scale):
        [ly.apply_vo_scale(scale) for ly in list_ly]
        # print("VO-Scale {0:.3f} was successfully applied to layouts [{1}-{2}].".format(
        #     scale,
        #     list_ly[0].idx,
        #     list_ly[-1].idx
        # ))

    @staticmethod
    def apply_gt_scale(list_ly, scale):
        [ly.apply_gt_scale(scale) for ly in list_ly]
        # print("GT-Scale {0:.3f} was successfully applied to layouts [{1}-{2}].".format(
        #     scale,
        #     list_ly[0].idx,
        #     list_ly[-1].idx
        # ))

    def estimate_vo_scale_by_batches(self):
        """
        Estimates the vo-scale by linear search in a coarse-to-fine manner
        """

        for iteration in tqdm(range(
                self.dt.cfg["scale_recover.max_loops_iterations"] *
                self.list_ly.__len__()), desc="Estimating VO-Scale..."):

            batch = self.get_next_batch()
            # print(self.vo_scale, batch[-1].idx)
            self.apply_vo_scale(batch, self.vo_scale)
            # ! Since every batch already has a vo-scale computed by a previous step
            # ! the estimate scale is a relative increment scale
            # ! i.e., scale = 0 keeps the already the estimated scale

            max_scale = 50 * self.dt.cfg["scale_recover.scale_step"]
            min_scale = -50 * self.dt.cfg["scale_recover.scale_step"]

            scale = self.vo_scale_recover.estimate_scale(
                # !Estimation using coarse-to-fine approach and only the last planes
                list_ly=batch,
                max_scale=max_scale,
                min_scale=min_scale,
                plot=False)
            # print(scale + self.vo_scale)
            self.update_vo_scale(scale + self.vo_scale)

            if self.internal_idx + self.dt.cfg[
                    "scale_recover.sliding_windows"] >= self.list_ly.__len__():
                self.internal_idx = 0

            if iteration > self.list_ly.__len__() * 0.2:
                if np.std(self.hist_vo_scales[-100:]
                          ) < self.dt.cfg["scale_recover.min_scale_variance"]:
                    break

        self.apply_vo_scale(self.list_ly, self.vo_scale)

        return True

    def estimate_initial_vo_scale(self, batch=None):
        """
        Recovers an initial vo-scale (initial guess), which later will be used as
        a pivot to refine the global vo-scale
        """

        if batch is None:
            # ! Number of frames for initial scale recovering
            self.internal_idx = self.dt.cfg["scale_recover.initial_batch"]
            batch = self.list_ly[:self.internal_idx]

        # > We need good LYs for initialize
        scale = self.vo_scale_recover.estimate_by_searching_in_range(
            list_ly=batch,
            max_scale=self.dt.cfg["scale_recover.max_vo_scale"],
            initial_scale=self.dt.cfg["scale_recover.min_vo_scale"],
            scale_step=self.dt.cfg["scale_recover.scale_step"],
            plot=False)
        if scale < self.dt.cfg["scale_recover.min_vo_scale"]:
            return False

        self.update_vo_scale(scale)

        self.apply_vo_scale(batch, self.vo_scale)
        return True

    def update_vo_scale(self, scale):
        """
        Sets an estimated scale to the system
        """
        self.hist_escales.append(scale)
        self.vo_scale = np.mean(self.hist_escales)
        self.hist_vo_scales.append(self.vo_scale)

    def estimate_vo_scale(self):
        """
        Recovers VO-scale by Entropy Minimization
        """
        # ! Using loaded layout in data_manager
        num_lys = int(self.dt.list_ly.__len__() * self.dt.cfg['scale_recover.lys_for_warmup'])
        self.list_ly = self.dt.list_ly[:num_lys]

        if not self.estimate_initial_vo_scale():
            raise ValueError("Initial vo-scale failed")

        if not self.estimate_vo_scale_by_batches():
            raise ValueError("Vo-scale recovering failed")

        return True

    def estimate_vo_and_gt_scale(self):
        """
        Estimates VO-scale and GT-scale using entropy optimization and gt camera poses
        """

        self.estimate_vo_scale()
        if self.dt.cfg['data.use_gt_poses']:
            self.gt_scale = self.gt_scale_recover.estimate(self)

            # * We are assuming that by estimating vo and gt scale is because we want to
            # * apply GT scale
            [ly.apply_gt_scale(self.gt_scale) for ly in self.dt.list_ly]

        return self.gt_scale, self.vo_scale

    def save_estimation(self, output_dir):
        """
        Saves the scale estimation into a directory
        """
        # ! Save image LY aligned
        plt.figure("GT-scale recover")
        plt.clf()
        plt.title("GT-scale recover - VO-Scale:{0:0.3f} - GT-Scale:{1:0.3f}".format(
            self.vo_scale, self.gt_scale))
        pcl_ly = np.hstack([
            ly.boundary for ly in self.dt.list_ly
            if ly.cam2boundary.max() < 20
        ])

        ubins, vbins = compute_uv_bins(pcl_ly, self.dt.cfg['scale_recover.grid_size'], padding=10)
        x = pcl_ly[0, :]
        z = pcl_ly[2, :]
        grid, _, _ = np.histogram2d(x, z, bins=(ubins, vbins))
        mask = grid > 20
        grid[mask] = 20
        grid = grid/20
        plt.imshow(grid)
        plt.draw()
        plt.savefig(os.path.join(output_dir, "scale_recovery.jpg"), bbox_inches='tight')

        # ! Save json data
        data = dict()
        for key in self.dt.cfg.keys():
            if "scale_recover" in key or "data" in key:
                data[key] = self.dt.cfg[key]

        data["data.vo_scale"] = float(self.vo_scale)
        data["data.gt_scale"] = float(self.gt_scale)

        filename = os.path.join(output_dir, "scale_recovery.yaml")
        with open(filename, "w") as file:
            yaml.dump(data, file)
