import numpy as np
from tqdm import tqdm

from .vo_scale_recover import VO_ScaleRecover
from .gt_scale_recover import GT_ScaleRecover


class ScaleRecover:
    def __init__(self, data_manager):
        self.dt = data_manager
        self.cfg = data_manager.cfg
        self.hist_escales = []
        self.hist_vo_scales = []

        self.internal_idx = 0
        self.vo_scale_recover = VO_ScaleRecover(self.cfg)
        self.gt_scale_recover = GT_ScaleRecover(self.cfg)
        self.vo_scale = 1
        self.gt_scale = 1
        self.list_ly = None

    def get_next_batch(self):
        batch = self.list_ly[self.internal_idx:self.internal_idx +
                             self.cfg["scale_recover.sliding_windows"]]
        self.internal_idx = self.internal_idx + self.cfg[
            "scale_recover.sliding_windows"]
        # print(f"Reading Batch LY.idx's: {batch[0].idx} - {batch[-1].idx}")
        return batch

    @staticmethod
    def apply_vo_scale(list_ly, scale):
        [ly.apply_vo_scale(scale) for ly in list_ly]
        print("VO-Scale {0:.3f} was successfully applied.".format(scale))

    @staticmethod
    def apply_gt_scale(list_ly, scale):
        [ly.apply_gt_scale(scale) for ly in list_ly]
        print("GT-Scale {0:.3f} was successfully applied.".format(scale))

    def fully_vo_scale_recovery_by_batches(self):
        """
        Estimates the vo-scale by linear search in a coarse-to-fine manner
        """

        max_scale = 50 * self.cfg["scale_recover.scale_step"]
        init_scale = -50 * self.cfg["scale_recover.scale_step"]

        for iteration in tqdm(range(
                self.cfg["scale_recover.max_loops_iterations"] *
                self.list_ly.__len__()), desc="Recovering VO-Scale..."):
            batch = self.get_next_batch()

            self.apply_vo_scale(batch, self.vo_scale)

            scale = self.vo_scale_recover.estimate_scale(
                # !Estimation using coarse-to-fine approach and only the last planes
                list_ly=batch,
                max_scale=max_scale,
                initial_scale=init_scale,
                scale_step=self.cfg["scale_recover.scale_step"],
                plot=False)
            self.update_vo_scale(self.vo_scale + scale)

            if self.internal_idx + self.cfg[
                    "scale_recover.sliding_windows"] >= self.list_ly.__len__():
                self.internal_idx = 0

            if iteration > self.list_ly.__len__() * 0.2:
                if np.std(self.hist_vo_scales[-100:]
                          ) < self.cfg["scale_recover.min_scale_variance"]:
                    break

        self.apply_vo_scale(self.list_ly, self.vo_scale)

        return True

    def recover_initial_vo_scale(self, batch=None):
        """
        Recovers an initial vo-scale (initial guess), which later will be used as
        a pivot to refine the global vo-scale
        """

        if batch is None:
            # ! Number of frames for initial scale recovering
            self.internal_idx = self.cfg["scale_recover.initial_batch"]
            batch = self.list_ly[:self.internal_idx]

        # > We need good LYs for initialize
        scale = self.vo_scale_recover.estimate_by_searching_in_range(
            list_ly=batch,
            max_scale=self.cfg["scale_recover.max_vo_scale"],
            initial_scale=self.cfg["scale_recover.min_vo_scale"],
            scale_step=self.cfg["scale_recover.scale_step"],
            plot=False)
        if scale < self.cfg["scale_recover.min_vo_scale"]:
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

    def fully_vo_scale_estimation(self, list_ly):
        """
        Recovers VO-scale by Entropy Minimization [360-DFPE]
        https://arxiv.org/abs/2112.06180 
        """
        self.list_ly = list_ly
        if not self.recover_initial_vo_scale():
            raise ValueError("Initial vo-scale failed")

        if not self.fully_vo_scale_recovery_by_batches():
            raise ValueError("Vo-scale recovering failed")

    def fully_estimate_vo_and_gt_scale(self, list_ly):
        """
        Estimates VO-scale and GT-scale using entropy optimization and gt camera poses
        """

        self.fully_vo_scale_estimation(list_ly)
        self.gt_scale = self.gt_scale_recover.estimate(self)

        return self.gt_scale, self.vo_scale