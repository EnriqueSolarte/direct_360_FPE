import numpy as np
from tqdm import tqdm

from .vo_scale_recover import VO_ScaleRecover
from .gt_scale_recover import GT_ScaleRecover


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
            "scale_recover.sliding_windows"]
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

        max_scale = 50 * self.dt.cfg["scale_recover.scale_step"]
        init_scale = -50 * self.dt.cfg["scale_recover.scale_step"]

        for iteration in tqdm(range(
                self.dt.cfg["scale_recover.max_loops_iterations"] *
                self.list_ly.__len__()), desc="Estimating VO-Scale..."):
            batch = self.get_next_batch()
            print(self.vo_scale, batch[-1].idx)
            self.apply_vo_scale(batch, self.vo_scale)

            scale = self.vo_scale_recover.estimate_scale(
                # !Estimation using coarse-to-fine approach and only the last planes
                list_ly=batch,
                max_scale=max_scale,
                initial_scale=init_scale,
                scale_step=self.dt.cfg["scale_recover.scale_step"],
                plot=True)
            self.update_vo_scale(self.vo_scale + scale)

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
        num_lys = int(self.dt.list_ly.__len__() * self.dt.cfg['scale_recover.lys_for_init'])
        self.list_ly = self.dt.list_ly[:num_lys]

        if not self.estimate_initial_vo_scale():
            raise ValueError("Initial vo-scale failed")

        if not self.estimate_vo_scale_by_batches():
            raise ValueError("Vo-scale recovering failed")

        return True

    def estimate_vo_and_gt_scale(self, list_ly):
        """
        Estimates VO-scale and GT-scale using entropy optimization and gt camera poses
        """

        self.estimate_vo_scale(list_ly)
        self.gt_scale = self.gt_scale_recover.estimate(self)

        return self.gt_scale, self.vo_scale
