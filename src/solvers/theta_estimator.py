from config import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy


class GaussianModel_1D:
    def __init__(self, mean=None, sigma=None):
        self.mean = mean
        self.sigma = sigma
        self.samples = [(mean, sigma)]
        self.mean_hist = [mean]
        self.sigma_hist = [sigma]

    def add_measurement(self, mean, sigma):
        self.samples.append((mean, sigma))
        # for m, s in self.samples:
        #     self.update(m, s)
        self.update(mean, sigma)

    def update(self, mean, sigma):
        new_mean = (self.sigma**2)*mean + (sigma**2)*(self.mean)
        new_mean /= (sigma**2 + self.sigma**2)

        new_sigma = (sigma*self.sigma)**2
        new_sigma /= (sigma**2 + self.sigma**2)

        self.mean = new_mean
        self.sigma = np.sqrt(new_sigma)

        self.mean_hist.append(self.mean)
        self.sigma_hist.append(self.sigma)

    @staticmethod
    def visual_model(x, mean, sigma):
        fnt = np.exp(-0.5 * ((x - mean) / sigma)**2)
        fnt /= np.sum(fnt)
        # mask = 0.02
        # fnt[fnt > mask] = mask
        return fnt

    def eval(self, x):
        fnt = np.exp(-0.5 * ((x - self.mean) / self.sigma)**2)
        fnt *= (1/(self.sigma * np.sqrt(2*np.pi)))
        return fnt

    def force_positive(self):
        if self.mean < 0:
            self.mean = 2*np.pi + self.mean

    def force_pi2pi_domain(self):
        if self.mean > np.pi:
            self.mean = self.mean - 2*np.pi


class ThetaEstimator:
    def __init__(self, data_manager):
        self.dt = data_manager
        
    def estimate_from_list_pl(self, list_pl, room_center, measurements=None, prune_out=True):
        if measurements is None:
            measurements = []
        for pl in tqdm(list_pl, desc="...Filtering Orientation"):
            # pl.was_used_for_orientation = True

            # ! Only planes over 0.9 explainability ratio are used
            if pl.explained_ratio < self.dt.cfg.get("theta_estimation.min_explainability_ratio", 0.8):
                continue

            dist = pl.distance - pl.pose.t.dot(pl.normal)
            orientation = pl.get_orientation_wrt_room_pose_ref(room_center)

            if measurements.__len__() < 1:
                measurements = [GaussianModel_1D(
                    mean=orientation,
                    sigma=np.radians(self.dt.cfg.get('theta_estimation.initial_sigma')))]
                continue

            closest_m = [np.abs(orientation - m.mean) < np.radians(self.dt.cfg.get('theta_estimation.max_clearance')) for m in measurements]

            if np.sum(closest_m) == 0:
                measurements.append(GaussianModel_1D(
                    mean=orientation, 
                    sigma=np.radians(self.dt.cfg.get('theta_estimation.initial_sigma'))))
                continue

            for i, msk1 in zip(range(closest_m.__len__()), closest_m):
                if msk1:
                    obs_sigma = np.radians(self.dt.cfg.get('theta_estimation.common_sigma')) + self.dt.cfg.get('theta_estimation.penalty')*dist
                    measurements[i].add_measurement(mean=orientation, sigma=obs_sigma)
        if prune_out:
            return self.prone_out(measurements)
        else:
            return measurements

    def prone_out(self, measurements):
        # ! Forcing only positive angles
        [z.force_positive() for z in measurements]
        # ! Pruning out multiple estimations
        zhat = []
        for i, z in enumerate(measurements):
            if zhat.__len__() < 1:
                zhat.append(z)
                continue
            closest_m = [np.abs(z.mean - m_.mean) < np.radians(self.dt.cfg.get('theta_estimation.max_clearance')) for m_ in zhat]

            if np.sum(closest_m) == 0:
                zhat.append(z)
            else:
                idx = np.argmax(closest_m)
                zhat[idx].update(mean=z.mean, sigma=z.sigma)

        # # ! Pruning out at circular boundary
        # for i, z in enumerate(zhat):
        #     closest_m = [np.abs(2*np.pi + z.mean - m_.mean) < np.radians(self.cfg.params.max_theta_clearance) for m_ in zhat]
        #     if np.sum(closest_m) > 0:
        #         idx = np.argmax(closest_m)
        #         zhat[idx].update(mean=2*np.pi + z.mean, sigma=z.sigma)
        #         if zhat[idx].mean > np.pi:
        #             zhat[idx].mean = zhat[idx].mean - 2*np.pi
        #         if zhat[idx].mean < -np.pi:
        #             zhat[idx].mean = zhat[idx].mean + 2*np.pi
        #         zhat.pop(i)

        measurements = []
        for z in zhat:
            circular_CW = [
                abs(z.mean - (m.mean - 2*np.pi)) < np.radians(self.dt.cfg.get('theta_estimation.max_clearance'))
                for m in zhat
            ]
            if np.sum(circular_CW) == 0:
                measurements.append(z)
            else:
                if np.sum(circular_CW) > 0:
                    idx = np.argmax(circular_CW)
                    measurements.append(z)
                    measurements[-1].update(mean=zhat[idx].mean - 2*np.pi, sigma=zhat[idx].sigma)

                else:
                    circular_CCW = [
                        abs(z.mean - (m.mean + 2*np.pi)) < np.radians(self.dt.cfg.get('theta_estimation.max_clearance'))
                        for m in zhat
                    ]
                    if np.sum(circular_CCW) > 0:
                        idx = np.argmax(circular_CW)
                        measurements.append(z)
                        measurements[-1].update(mean=zhat[idx].mean + 2*np.pi, sigma=zhat[idx].sigma)

        [z.force_pi2pi_domain() for z in zhat]
        return measurements
        # smallest_m = [m_.sigma < self.cfg.BF_SIGMA_THETA_COMMON_MEASUREMENT for m_ in zhat]
        # prune_zhat = []
        # for i, msk1 in zip(range(smallest_m.__len__()), smallest_m):
        #     if msk1:
        #         prune_zhat.append(zhat[i])

        # return zhat
        # return [z for z in zhat if z.sigma < np.radians(self.cfg.params.initial_sigma*self.cfg.params.min_sigma_reduction)]
