import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import data_manager


class VO_ScaleRecover:
    def __init__(self, data_manager):
        self.dt = data_manager
        self.hist_entropy = []
        self.hist_scale = []

    def apply_vo_scale(self, scale):
        # ! note that every LY is referenced as WC_SO3_REF (only Rot was applied)
        return np.hstack([
            obj.boundary + (scale / obj.pose_est.vo_scale) *
            np.ones_like(obj.boundary) * obj.pose_est.t.reshape(3, 1)
            for obj in self.list_ly
        ])

    def reset_all(self):
        self.hist_entropy = []
        self.hist_scale = []
        self.hist_best_scale = []

    def estimate_scale(self,
                       list_ly,
                       max_scale,
                       initial_scale,
                       scale_step,
                       plot=False):

        self.list_ly = list_ly
        scale = initial_scale
        self.reset_all()

        best_scale_hist = []
        # for c2f in tqdm(range(self.config["scale_recover.coarse_levels"]),
        #                 desc="...Estimating Scale"):
        for c2f in range(self.dt.cfg["scale_recover.coarse_levels"]):
            scale = initial_scale
            self.reset_all()
            scale_step = (max_scale - initial_scale) / 10
            while True:
                # ! Applying scale
                pcl = self.apply_vo_scale(scale=scale)
                # ! Computing Entropy
                h = compute_entropy_from_pcl(
                    pcl=pcl, grid_size=self.dt.cfg["scale_recover.grid_size"])

                if plot and self.hist_entropy.__len__() > 0:
                    grid, xedges, zedges = get_ocg_map(
                        pcl=pcl,
                        grid_size=self.dt.cfg["scale_recover.grid_size"])
                    grid = grid / np.max(grid)
                    fig = plt.figure("Optimization", figsize=(10, 4))
                    ax1 = fig.add_subplot(121)
                    ax1.clear()
                    ax1.set_title("OCG map @ scale:{0:0.4f}".format(scale))
                    ax1.imshow(grid)

                    ax2 = fig.add_subplot(122)
                    ax2.clear()
                    ax2.set_title("Entropy Optimization")
                    ax2.plot(self.hist_scale, self.hist_entropy)
                    idx_min = np.argmin(self.hist_entropy)
                    best_scale = self.hist_scale[idx_min]
                    ax2.scatter(
                        best_scale,
                        np.min(self.hist_entropy),
                        label="Best Scale:{0:0.2f}\nLowest H:{1:0.2f}".format(
                            best_scale, np.min(self.hist_entropy)),
                        c="red")
                    ax2.set_xlabel("Scale")
                    ax2.set_ylabel("Entropy")
                    ax2.grid()
                    plt.draw()
                    plt.waitforbuttonpress(0.01)
                    # if wait is None:
                    #     wait = input("\nPress enter: >>>>")

                self.hist_entropy.append(h)
                self.hist_scale.append(scale)
                scale += scale_step
                if scale > max_scale:
                    if np.max(self.hist_entropy) - np.min(
                            self.hist_entropy
                    ) < self.dt.cfg["scale_recover.min_scale_variance"]:
                        best_scale_hist.append(0)
                    else:
                        idx_min = np.argmin(self.hist_entropy)
                        best_scale_hist.append(self.hist_scale[idx_min])

                    initial_scale = np.mean(best_scale_hist) - scale_step * 2
                    max_scale = np.mean(best_scale_hist) + scale_step * 2
                    break

        return np.mean(best_scale_hist)

    def estimate_by_searching_in_range(self,
                                       list_ly,
                                       max_scale,
                                       initial_scale,
                                       scale_step,
                                       plot=False):
        assert np.random.choice(list_ly, size=1)[0].cam_ref == "WC_SO3", "WC_SO3 references is need for Initial Guess in Scale Recovering"
        self.list_ly = list_ly

        scale = initial_scale
        self.reset_all()

        while True:
            # ! Applying scale
            pcl = self.apply_vo_scale(scale=scale)
            # ! Computing Entropy
            h = compute_entropy_from_pcl(
                pcl=pcl, grid_size=self.dt.cfg["scale_recover.grid_size"])

            if plot and self.hist_entropy.__len__() > 0:
                grid, xedges, zedges = get_ocg_map(
                    pcl=pcl, grid_size=self.dt.cfg["scale_recover.grid_size"])
                grid = grid / np.max(grid)
                fig = plt.figure("Optimization", figsize=(10, 4))
                ax1 = fig.add_subplot(121)
                ax1.clear()
                ax1.set_title("OCG map @ scale:{0:0.4f}".format(scale))
                ax1.imshow(grid)

                ax2 = fig.add_subplot(122)
                ax2.clear()
                ax2.set_title("Entropy Optimization")
                ax2.plot(self.hist_scale, self.hist_entropy)
                idx_min = np.argmin(self.hist_entropy)
                best_scale = self.hist_scale[idx_min]
                ax2.scatter(
                    best_scale,
                    np.min(self.hist_entropy),
                    label="Best Scale:{0:0.2f}\nLowest H:{1:0.2f}".format(
                        best_scale, np.min(self.hist_entropy)),
                    c="red")
                ax2.set_xlabel("Scale")
                ax2.set_ylabel("Entropy")
                ax2.grid()
                plt.draw()
                plt.waitforbuttonpress(0.01)
                # if wait is None:
                #     wait = input("\nPress enter: >>>>")

            self.hist_entropy.append(h)
            self.hist_scale.append(scale)
            scale += scale_step
            if scale > max_scale:
                idx_min = np.argmin(self.hist_entropy)
                best_scale = self.hist_scale[idx_min]
                break
        return best_scale


def get_ocg_map(pcl,
                grid_size=None,
                weights=None,
                xedges=None,
                zedges=None,
                padding=100):
    x = pcl[0, :]
    z = pcl[2, :]

    if (xedges is None) or (zedges is None):
        xedges = np.mgrid[np.min(x) - padding * grid_size:np.max(x) +
                          padding * grid_size:grid_size]
        zedges = np.mgrid[np.min(z) - padding * grid_size:np.max(z) +
                          padding * grid_size:grid_size]

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights /= np.max(weights)

    grid, xedges, zedges = np.histogram2d(x,
                                          z,
                                          weights=1 / weights,
                                          bins=(xedges, zedges))
    # grid = grid/np.sum(grid)
    mask = grid > 20
    grid[mask] = 20
    grid = grid / 20

    return grid, xedges, zedges


def compute_entropy_from_pcl(pcl,
                             grid_size,
                             weights=None,
                             xedges=None,
                             zedges=None):
    grid, _, _ = get_ocg_map(pcl=pcl,
                             grid_size=grid_size,
                             weights=weights,
                             xedges=xedges,
                             zedges=zedges)
    return compute_entropy_from_ocg_map(grid)


def compute_entropy_from_ocg_map(ocg_map):
    mask = ocg_map > 0
    # * Entropy
    H = np.sum(-ocg_map[mask] * np.log2(ocg_map[mask]))
    return H
