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
        return np.hstack([
            obj.boundary[:, obj.cam2boundary_mask] + (scale / obj.pose_est.vo_scale) *
            np.ones_like(obj.boundary[:, obj.cam2boundary_mask]) * obj.pose_est.t.reshape(3, 1)
            for obj in self.list_ly
        ])

    def reset_all(self):
        self.hist_entropy = []
        self.hist_scale = []
        self.hist_best_scale = []

    def estimate_scale(self,
                       list_ly,
                       max_scale,
                       min_scale,
                       plot=False):

        self.list_ly = list_ly
        # ! Scale already applied  to the passed list_ly
        vo_scale = list_ly[0].pose_est.vo_scale
        scale = min_scale
        self.reset_all()

        best_scale_hist = []

        invalid_estimation = False
        for c2f in range(self.dt.cfg["scale_recover.coarse_levels"]):
            scale = min_scale
            self.reset_all()
            scale_step = (max_scale - min_scale) / 10
            while True:
                # ! Applying scale
                pcl = self.apply_vo_scale(scale=scale)
                # ! Computing Entropy
                h, ocg_map = compute_entropy_from_pcl(
                    pcl=pcl, grid_size=self.dt.cfg["scale_recover.grid_size"], 
                    return_ocg_map=True)

                # print(ocg_map.shape, ocg_map.size)
                if ocg_map.size > np.prod(self.dt.cfg["scale_recover.max_ocg_map_size"]):
                    print("The scene content invalid regions...")
                    invalid_estimation = True
                    break

                if plot and self.hist_entropy.__len__() > 0:
                    # ocg_map, xedges, zedges = get_ocg_map(
                    #     pcl=pcl,
                    #     grid_size=self.dt.cfg["scale_recover.grid_size"])
                    ocg_map = ocg_map / np.max(ocg_map)
                    fig = plt.figure("Optimization", figsize=(10, 4))
                    plt.clf()
                    ax1 = fig.add_subplot(121)
                    ax1.clear()
                    ax1.set_title("OCG map @ scale:{0:0.4f}".format(vo_scale + scale))
                    ax1.imshow(ocg_map)

                    ax2 = fig.add_subplot(122)
                    ax2.clear()
                    ax2.set_title("Entropy Optimization")
                    ax2.plot(self.hist_scale, self.hist_entropy)
                    idx_min = np.argmin(self.hist_entropy)
                    best_scale = self.hist_scale[idx_min]
                    ax2.scatter(
                        best_scale + vo_scale,
                        np.min(self.hist_entropy),
                        label="Best increment-scale:{0:0.2f}\nLowest H:{1:0.2f}".format(
                            best_scale, np.min(self.hist_entropy)),
                        c="red")
                    ax2.set_xlabel("Increment Scale")
                    ax2.set_ylabel("Entropy")
                    ax2.legend()
                    ax2.grid()
                    plt.draw()
                    plt.waitforbuttonpress(0.01)
                    # if wait is None:
                    #     wait = input("\nPress enter: >>>>")

                self.hist_entropy.append(h)
                self.hist_scale.append(scale)
                scale += scale_step
                if vo_scale + scale < self.dt.cfg["scale_recover.min_vo_scale"]:
                    invalid_estimation = True
                    break

                if scale > max_scale or scale < min_scale:
                    idx_min = np.argmin(self.hist_entropy)
                    best_scale_hist.append(self.hist_scale[idx_min])

                    min_scale = np.mean(best_scale_hist) - scale_step * 2
                    max_scale = np.mean(best_scale_hist) + scale_step * 2
                    break

            if invalid_estimation:
                # ! Forcing to return a zero relative scale
                best_scale_hist = [0]
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
                             zedges=None,
                             return_ocg_map=False):
    ocg_map, _, _ = get_ocg_map(pcl=pcl,
                                grid_size=grid_size,
                                weights=weights,
                                xedges=xedges,
                                zedges=zedges)
    if return_ocg_map:
        return compute_entropy_from_ocg_map(ocg_map), ocg_map
    return compute_entropy_from_ocg_map(ocg_map)


def compute_entropy_from_ocg_map(ocg_map):
    mask = ocg_map > 0
    # * Entropy
    H = np.sum(-ocg_map[mask] * np.log2(ocg_map[mask]))
    return H
