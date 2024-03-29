import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from geometry_perception_utils.vispy_utils import plot_color_plc
import os


class VO_ScaleRecover:
    def __init__(self, cfg):
        self.cfg = cfg
        self.hist_entropy = []
        self.hist_scale = []
        [setattr(self, key, val) for key, val in cfg.items()]
        
        
    def reset_all(self):
        self.hist_entropy = []
        self.hist_scale = []
        self.hist_best_scale = []
        self.hist_best_entropy = []
        
    def estimate_scale(self,
                       list_ly,
                       vis_plot_dir=None):
        self.reset_all()
        for batch in tqdm(range(0, list_ly.__len__(), self.sliding_windows//2), desc="Running VO Scale Recovery"):
            _list_ly = list_ly[batch:batch + self.sliding_windows]
            # * setting min and max scale as initial scale value
            min_scale = self.min_vo_scale
            max_scale = self.max_vo_scale

            # for c2f in tqdm(self.coarse_levels, desc="Coarse-to-Fine estimation"):
            for c2f in self.coarse_levels:
                scale = min_scale
                self.hist_entropy = []
                self.hist_scale = []
                scale_step = c2f
                # list_ly x boundary_floor.shape[1]
                while True:
                    # ! Applying scale
                    pcl = apply_vo_scale(scale=scale, list_ly=_list_ly)
                    pcl = mask_noisy_points(_list_ly)
                    # ! Computing Entropy
                    h, ocg_map = compute_entropy_from_pcl(
                        pcl=pcl, grid_size=self.grid_size, 
                        return_ocg_map=True)

                    # print(ocg_map.shape, ocg_map.size)
                    if ocg_map.size > np.prod(self.max_ocg_map_size):
                        raise ValueError("Ocg map size is too large")
                    
                    self.hist_entropy.append(h)
                    self.hist_scale.append(scale)
                
                    scale += scale_step
                    
                    if scale >= max_scale or scale < min_scale:
                        idx_min = np.argmin(self.hist_entropy)
                        # self.hist_best_scale.append(self.hist_scale[idx_min])
                        local_best_scale = self.hist_scale[idx_min]
                        min_scale = np.min(local_best_scale) - scale_step
                        if min_scale < self.min_vo_scale:
                            min_scale = self.min_vo_scale
                        max_scale = np.max(local_best_scale) + scale_step
                        if max_scale > self.max_vo_scale:
                            max_scale = self.max_vo_scale
                        break
            
            # * Considering only scale at the last c2f level
            self.hist_best_scale.append(local_best_scale)
            self.hist_best_entropy.append(self.hist_entropy[idx_min])
            
        best_vo_scale = self.compute_best_vo_scale()
        apply_vo_scale(scale=best_vo_scale, list_ly=list_ly)
        if vis_plot_dir is not None:
            fn = os.path.join(vis_plot_dir, f"scale_recover.png")                
            plot_scale_recover_vis(list_ly, best_vo_scale, self.grid_size, fn, hist_scale=self.hist_best_scale, hist_entropy=self.hist_best_entropy)
            
    def compute_best_vo_scale(self):
        # idx_min = np.argmin(self.hist_best_entropy)
        # best_vo_scale = self.hist_best_scale[idx_min]
        if self.hist_best_entropy.__len__() == 1:
            return self.hist_best_scale[0]
        if self.cfg.get("best_h_func") is None:
            w = np.max(self.hist_best_entropy) - np.array(self.hist_best_entropy)
            w = w / np.sum(w) 
            return np.sum(np.array(self.hist_best_scale) * w)
        if self.cfg.get("best_h_func") == 'median':
            return np.median(self.hist_best_scale)
        elif self.cfg.get("best_h_func") == 'mean':
            return np.mean(self.hist_best_scale)
        elif self.cfg.get("best_h_func") == 'max':
            return np.max(self.hist_best_scale)
        elif self.cfg.get("best_h_func") == 'min':
            return np.min(self.hist_best_scale)
        elif q in self.cfg.get("best_h_func"):
            quantile = self.cfg.get("best_h_func").split("q")
            return np.quantile(self.hist_best_scale, float(quantile[-1]))
        
    
def get_ocg_map(pcl,
                grid_size=None,
                weights=None,
                xedges=None,
                zedges=None,
                padding=10):
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
    grid = grid/np.sum(grid)
    # mask = grid > 20
    # grid[mask] = 20
    # grid = grid / 20

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


def apply_vo_scale(scale, list_ly):
    [ly.pose.set_vo_scale(scale) for ly in list_ly]
    [ly.recompute_data() for ly in list_ly]
    return np.hstack([ly.boundary_floor for ly in list_ly])


def plot_scale_recover_vis(list_ly, scale, grid_size, fn, hist_scale=None, hist_entropy=None):
    
    if hist_scale is None:
        hist_scale = list(np.ones((10)) * scale)
    if hist_entropy is None:
        hist_entropy = list(np.ones((10)) * scale)
        
    # ! Computing Entropy Plot
    apply_vo_scale(scale=scale, list_ly=list_ly)
    pcl = mask_noisy_points(list_ly)
    _, ocg_map = compute_entropy_from_pcl(
    pcl=pcl, grid_size=grid_size, 
    return_ocg_map=True)    
    ocg_map = ocg_map / np.max(ocg_map)
    
    fig = plt.figure("Optimization", figsize=(10, 4))
    plt.clf()
    ax1 = fig.add_subplot(121)
    ax1.clear()
    ax1.set_title("OCG map @ opt scale:{0:0.4f}".format(scale))
    ax1.imshow(ocg_map)
    ax2 = fig.add_subplot(122)
    ax2.clear()
    ax2.set_title("Entropy Optimization")
    ax2.plot((scale, scale), (np.min(hist_entropy), np.max(hist_entropy)), c="r", label="Optimal scale")
    ax2.scatter(hist_scale, hist_entropy, 
                s=[np.exp(h) for h in hist_entropy], alpha=0.5)
    ax2.set_xlabel("Best scale history")
    ax2.set_ylabel("Best entropy history")
    ax2.legend()
    ax2.grid()
    plt.draw()
    plt.waitforbuttonpress(0.01)
    plt.savefig(fn)
    plt.close(fig)

def mask_noisy_points(list_ly):
    distances = np.hstack([ly.cam2boundary for ly in list_ly])
    mask_points = distances < np.median(distances) * 2
    pcl = np.hstack([ly.boundary_floor for ly in list_ly])
    pcl = pcl[:, mask_points]
    return pcl
