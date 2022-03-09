import numpy as np
import cv2
from skimage.transform import resize
from utils.geometry_utils import extend_array_to_homogeneous
from utils.ocg_utils import compute_uv_bins, project_xyz_to_uv
from src.data_structure import layout


class OCGPatches:
    """
    This class handles multiples Patches (defined from every Layout) by 
    aggregating, combining, and pruning them out properly.
    """
    @property
    def u_bins(self):
        return self.__u_bins

    @u_bins.setter
    def u_bins(self, value):
        if value is None:
            return
        self.__u_bins = value
        self.W = int(self.__u_bins.size - 1)
        self.uv_ref[0] = self.__u_bins[0]

    @property
    def v_bins(self):
        return self.__v_bins

    @v_bins.setter
    def v_bins(self, value):
        if value is None:
            return
        self.__v_bins = value
        self.H = int(self.v_bins.size - 1)
        self.uv_ref[1] = self.__v_bins[0]

    def __init__(self, data_manager):
        self.dt = data_manager
        self.v_bins = None
        self.u_bins = None
        self.ocg_map = None
        self.list_patches = []
        self.dynamic_bins = True
        self.is_initialized = False
        self.H, self.W = None, None
        self.uv_ref = [None, None]
        self.grid_size = self.dt.cfg['room_id.grid_size']

    def project_xyz_to_uv(self, xyz_points):

        x_cell_u = [np.argmin(abs(p - self.u_bins-self.grid_size*0.5)) % self.get_shape()[1]
                    for p in xyz_points[0, :]]

        if xyz_points.shape[0] == 2:
            # xyz_points: (2, N)
            z_cell_v = [np.argmin(abs(p - self.v_bins-self.grid_size*0.5)) % self.get_shape()[0]
                        for p in xyz_points[1, :]]
        else:
            # xyz_points: (3, N)
            z_cell_v = [np.argmin(abs(p - self.v_bins-self.grid_size*0.5)) % self.get_shape()[0]
                        for p in xyz_points[2, :]]
        # ! this potentially can change the order of the points
        # if unique:
        #     return np.unique(np.vstack((x_cell_u, z_cell_v)), axis=0), True
        return np.vstack((x_cell_u, z_cell_v))

    def project_uv_to_xyz(self, uv_points):
        if uv_points.dtype != np.int32:
            uv_points = uv_points.astype(np.int32)
        v_size, u_size = self.get_shape()
        assert np.all(uv_points[0, :] <= u_size), f"{np.max(uv_points[0, :])}, {u_size}"
        assert np.all(uv_points[1, :] <= v_size), f"{np.max(uv_points[1, :])}, {v_size}"
        xs = self.u_bins[uv_points[0, :]] + self.grid_size * 0.5
        ys = np.zeros_like(xs)
        zs = self.v_bins[uv_points[1, :]] + self.grid_size * 0.5
        return np.vstack([xs, ys, zs])

    def project_xyz_points_to_hist(self, xyz_points):
        x = xyz_points[0, :]
        if xyz_points.shape[0] == 2:
            # (x, z)
            z = xyz_points[1, :]
        else:
            # (x, y, z)
            z = xyz_points[2, :]
        grid, _, _ = np.histogram2d(z, x, bins=(self.v_bins, self.u_bins))

        return grid

    def initialize(self, patch):
        """
        Initializes the OCGPatch class by using the passed patch
        """
        assert patch.is_initialized, "Passed patch mst be initialized first..."

        # self.ocg_map = np.expand_dims(patch.ocg_map, 2)
        self.ocg_map = np.copy(patch.ocg_map)
        self.list_patches.append(patch)

        self.u_bins, self.v_bins = patch.u_bins, patch.v_bins
        self.is_initialized = True
        return self.is_initialized

    def update_bins(self):
        """
        Updates bins based on the patches registered in list_patches
        """
        bins = np.vstack([(patch.u_bins[0], patch.v_bins[0],
                           patch.u_bins[-1], patch.v_bins[-1])
                          for patch in self.list_patches
                          ])

        min_points = np.min(bins, axis=0)[0:2]
        max_points = np.max(bins, axis=0)[2:4]
        self.u_bins = np.mgrid[min_points[0]:max_points[0]+self.grid_size: self.grid_size]
        self.v_bins = np.mgrid[min_points[1]:max_points[1]+self.grid_size: self.grid_size]
        return self.u_bins, self.v_bins

    def update_ocg_map(self, binary_map=False):
        """
        Updates the OCG map by aggregating layers of registered Patches (layers, H, W)
        """
        H, W = self.get_shape()
        self.ocg_map = np.zeros((self.list_patches.__len__(), H, W))
        for idx, patch in enumerate(self.list_patches):
            h, w = patch.H, patch.W
            uv = project_xyz_to_uv(
                xyz_points=np.array((patch.uv_ref[0], 0, patch.uv_ref[1])).reshape((3, 1)),
                u_bins=self.u_bins,
                v_bins=self.v_bins
            ).squeeze()

            self.ocg_map[idx, uv[1]:uv[1]+h, uv[0]:uv[0]+w] = patch.ocg_map

        if binary_map:
            self.ocg_map[self.ocg_map > self.dt.cfg.get("room_id.ocg_threshold", 0.5)] = 1
            self.ocg_map = self.ocg_map.astype(np.int)

    def update_ocg_map2(self):
        """
        Updates the OCG map by aggregating Patches using a temporal weight constraint
        """
        H, W = self.get_shape()
        self.ocg_map = np.zeros((H, W))
        temporal_weight = np.linspace(1, 0, self.list_patches.__len__())
        for idx, patch in enumerate(self.list_patches):
            h, w = patch.H, patch.W
            uv = project_xyz_to_uv(
                xyz_points=np.array((patch.uv_ref[0], 0, patch.uv_ref[1])).reshape((3, 1)),
                u_bins=self.u_bins,
                v_bins=self.v_bins
            ).squeeze()

            if self.dt.cfg.get("room_id.temporal_weighting", True):
                self.ocg_map[uv[1]:uv[1]+h, uv[0]:uv[0]+w] += patch.ocg_map * temporal_weight[idx]
            else:
                self.ocg_map[uv[1]:uv[1]+h, uv[0]:uv[0]+w] += patch.ocg_map
            
             # ! Adding non-isotropic Normalization 
            if self.dt.cfg.get("room_id.non_isotropic_normalization", False):
                self.ocg_map = self.ocg_map/self.ocg_map.max()
                    
        # ! Adding isotropic Normalization 
        if not self.dt.cfg.get("room_id.non_isotropic_normalization", False):
            self.ocg_map = self.ocg_map/self.ocg_map.max()
        

            # TODO: combine/ add update_ocg_map() as binary map option

    def add_patch(self, patch):
        """
        Adds a new Patch and Updated the bins definitions
        """
        self.list_patches.append(patch)
        self.update_bins()

    def get_shape(self):
        return (self.v_bins.size-1, self.u_bins.size-1)

    def get_mask(self):
        ocg_map = self.ocg_map
        ocg_map = ocg_map / np.max(ocg_map)
        mask = ocg_map > self.dt.cfg.get("room_id.ocg_threshold", 0.5)
        return mask

    def resize(self, scale):
        v_bins = self.v_bins
        u_bins = self.u_bins
        v_bins = self.resize_bins(self.v_bins, scale)
        u_bins = self.resize_bins(self.u_bins, scale)

        height, width = self.get_shape()
        height = round(height * scale)
        width = round(width * scale)

        self.v_bins = v_bins
        self.u_bins = u_bins
        self.ocg_map = resize(self.ocg_map, (height, width))
        self.grid_size = u_bins[1] - u_bins[0]

    def resize_bins(self, bins, scale):
        num = round((len(bins) - 1) * scale) + 1
        new_bins = np.linspace(bins[0], bins[-1], num)
        return new_bins


class Patch:
    """
    This Class handles the patch represtation for a Layout. 
    Layout information projected in 2D
    """
    @property
    def u_bins(self):
        return self.__u_bins

    @u_bins.setter
    def u_bins(self, value):
        if value is None:
            return
        self.__u_bins = value
        self.W = int(self.__u_bins.size - 1)
        self.uv_ref[0] = self.__u_bins[0]

    @property
    def v_bins(self):
        return self.__v_bins

    @v_bins.setter
    def v_bins(self, value):
        if value is None:
            return
        self.__v_bins = value
        self.H = int(self.v_bins.size - 1)
        self.uv_ref[1] = self.__v_bins[0]

    def __init__(self, dt):
        self.layout = None
        self.dt = dt
        self.ocg_map = None
        self.uv_boundary = None
        self.u_bins, self.v_bins = None, None
        self.is_initialized = False
        self.H, self.W = None, None
        self.uv_ref = [None, None]

    def initialize(self):
        """
        Initialize the current Patch
        Note each patch is linked to a unique Layout
        """
        self.is_initialized = False

        self.u_bins, self.v_bins = compute_uv_bins(
            pcl=self.layout.boundary[:, self.layout.cam2boundary_mask],
            grid_size=self.dt.cfg["room_id.grid_size"],
            padding=self.dt.cfg["room_id.grid_padding"]
        )

        clipped_boundary = self.layout.get_clipped_boundary()
        uv = project_xyz_to_uv(
            xyz_points=clipped_boundary,
            u_bins=self.u_bins,
            v_bins=self.v_bins
        )

        self.ocg_map = np.uint8(np.zeros((self.H, self.W)))
        cv2.fillPoly(self.ocg_map, [uv.T], color=(1, 1, 1))
        self.uv_boundary = uv

        self.is_initialized = True
        return self.is_initialized
