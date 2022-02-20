import numpy as np
import cv2
from utils.geometry_utils import extend_array_to_homogeneous
from utils.ocg_utils import compute_uv_bins, project_xyz_to_uv
from src.data_structure import layout


class OCGPatches:
    """
    This class handles multiples Patches (defined from every Layout) by 
    aggregating, combining, and pruning them out properly.
    """

    def __init__(self, data_manager):
        self.dt = data_manager
        self.v_bins = None
        self.u_bins = None
        self.ocg_map = None
        self.list_patches = []
        self.dynamic_bins = True
        self.is_initialized = False

    def project_xyz_to_uv(self, xyz_points):

        grid_size = self.dt.cfg["room_id.grid_size"]
        x_cell_u = [np.argmin(abs(p - self.u_bins-grid_size*0.5)) % self.get_shape()[1]
                    for p in xyz_points[0, :]]

        if xyz_points.shape[0] == 2:
            # xyz_points: (2, N)
            z_cell_v = [np.argmin(abs(p - self.v_bins-grid_size*0.5)) % self.get_shape()[0]
                        for p in xyz_points[1, :]]
        else:
            # xyz_points: (3, N)
            z_cell_v = [np.argmin(abs(p - self.v_bins-grid_size*0.5)) % self.get_shape()[0]
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

    def initialize(self, patch):
        """
        Initializes the OCGPatch class by using the passed patch
        """
        self.is_initialized = False
        if not patch.is_initialized:
            print("Current patch is not initialized... Initializing...!")
            if not patch.initialize():
                return self.is_initialized

        self.ocg_map = np.expand_dims(patch.ocg_map, 2)
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

        grid_size = self.dt.cfg["room_id.grid_size"]
        min_points = np.min(bins, axis=0)[0:2]
        max_points = np.max(bins, axis=0)[2:4]
        self.u_bins = np.mgrid[min_points[0]:max_points[0]+grid_size: grid_size]
        self.v_bins = np.mgrid[min_points[1]:max_points[1]+grid_size: grid_size]
        return self.u_bins, self.v_bins

    def update_ocg_map(self):

        H, W = self.get_shape()
        self.ocg_map = np.zeros((H, W, self.list_patches.__len__()))
        for idx, patch in enumerate(self.list_patches):
            h, w = patch.H, patch.W
            uv = project_xyz_to_uv(
                xyz_points=np.array((patch.uv_ref[0], 0, patch.uv_ref[1])).reshape((3, 1)),
                u_bins=self.u_bins,
                v_bins=self.v_bins
            ).squeeze()

            self.ocg_map[uv[1]:uv[1]+h, uv[0]:uv[0]+w, idx] = patch.ocg_map

    def add_patch(self, patch):
        self.list_patches.append(patch)
        self.update_bins()
        self.update_ocg_map()

    def get_shape(self):
        return (self.v_bins.size-1, self.u_bins.size-1)


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
