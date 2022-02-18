import numpy as np
import cv2
from utils.ocg_utils import compute_uv_bins, project_xyz_to_uv
from src.data_structure import layout


class OCGPatch:
    """
    This class handles multiples Patches (defined from every Layout) by 
    aggregating, combining, and pruning the out.
    """

    def __init__(self, data_manager):
        self.dt = data_manager
        self.v_bins = None
        self.u_bins = None
        self.ocg_map = None
        self.list_ly = []
        self.grid_size = self.dt.cfg['room_id.grid_size']  # TODO A explicit param is better
        self.list_patch = []

        self.is_initialized = False
        self.dynamic_bins = True

    def get_patch_and_register_layout(self, layout):
        """
        Returns a 2D map (Patch) given a layout
        """

        boundary = layout.get_closest_boundnary_pts(self.dt.params.sampling_boundary)
        # ! If there are not enough points in the boundary the None is returned above
        if boundary is None:
            boundary = layout.pose.t.reshape((3, 1))
        else:
            boundary = np.hstack((boundary, layout.pose.t.reshape((3, 1))))
        # patch, _ = self.get_patch_from_xyz_bounds(boundary)
        patch = self.create_patch(boundary)
        self.list_ly.append(layout)
        return patch

    def eval_and_update_current_bins(self, boundary):
        """Evaluates the passed boundary using the current registered bins.
        If the evaluation detects that the current bins is obsolete (to small to describe the boundary)
        it updates the bins.
        :param boundary: array 3,n (xyz points)
        :type boundary: ((uv), ret) --> ret: bins has change or not
        """
        # from visualization.visualization import plot_list_pcl
        # ! Flag that returns information that the current bins has been changed
        ret = False

        if self.dynamic_bins:
            # ! When any point in the current boundary exceed the current grid
            reduced_padding = self.dt.params.ocg_padding
            max_bound = np.max(boundary, axis=1)
            min_bound = np.min(boundary, axis=1)

            # ! u direction exceed max u_bin
            if self.u_bins[-reduced_padding] < max_bound[0]:
                d_u = np.round(abs(max_bound[0] - self.u_bins[-reduced_padding]) / self.grid_size).astype(np.int)
                extra_bins = np.mgrid[self.u_bins[-1] + self.grid_size:self.u_bins[-1] + d_u*self.grid_size:self.grid_size]
                self.u_bins = np.append(self.u_bins, extra_bins)
                print_info(f"Adding u_bins {extra_bins.shape[0]} to the right")
                ret = True

            # ! u direction exceed min u_bin
            if self.u_bins[reduced_padding] > min_bound[0]:
                d_u = np.round(abs(min_bound[0] - self.u_bins[reduced_padding]) / self.grid_size).astype(np.int)
                extra_bins = np.mgrid[self.u_bins[0] - d_u*self.grid_size: self.u_bins[0] - self.grid_size:self.grid_size]
                self.u_bins = np.append(extra_bins, self.u_bins)
                print_info(f"Adding u_bins {extra_bins.shape[0]} to the left")
                ret = True

            # ! v direction exceed max v_bin
            if self.v_bins[-reduced_padding] < max_bound[2]:
                d_v = np.round(abs(max_bound[2] - self.v_bins[-reduced_padding]) / self.grid_size).astype(np.int)
                extra_bins = np.mgrid[self.v_bins[-1] + self.grid_size:self.v_bins[-1] + d_v*self.grid_size:self.grid_size]
                self.v_bins = np.append(self.v_bins, extra_bins)
                print_info(f"Adding v_bins {extra_bins.shape[0]} to the right")
                ret = True

            # ! v direction exceed min v_bin
            if self.v_bins[reduced_padding] > min_bound[2]:
                d_v = np.round(abs(min_bound[2] - self.v_bins[reduced_padding]) / self.grid_size).astype(np.int)
                extra_bins = np.mgrid[self.v_bins[0] - d_v*self.grid_size: self.v_bins[0] - self.grid_size:self.grid_size]
                self.v_bins = np.append(extra_bins, self.v_bins)
                print_info(f"Adding v_bins {extra_bins.shape[0]} to the left")
                ret = True

        x_cell_u = [np.argmin(abs(p - self.u_bins - self.grid_size*0.5)) % self.get_shape()[1]
                    for p in boundary[0, :]]
        z_cell_v = [np.argmin(abs(p - self.v_bins - self.grid_size*0.5)) % self.get_shape()[0]
                    for p in boundary[2, :]]

        return np.vstack((x_cell_u, z_cell_v)), ret

    def create_patch(self, layout):
        """Returns an obj Patch based on the passed layout
        """

        patch = Patch(layout.dt)
        patch.initialize(layout)
        return patch

    def get_patch_from_xyz_bounds(self, boundary):
        """
        Returns a 2D-patch from a boundary (xyz-points)

        :param boundary: set of xyz-points (3, n)
        :return 2d-map array (v-bins, u-bins)
        """
        uv, ret = self.eval_and_update_current_bins(boundary)
        if ret:
            print_info("Bins in the current OCGPatch has been updated")
            print_info(f"Bins shape ({self.get_shape()})")

        # uv = proj_xyz_to_uv(xyz_points=boundary, u_bins=self.u_bins, v_bins=self.v_bins)
        map_2d = np.uint8(np.zeros(self.get_shape()))
        cv2.fillPoly(map_2d, [uv.T], color=(1, 1, 1))
        # cv2.polylines(map_2d, [uv.T], color=(1, 1, 1), isClosed=False)
        # [cv2.line(map_2d, uv[:, i], uv[:, i+1], color=(1, 1, 1)) for i in range(uv.shape[1]-1)]
        # plt.imshow(map_2d)
        # plt.show()

        patch = Patch(self.dt)
        patch.map = map_2d
        patch.uv = uv
        patch.boundary = boundary
        patch.uv_ref = self.u_bins[0], self.v_bins[0]

        return patch, ret

    def get_patch_as_3D_pts(self, layout):
        """
        Returns a patch structure as a set of 3D points
        """
        # ! Getting only the point which are close to camera ref
        boundary = layout.get_closest_boundnary_pts()
        # boundary = np.hstack([pl.get_masked_boundary() for pl in layout.list_planes])

        # ! If boundary is empty
        if boundary.shape[1] == 0:
            return boundary

        patch = np.hstack([get_3D_lines_special(
            node_0=layout.pose.t,
            node_1=p,
            number=100,
            return_array=True, initial_ratio=0.5
        ) for p in boundary.T])

        return patch

    def get_shape(self):
        return (self.v_bins.size-1, self.u_bins.size-1)

    def get_map_ref(self):
        """Returns the most top-left coordinates for the local patch as references
        """
        return self.u_bins[0], self.v_bins[0]

    def temporal_weight(self, idx):
        # # ! assuming a linear weighting
        # # > w = mx + b --> x: idx  [0=older    L= newest]
        minimal = 0
        m = (minimal - 1) / self.list_patch.__len__()
        return m*idx + 1

    def get_mask_by_med(self):
        # mask_0 = self.ocg_map/np.max(self.ocg_map) > 0.0
        tmp = self.ocg_map/np.max(self.ocg_map)
        mean = np.median(tmp[tmp > 0])
        return self.ocg_map/np.max(self.ocg_map) >= mean

    def get_mask_by_threshold(self, ocg_map=None):

        if self.dt.forced_thr_room_id is None:
            threshold = self.dt.params.patches_room_threshold
        else:
            threshold = self.dt.forced_thr_room_id
        if ocg_map is not None:
            tmp = ocg_map/np.max(ocg_map)
            # mask = tmp > self.cfg.params.patches_room_threshold
            mask = tmp > threshold

            return mask

        # mask_0 = self.ocg_map/np.max(self.ocg_map) > 0.0
        tmp = self.ocg_map/np.max(self.ocg_map)
        # mask = tmp > self.cfg.params.patches_room_threshold
        mask = tmp > threshold

        return mask

    def get_mask(self):
        """
        Returns a mask OCG (same size that ocg_map) describing the valid pixel in the current ocg_map
        """
        # ! So far our best policy is to define the threshold as mean of non-zero values. (valid pixels in the final patch)
        flag = self.dt.params.mask_in_patches
        if self.dt.forced_thr_room_id is None:
            threshold = self.dt.params.patches_room_threshold
        else:
            threshold = self.dt.forced_thr_room_id

        if flag == Enum.PATCH_THR_CONST:
            tmp = self.ocg_map/np.max(self.ocg_map)
            mask = tmp > threshold
            return mask

        if flag == Enum.PATCH_THR_BY_MEAN:
            tmp = self.ocg_map/np.max(self.ocg_map)
            threshold = np.mean(tmp[tmp > 0])
            return self.ocg_map/np.max(self.ocg_map) >= threshold

        if flag == Enum.PATCH_THR_BY_MED:
            tmp = self.ocg_map/np.max(self.ocg_map)
            threshold = np.median(tmp[tmp > 0])
            return self.ocg_map/np.max(self.ocg_map) >= threshold

    def compute_global_bins(self):
        bins = np.vstack([(local_patch.u_bins[0], local_patch.v_bins[0],
                           local_patch.u_bins[-1], local_patch.v_bins[-1])
                          for local_patch in self.list_patch
                          ])

        min_points = np.min(bins, axis=0)[0:2]
        max_points = np.max(bins, axis=0)[2:4]
        self.u_bins = np.mgrid[min_points[0]:max_points[0]+self.dt.params.roomi_ocg_grid_size: self.dt.params.roomi_ocg_grid_size]
        self.v_bins = np.mgrid[min_points[1]:max_points[1]+self.dt.params.roomi_ocg_grid_size: self.dt.params.roomi_ocg_grid_size]
        return self.u_bins, self.v_bins

    def add_measurements(self, list_ly):
        """
        Adds measurement patch from input layout
        """
        if not self.isInitialized:
            for ly in list_ly:
                if self.initialize(ly):
                    break
            # raise ValueError("OCGPatch has not be initialized yet")

        """
        Notes:
        1. If it is initialized, then there are bins already defined
        2. Given a new layout, we have a new Patch (one for each)
        3. Every new Patch will be defined at different 2D location. We
        Need to dynamicall alter u- and v-bins
        """

        # #! Create a list of patches only for the new ly
        [self.list_patch.append(
            # self.get_patch_as_3D_pts(ly)
            self.get_patch_and_register_layout(ly))
            # self.create_and_register_patch(ly))
            for ly in list_ly
            if ly not in self.list_ly
         ]

        # # TODO We can improve here by adding only the last patch
        self.compute_global_bins()
        H, W = self.get_shape()
        self.ocg_map = np.zeros((H, W))

        for idx, patch in enumerate(self.list_patch):

            h, w = patch.v_bins.size - 1, patch.u_bins.size - 1
            local_ref_xz = patch.u_bins[0], patch.v_bins[0]
            uv = proj_xyz_to_uv(
                xyz_points=np.array((local_ref_xz[0], 0, local_ref_xz[1])).reshape((3, 1)),
                u_bins=self.u_bins,
                v_bins=self.v_bins
            ).squeeze()
            # weight = self.temporal_weight(idx)
            weight = 1
            self.ocg_map[uv[1]:uv[1]+h, uv[0]:uv[0]+w] += patch.map * weight

        assert self.ocg_map.shape == self.get_shape()
        return True

    def project_xyz_points_to_hist(self, xyz_points):
        x = xyz_points[0, :]
        if xyz_points.shape[0] == 2:
            z = xyz_points[1, :]
        else:
            z = xyz_points[2, :]
        grid, _, _ = np.histogram2d(z, x, bins=(self.v_bins, self.u_bins))
        # mask = grid > 0
        # grid[mask] = 1
        return grid

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

    def project_from_patch(self, local_ocg_patch):
        """Returns the passed patch drew using the self.bins defined in this class

        :param local_patch: [description] ext OCGPatch obj
        :type local_patch: [type] patch map
        """
        ocg_map = np.zeros(self.get_shape())
        local_mask = local_ocg_patch.get_mask()
        h, w = local_ocg_patch.get_shape()
        local_ref_xz = local_ocg_patch.u_bins[0], local_ocg_patch.v_bins[0]
        uv = self.project_xyz_to_uv(
            xyz_points=np.array((local_ref_xz[0], 0, local_ref_xz[1])).reshape((3, 1))
        ).squeeze()

        ocg_map[uv[1]:uv[1]+h, uv[0]:uv[0]+w][local_mask] = local_ocg_patch.ocg_map[local_mask]
        return ocg_map

    def initialize(self, layout):
        """
        Initializes the OCGPatch class by using the passed Layout
        """

        # ! Creating  a Patch obj
        patch = self.create_patch(layout)

        self.ocg_map = np.copy(patch.map)
        self.list_ly.append(layout)
        self.list_patch.append(patch)

        self.u_bins, self.v_bins = patch.u_bins, patch.v_bins
        self.is_initialized = True
        return self.is_initialized


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
        self.uv_ref[1] = self.__u_bins[0]

    @property
    def v_bins(self):
        return self.__v_bins

    @v_bins.setter
    def v_bins(self, value):
        if value is None:
            return
        self.__v_bins = value
        self.H = int(self.v_bins.size - 1)
        self.uv_ref[0] = self.__v_bins[0]

    def __init__(self, dt):
        self.layout = None
        self.dt = dt
        self.map = None
        self.uv_boundary = None
        self.u_bins, self.v_bins = None, None
        self.is_initialize = False
        self.H, self.W = None, None
        self.uv_ref = [None, None]

    def initialize(self, layout):
        """
        Initialize the current Patch with passed layout
        """
        if not self.is_initialize:
            self.layout = layout
            self.u_bins, self.v_bins = compute_uv_bins(
                pcl=self.layout.boundary,
                grid_size=self.dt.cfg["room_id.grid_size"],
                padding=self.dt.cfg["room_id.grid_padding"]
            )

            clipped_boundary = layout.get_clipped_boundary()
            uv = project_xyz_to_uv(
                xyz_points=clipped_boundary,
                u_bins=self.u_bins,
                v_bins=self.v_bins
            )

            self.map = np.uint8(np.zeros((self.H, self.W)))
            cv2.fillPoly(self.map, [uv.T], color=(1, 1, 1))
            self.uv_boundary = uv

