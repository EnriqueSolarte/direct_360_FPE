import numpy as np


def get_ocg_map_from_list_obj(list_obj, grid_size=None, weights=None, xedges=None, zedges=None, saturated=False):
    pcl = np.hstack([obj.boundary for obj in list_obj])
    ocg_map = get_ocg_map(
        pcl=pcl,
        grid_size=grid_size,
        xedges=xedges, zedges=zedges
    )[0]
    if saturated:
        ocg_map[ocg_map > 0] = 1

    return ocg_map


def proj_ocg_to_r3(k, R, uv_points):
    pts = np.linalg.inv(k) @ uv_points
    pts[2, :] = 0
    return R.T @ pts


def proj_r3_to_ocg(k, R, xyz_points):
    pts = R @ xyz_points
    pts[2, :] = 1
    return k @ pts


# def proj_patch_

def proj_xyz_to_uv(xyz_points, u_bins, v_bins):
    grid_size = abs(u_bins[1] - u_bins[0])
    shape = (v_bins.size-1, u_bins.size-1)

    x_cell_u = [np.argmin(abs(p - u_bins-grid_size*0.5)) % shape[1]
                for p in xyz_points[0, :]]
    z_cell_v = [np.argmin(abs(p - v_bins-grid_size*0.5)) % shape[0]
                for p in xyz_points[2, :]]
    # ! this potentially can change the order of the points
    # if unique:
    #     return np.unique(np.vstack((x_cell_u, z_cell_v)), axis=0), True
    return np.vstack((x_cell_u, z_cell_v))


def project_single_xyz_to_uv(xyz, u_bins, v_bins):
    x = xyz[0]
    z = xyz[2]

    u = 0
    for i in range(u_bins.shape[0] - 1):
        if x >= u_bins[i] and x < u_bins[i+1]:
            u = i
            break
    if x >= u_bins[-1]:
        u = u_bins.shape[0] - 1

    v = 0
    for i in range(v_bins.shape[0] - 1):
        if z >= v_bins[i] and z < v_bins[i+1]:
            v = i
            break
    if z >= v_bins[-1]:
        v = v_bins.shape[0] - 1
    return u, v


def compute_uv_bins(pcl, grid_size=None, padding=100):
    x = pcl[0, :]
    z = pcl[2, :]

    u_edges = np.mgrid[np.min(x)-padding*grid_size:np.max(x)+padding*grid_size:grid_size]
    v_edges = np.mgrid[np.min(z)-padding*grid_size:np.max(z)+padding*grid_size:grid_size]
    return u_edges, v_edges


# def get_ocg_map(pcl, grid_size=None, weights=None, xedges=None, zedges=None, padding=10):
def project_to_ocg(pcl, grid_size=None, weights=None, xedges=None, zedges=None, padding=100):
    x = pcl[0, :]
    z = pcl[2, :]

    if (xedges is None) or (zedges is None):
        xedges = np.mgrid[np.min(x)-padding*grid_size:np.max(x)+padding*grid_size:grid_size]
        zedges = np.mgrid[np.min(z)-padding*grid_size:np.max(z)+padding*grid_size:grid_size]

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights /= np.max(weights)

    grid, xedges, zedges = np.histogram2d(x, z, weights=1/weights, bins=(xedges, zedges))

    mask = grid > 0
    grid[mask] = 1

    # grid = gaussian_filter(grid, sigma=0.5)
    # grid = grid/np.sum(grid)

    return grid, xedges, zedges


def get_ocg_map(pcl, grid_size=None, weights=None, xedges=None, zedges=None, padding=100):
    x = pcl[0, :]
    z = pcl[2, :]

    if (xedges is None) or (zedges is None):
        xedges = np.mgrid[np.min(x)-padding*grid_size:np.max(x)+padding*grid_size:grid_size]
        zedges = np.mgrid[np.min(z)-padding*grid_size:np.max(z)+padding*grid_size:grid_size]

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights /= np.max(weights)

    grid, xedges, zedges = np.histogram2d(x, z, weights=1/weights, bins=(xedges, zedges))
    # grid = grid/np.sum(grid)
    mask = grid > 20
    grid[mask] = 20
    grid = grid/20

    return grid, xedges, zedges


def compute_entropy_from_pcl(pcl, grid_size, weights=None, xedges=None, zedges=None):
    grid, _, _ = get_ocg_map(pcl=pcl, grid_size=grid_size, weights=weights, xedges=xedges, zedges=zedges)

    return compute_entropy_from_ocg_map(grid)


def compute_entropy_from_ocg_map(ocg_map):
    mask = ocg_map > 0
    # * Entropy
    H = np.sum(-ocg_map[mask] * np.log2(ocg_map[mask]))
    return H


def get_3D_line_from_ends_by_delta_step(node_0, node_1, step):
    edge_vector = node_1 - node_0
    number = int(np.linalg.norm(edge_vector)/step)
    return get_3D_line_from_ends(node_0, node_1, number)


def get_3D_line_from_geometric_sampling(node_0, node_1, number=10, return_array=False):

    edge_vector = node_1 - node_0
    phi_max = np.arctan2(np.linalg.norm(edge_vector), 1)

    dir_vector = edge_vector / np.linalg.norm(edge_vector)
    list_pts = [node_0 + s * dir_vector for s in np.tan(np.linspace(0, phi_max, number))]
    if return_array:
        return np.vstack(list_pts).T
    return list_pts


def get_3D_lines_special(node_0, node_1, number=100, return_array=False, initial_ratio=0):
    # edge_vector = node_1 - node_0
    # dir_vector = edge_vector / np.linalg.norm(edge_vector)
    # list_pts = [node_0 + np.linalg.norm(edge_vector)*s * dir_vector for s in np.linspace(initial_ratio, 1, number)]
    # if return_array:
    #     return np.vstack(list_pts).T
    # return list_pts
    x = np.linspace(node_0[0], node_1[0], number)
    z = np.linspace(node_0[2], node_1[2], number)

    return np.vstack((x, np.zeros_like(x), z))


def get_3D_line_from_ends(node_0, node_1, number=100, return_array=False, initial_ratio=0):
    edge_vector = node_1 - node_0
    dir_vector = edge_vector / np.linalg.norm(edge_vector)
    list_pts = [node_0 + np.linalg.norm(edge_vector)*s * dir_vector for s in np.linspace(initial_ratio, 1, number)]
    if return_array:
        return np.vstack(list_pts).T
    return list_pts


def get_edge_ocg(node_0, node_1, ocg_data, number=100):
    line = get_3D_line_from_ends(
        node_0=node_0, node_1=node_1, number=number
    )
    ocg = np.zeros_like(ocg_data.ocg_map)
    grid = (ocg_data.xbins[1] - ocg_data.xbins[0])*0.5
    for bdg in line:
        x_cell = np.argmin(abs(bdg[0] - ocg_data.xbins - grid))
        z_cell = np.argmin(abs(bdg[2] - ocg_data.zbins - grid))
        ocg[x_cell-1, z_cell-1] = 1
    return ocg


def saturate_ocg_map(ocg_map):
    ocg_map = copy.deepcopy(ocg_map)
    mask = ocg_map > 0
    ocg_map[mask] = 1
    return ocg_map


def add_list_pts_to_ocg(ocg_map, xbins, zbins, list_pts, value):
    grid = (xbins[1] - xbins[0])*0.5
    for bdg in list_pts:
        x_cell = np.argmin(abs(bdg[0] - xbins - grid))
        z_cell = np.argmin(abs(bdg[2] - zbins - grid))
        ocg_map[x_cell-1, z_cell-1] = value

    return ocg_map


def get_line_pixels(x0, x1):
    # Get pixels of the lines on integer grid with Bresenham's algorithm
    d0, d1 = np.abs(x0 - x1)
    if d0 > d1:
        p0 = np.linspace(x0[0], x1[0], d0+1, dtype=np.int32)
        p1 = np.round(np.linspace(x0[1], x1[1], d0+1)).astype(np.int32)
    else:
        p1 = np.linspace(x0[1], x1[1], d1+1, dtype=np.int32)
        p0 = np.round(np.linspace(x0[0], x1[0], d1+1)).astype(np.int32)
    return p0, p1


def draw_ocg_line_uv(ocg_map, x0, x1, color, use_cv=True):
    if use_cv:
        cv2.line(ocg_map, (x0[1], x0[0]), (x1[1], x1[0]), color, 1)
    else:
        p0, p1 = get_line_pixels(x0, x1)
        ocg_map[p0, p1] = color

    return ocg_map


def compute_all_ly_pdf(self):

    all_points = np.hstack([ly.boundary for ly in self.list_ly])
    ocg_map_all, self.xbins, self.zbins = project_to_ocg(
        pcl=all_points,
        grid_size=self.cfg.params.roomi_ocg_grid_size,
        padding=self.cfg.params.ocg_padding
    )

    self.pdf_n = np.zeros_like(ocg_map_all)
    for ly in self.list_ly:
        patch = self.create_ly_patch(ly)
        pdf, _, _ = project_to_ocg(patch, xedges=self.xbins, zedges=self.zbins)

        self.pdf_n = pdf + self.pdf_n

        self.pdf_n = self.pdf_n
        plt.figure(0)
        plt.clf()
        plt.imshow(self.pdf_n)
        plt.draw()
        plt.waitforbuttonpress(0.001)

    # mask = self.pdf_n > 0.01
    # self.pdf_n[mask] = 1
    planes = np.hstack([pl.boundary for pl in self.list_pl if pl.isCandidate]).T
    final = add_list_pts_to_ocg(
        ocg_map=self.pdf_n,
        xbins=self.xbins,
        zbins=self.zbins,
        list_pts=planes,
        value=-1
    )

    plt.figure(1)
    plt.imshow(final)
    plt.show()


def compute_patch_iou(patch_a, patch_b):

    patch_a[patch_a > 0] = 1
    patch_b[patch_b > 0] = 1

    union = patch_a + patch_b

    overlap = np.sum(union == 2)

    union = np.sum(union >= 1)
    return overlap/union


class OCGRoom:
    def __init__(self, x_min, x_max, grid_size, padding, min_size, max_size):
        self.x_min = x_min
        self.x_max = x_max
        self.padding = padding

        self.grid_size = grid_size

        # This ensure the resolution for a room will not be too large or too small
        min_edge = np.min(self.x_max.reshape(-1) - self.x_min.reshape(-1))
        max_edge = np.max(self.x_max.reshape(-1) - self.x_min.reshape(-1))
        H, W = self.get_shape()
        max_grid_edge = max(H - self.padding*2, W - self.padding*2)
        if max_size is not None and max_grid_edge > max_size:
            self.grid_size = max_edge / max_size
            print(f'Origin grid_size {grid_size:.2f}, new larger grid_size={self.grid_size:.2f}')

        H, W = self.get_shape()
        min_grid_edge = min(H - self.padding*2, W - self.padding*2)
        if min_size is not None and min_grid_edge < min_size:
            self.grid_size = min_edge / min_size
            print(f'Origin grid_size {grid_size:.2f}, new small grid_size={self.grid_size:.2f}')

    def get_ocg_map(self, xy_points):
        uv_points = self.coords_to_ocg(xy_points)
        uv_max = np.ceil(self.coords_to_ocg(self.x_max).reshape(-1))
        xbins = np.arange(0, uv_max[0]+1, 1)
        ybins = np.arange(0, uv_max[1]+1, 1)

        grid, xedges, yedges = np.histogram2d(
            uv_points[0, :],
            uv_points[1, :],
            bins=(xbins, ybins))
        grid = np.pad(
            grid,
            ((0, self.padding), (0, self.padding)),
            mode='constant',
            constant_values=0
        )
        return grid

    def draw_ocg_line_xy(self, ocg_map, x0, x1, color):
        x0 = self.coords_to_ocg(np.expand_dims(x0, 1)).reshape(-1)
        x1 = self.coords_to_ocg(np.expand_dims(x1, 1)).reshape(-1)
        x0 = np.round(x0).astype(np.int32)
        x1 = np.round(x1).astype(np.int32)
        H, W = ocg_map.shape
        x0 = np.clip(x0, [0, 0], [H-1, W-1])
        x1 = np.clip(x1, [0, 0], [H-1, W-1])
        return draw_ocg_line_uv(ocg_map, x0, x1, color)

    def point_to_ocg(self, xy_point):
        return (xy_point - self.x_min.reshape(-1)) / self.grid_size + self.padding

    def ocg_to_point(self, uv_point):
        return (uv_point - self.padding) * self.grid_size + self.x_min.reshape(-1)

    def coords_to_ocg(self, xy_points):
        return (xy_points - self.x_min) / self.grid_size + self.padding

    def ocg_to_coords(self, uv_points):
        return (uv_points - self.padding) * self.grid_size + self.x_min

    def get_shape(self):
        shape = (self.x_max.reshape(-1) - self.x_min.reshape(-1)) / self.grid_size + self.padding * 2
        shape = np.ceil(shape).astype(np.int32)
        return shape


class Patch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.map = None
        self.uv_ref = None
        self.uv = None
        self.boundary = None
        self.u_bins, self.v_bins = None, None


class OCGPatch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.v_bins = None
        self.u_bins = None
        self.ocg_map = None
        self.list_ly = []
        self.grid_size = self.cfg.params.roomi_ocg_grid_size  # TODO A explicit param is better
        # self.padding = self.cfg.params.ocg_padding
        # self.pdf = None
        self.list_patch = []
        # ! uv coordinates for the 1st layout
        # self.uv_ref = None
        self.isInitialized = False
        self.dynamic_bins = True

    def get_patch_and_register_layout(self, layout):
        """
        Returns a 2D map (Patch) given a layout
        """

        boundary = layout.get_closest_boundnary_pts(self.cfg.params.sampling_boundary)
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
            reduced_padding = self.cfg.params.ocg_padding
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

    def create_patch(self, boundary):
        """Returns an obj Patch based on the passed boundary

        :param boundary: [description]
        :type boundary: [type]
        """

        patch = Patch(self.cfg)
        patch.u_bins, patch.v_bins = compute_uv_bins(
            pcl=boundary,
            grid_size=self.cfg.params.roomi_ocg_grid_size,
            padding=self.cfg.params.ocg_padding
        )

        h, w = patch.v_bins.size-1, patch.u_bins.size - 1
        # grid_size = abs(patch.v_bins[1] - patch.v_bins[0])

        uv = proj_xyz_to_uv(
            xyz_points=boundary,
            u_bins=patch.u_bins,
            v_bins=patch.v_bins
        )

        map_2d = np.uint8(np.zeros((int(h), int(w))))
        cv2.fillPoly(map_2d, [uv.T], color=(1, 1, 1))

        patch.map = map_2d
        patch.uv = uv
        patch.boundary = boundary
        patch.uv_ref = patch.u_bins[0], patch.v_bins[0]
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

        patch = Patch(self.cfg)
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
        
        if self.cfg.forced_thr_room_id is None:
            threshold = self.cfg.params.patches_room_threshold
        else:
            threshold = self.cfg.forced_thr_room_id
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
        flag = self.cfg.params.mask_in_patches
        if self.cfg.forced_thr_room_id is None:
            threshold = self.cfg.params.patches_room_threshold
        else:
            threshold = self.cfg.forced_thr_room_id
            
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

        # mask_0 = self.ocg_map/np.max(self.ocg_map) > 0.0

    def compute_global_bins(self):
        bins = np.vstack([(local_patch.u_bins[0], local_patch.v_bins[0],
                           local_patch.u_bins[-1], local_patch.v_bins[-1])
                          for local_patch in self.list_patch
                          ])

        min_points = np.min(bins, axis=0)[0:2]
        max_points = np.max(bins, axis=0)[2:4]
        self.u_bins = np.mgrid[min_points[0]:max_points[0]+self.cfg.params.roomi_ocg_grid_size: self.cfg.params.roomi_ocg_grid_size]
        self.v_bins = np.mgrid[min_points[1]:max_points[1]+self.cfg.params.roomi_ocg_grid_size: self.cfg.params.roomi_ocg_grid_size]
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
        # ! getting only the closest points in the boundary as well as sampling them
        boundary = layout.get_closest_boundnary_pts(sampling=layout.cfg.params.sampling_boundary)
        # ! if no point is found
        if boundary is None:
            return False

        # ! Creating  a Patch obj
        patch = self.create_patch(boundary)
        # patch.uv_ref = self.u_bins[0], self.v_bins[0]
        self.ocg_map = np.copy(patch.map)
        self.list_ly.append(layout)
        self.list_patch.append(patch)
        # self.compute_global_bins()
        self.u_bins, self.v_bins = patch.u_bins, patch.v_bins
        self.isInitialized = True
        return True


class OCGData:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ocg_map = None
        self.zbins = None
        self.xbins = None
        self.list_dt = None

    def set_ocg_map(self, list_dt, factor=1, saturated=False):
        """
        Sets the local ocg map based on list of obj (dt.boundary).
        """
        pcl = np.hstack([obj.boundary for obj in list_dt])
        # pcl = np.hstack([obj.get_sampled_boundary() for obj in list_dt])
        ocg_map, self.xbins, self.zbins = get_ocg_map(
            pcl=pcl,
            grid_size=self.cfg.params.roomi_ocg_grid_size*factor,
            padding=self.cfg.params.ocg_padding
        )
        if saturated:
            msk = ocg_map > 0
            self.ocg_map = np.zeros_like(ocg_map)
            self.ocg_map[msk] = 1
        else:
            # self.ocg = ocg_m
            self.ocg_map = ocg_map/np.max(ocg_map)
        #     mask

        self.list_dt = list_dt
        return self.ocg_map

    def get_visual_ocg(self, saturated=False, caption="OCG map"):
        if saturated:
            msk = self.ocg_map > self.cfg.params.ocg_threshold
            ocg = deepcopy(self.ocg_map)
            ocg[msk] = 1.0
        else:
            ocg = deepcopy(self.ocg_map)

        if self.list_dt.__len__() > 0:
            ocg = add_list_pts_to_ocg(
                ocg_map=ocg,
                xbins=self.xbins,
                zbins=self.zbins,
                list_pts=[self.list_dt[-1].pose.t, ],
                value=-1,
            )

        return cv2.rotate(ocg, cv2.ROTATE_180)

    def plot(self, block=True, saturated=False, caption="OCG map"):
        rotate_ocg = self.get_visual_ocg(saturated=saturated, caption="OCG map")

        plt.figure(caption)
        plt.clf()
        plt.title(caption)
        # rotate_ocg = cv2.rotate(ocg, cv2.ROTATE_180)
        # # rotate_ocg = cv2.flip(rotate_ocg, 1)

        plt.imshow(rotate_ocg)
        plt.draw()
        plt.waitforbuttonpress(0.001)
        plt.show(block=block)
