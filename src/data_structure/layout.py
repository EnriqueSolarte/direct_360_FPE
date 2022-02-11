import numpy as np 

class Layout:

    def __init__(self, data_manager):
        self.dt = data_manager

        self.boundary = None
        self.bearings = None

        # self.corners_id = None
        # self.corner_coords = Non
        self.pose_gt = None
        self.pose_est = None
        self.idx = None

        # !Planes features defined in this LY
        self.list_pl = []
        self.list_corners = []

        # >> used in room identifier
        self.central_pose = None

        self.ly_data = None
        self.cam_ref = None
        self.height_ratio = 1

    def set_room(self, room_info):
        # print_info("LY idx {} was set into ROOM ID #{}".format(self.idx, room_info.idx))
        self.pose.room_info = room_info


    def set_room_ref(self, room_ref):
        self.pose.room_info.pose_ref = room_ref


    def apply_scale(self, scale):

        if self.cam_ref == Enum.WC_SO3_REF:
            self.boundary = self.boundary + (scale/self.pose.scale) * np.ones_like(self.boundary) * self.pose.t.reshape(3, 1)
            self.cam_ref = Enum.WC_REF

        elif self.cam_ref == Enum.WC_REF:
            delta_scale = scale - self.pose.scale
            self.boundary = self.boundary + (delta_scale/self.pose.scale) * np.ones_like(self.boundary) * self.pose.t.reshape(3, 1)

        if self.list_planes is not None:
            if self.list_planes.__len__() > 0:
                [pl.apply_scale(scale) for pl in self.list_planes if pl is not None]

        if self.list_corners.__len__() > 0:
            [cr.apply_scale(scale) for cr in self.list_corners]

        self.pose.scale = scale

        return True

    def compute_plane_features(self):
        pcl = self.boundary

        # self.cfg.corner_distance_max = self.pose.ly_size * self.cfg.params.corner_distance_threshold
        # self.cfg.plane_distance_max = self.pose.ly_size * self.cfg.params.plane_distance_threshold

        # self.cfg.corner_distance_max = np.tan(self.cfg.params.corner_distance_threshold*np.pi/2)
        # self.cfg.plane_distance_max = np.tan(self.cfg.params.plane_distance_threshold*np.pi/2)

        # self.corner_distance_max = np.sin(self.cfg.params.corner_distance_threshold*np.pi/2)*self.pose.ly_size
        # self.plane_distance_max = np.sin(self.cfg.params.plane_distance_threshold*np.pi/2)*self.pose.ly_size

        self.corner_distance_max_ly = self.dt.params.corner_distance_threshold*self.pose.ly_size
        self.plane_distance_max_ly = self.dt.params.plane_distance_threshold*self.pose.ly_size

        # ! Setting the closest corners
        corn_idx, _ = find_N_peaks(self.ly_data[2, :], r=self.dt.params.radius_peak)
        corners = [pcl[:, i] for i in corn_idx]
        mask = [np.linalg.norm(c-self.pose.t) < self.corner_distance_max_ly for c in corners]

        self.list_corners = [Corner.from_position_and_pose(c, self.pose) for c, m in zip(corners, mask)
                             if m
                             ]

        pcl_list = [pcl[:, corn_idx[i]:corn_idx[i + 1]] for i in range(len(corn_idx) - 1)]
        pcl_list.append(np.hstack((pcl[:, corn_idx[-1]:], pcl[:, 0:corn_idx[0]])))

        planes = []
        for dt in pcl_list:
            rpl = RansacPlanes(self.dt, pose=self.pose, idx=self.idx, label_ref=self.cam_ref)
            pl, flag_success = rpl.estimate_plane(dt)
            if not flag_success:
                continue

            pl.compute_distance2cam()
            if pl.distance2cam < self.plane_distance_max_ly:
                planes.append(pl)

        if planes.__len__() == 0:
            return False
        else:
            self.list_planes = planes
            return True

    def get_sampled_boundary(self):
        return get_sampling_dt(self.boundary,
                               self.dt.params.ratio_sampling_for_ocg)

    def get_closest_boundnary_pts(self, sampling=0.5):
        """
        This function mask-out point boundaries that are far from the current camera pose
        by leveraging the angular position ($\phi$). close to np.pi/2 is close --> 0 is far
        """
        # from visualization.visualization import plot_pcl_list
        # ratio = abs(2 - self.pose.ly_size )/2

        mask = abs(self.bearings_phi) > (1-self.dt.params.point_distance_threshold) * (np.pi/2)
        pcl = self.boundary[:, mask]
        # return pcl
        if sampling > 0:
            idx = np.linspace(0, pcl.shape[1] - 1, int(pcl.shape[1]*sampling)).astype(int)
            if idx.size == 0:
                return None
            return pcl[:, idx]
        return pcl

    def apply_gt_scale(self, scale):
        self.boundary = self.boundary*scale
        self.pose.scale = scale

    def estimate_height_ratio(self):
        """
        Estimates the height ratio that describes the distance ratio of camera-floor over the
        camera-ceiling distance. This information is important to recover the 3D
        structure of the predicted Layout
        """
        floor = np.abs(self.ly_data[1, :])
        ceiling = np.abs(self.ly_data[0, :])

        ceiling[ceiling > np.radians(80)] = np.radians(80)
        ceiling[ceiling < np.radians(5)] = np.radians(5)
        floor[floor > np.radians(80)] = np.radians(80)
        floor[floor < np.radians(5)] = np.radians(5)

        self.height_ratio = np.mean(np.tan(ceiling)/np.tan(floor))

