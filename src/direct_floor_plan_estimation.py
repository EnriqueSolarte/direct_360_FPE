import numpy as np
from tqdm import tqdm

from src.scale_recover import ScaleRecover
from src.solvers.plane_estimator import PlaneEstimator
from src.data_structure import OCGPatches, Room
from utils.geometry_utils import find_N_peaks
from utils.ocg_utils import compute_iou_ocg_map
from utils.enum import ROOM_STATUS
from utils.visualization.room_utils import plot_curr_room_by_patches
from utils.visualization.room_utils import plot_all_rooms_by_patches
from utils.visualization.room_utils import plot_estimated_orientations


class DirectFloorPlanEstimation:

    def __init__(self, data_manager):
        self.dt = data_manager
        self.scale_recover = ScaleRecover(self.dt)
        self.plane_estimator = PlaneEstimator(self.dt)
        self.global_ocg_patch = OCGPatches(self.dt)
        self.list_ly = []
        self.list_pl = []

        self.list_rooms = []

        self.curr_room = None
        self.is_initialized = False

        print("DirectFloorPlanEstimation initialized successfully")

    def estimate(self, layout):
        """
        It add the passed Layout to the systems and estimated the floor plan
        """

        if not self.is_initialized:
            self.initialize(layout)
            return

        if not self.initialize_layout(layout):
            return layout.is_initialized

        if self.eval_new_room_creation(layout):
            self.eval_room_overlapping()
            prev_room = self.curr_room
            self.curr_room = self.select_room(layout)
            if self.curr_room is None:
                # Estimate room shape sequentially
                # TODO: Compute room shape sequentially
                # out_dict = prev_room.compute_room_shape()

                # ! New Room in the system
                self.curr_room = Room(self.dt)
                # ! Initialize current room
                if self.curr_room.initialize(layout):
                    self.list_rooms.append(self.curr_room)
                    self.global_ocg_patch.list_patches.append(
                        self.curr_room.local_ocg_patches
                    )
                return

        self.update_data(layout)
        # plot_curr_room_by_patches(self)
        # plot_all_rooms_by_patches(self)
        # plot_estimated_orientations(self.curr_room.theta_z)

    def update_data(self, layout):
        """
        Updates all data in the system given the new current passed layout
        """
        if not layout.is_initialized:
            raise ValueError("Passed Layout must be initialized first...")

        self.curr_room.add_layout(layout)
        self.add_layout(layout)

    def add_layout(self, layout):
        """
        Adds a new Layout to FPE class
        """
        assert layout.is_initialized, "Passed layout must be initialized first... "

        # self.global_ocg_patch.list_patches.append(self.curr_room.local_ocg_patches)
        self.list_ly.append(layout)
        [self.list_pl.append(pl) for pl in layout.list_pl]

    def select_room(self, layout):
        """
        Returns the most likely room based on the current layout camera pose
        """
        # ! Reading local OCGPatches (rooms)
        selected_rooms = []
        for idx, ocg_room in enumerate(self.global_ocg_patch.list_patches):
            pose_uv = ocg_room.project_xyz_to_uv(
                xyz_points=layout.pose_est.t.reshape((3, 1))
            )
            eval_pose = ocg_room.ocg_map[pose_uv[1, :], pose_uv[0, :]]/ocg_room.ocg_map.max()

            if eval_pose > self.dt.cfg["room_id.ocg_threshold"]:
                selected_rooms.append((eval_pose, self.list_rooms[idx]))

        if selected_rooms.__len__() == 0:
            # ! There is not any room for the passed layout
            return None
        else:
            likelihood = max(selected_rooms, key=lambda p: p[0])
            return likelihood[1]

    def initialize_layout(self, layout):
        """
        Initializes the passed layout. This function has to be applied to all
        layout before any FEP module
        """
        # layout.compute_cam2boundary()
        # layout.patch.initialize()
        self.apply_vo_scale(layout)
        self.compute_planes(layout)
        layout.initialize()
        layout.is_initialized = True
        return layout.is_initialized

    def initialize(self, layout):
        """
        Initializes the system
        """
        self.is_initialized = False
        if not self.scale_recover.estimate_vo_and_gt_scale():
            return self.is_initialized

        # ! Create very first Room
        self.curr_room = Room(self.dt)

        # ! Initialize current layout
        if not self.initialize_layout(layout):
            return self.is_initialized

        # ! Initialize current room
        if not self.curr_room.initialize(layout):
            return self.is_initialized

        self.list_ly.append(layout)
        [self.list_pl.append(pl) for pl in layout.list_pl]

        # ! Initialize Global Patches
        # > NOTE: Global Patches is build from Local OCGPatches defined in each room
        # > Local OCGPatches in each Room is defined using Patches only.
        # * The reason is because at ROOM level we want to aggregate Patches individually for each new Layout
        # * at FPE level, we care about individuals ROOMS, i.e., we don't care how many LYs are en each room but
        # * the aggregated OCG-map only
        if not self.global_ocg_patch.initialize(self.curr_room.local_ocg_patches):
            return self.is_initialized

        # ! Only if the room is successfully initialized
        self.list_rooms.append(self.curr_room)

        self.is_initialized = True
        return self.is_initialized

    def eval_new_room_creation(self, layout):
        """
        Evaluates whether the passed layout triggers a new room
        """
        assert layout.is_initialized, "Layout must be initialized before..."

        pose_uv = self.curr_room.local_ocg_patches.project_xyz_to_uv(
            xyz_points=layout.pose_est.t.reshape((3, 1))
        )
        room_ocg_map = np.copy(self.curr_room.local_ocg_patches.ocg_map)

        # curr_room_idx = self.list_rooms.index(self.curr_room)
        # tmp_ocg = self.global_ocg_patch.ocg_map[:, :, curr_room_idx]
        tmp_ocg = room_ocg_map
        eval_pose = tmp_ocg[pose_uv[1, :], pose_uv[0, :]]/tmp_ocg.max()
        self.curr_room.p_pose.append(eval_pose)

        if eval_pose < self.dt.cfg["room_id.ocg_threshold"]:
            return True
        else:
            return False

    def apply_vo_scale(self, layout):
        """
        Applies VO-scale to the passed layout
        """
        layout.apply_vo_scale(self.scale_recover.vo_scale)
        print("VO-scale {0:2.2f} applied to Layout {1:1d}".format(
            self.scale_recover.vo_scale,
            layout.idx)
        )

    def compute_planes(self, layout):
        """
        Computes Planes in the passed layout
        """
        corn_idx, _ = find_N_peaks(layout.ly_data[2, :], r=100)

        pl_hypotheses = [layout.boundary[:, corn_idx[i]:corn_idx[i + 1]] for i in range(len(corn_idx) - 1)]
        pl_hypotheses.append(np.hstack((layout.boundary[:, corn_idx[-1]:], layout.boundary[:, 0:corn_idx[0]])))

        list_pl = []
        for pl_h in pl_hypotheses:
            pl, flag_success = self.plane_estimator.estimate_plane(pl_h)
            if not flag_success:
                continue

            pl.pose = layout.pose_est
            list_pl.append(pl)

        layout.list_pl = list_pl

    def eval_room_overlapping(self):
        """
        Merges Rooms based on the overlapping of OCGPatches (local_ocg_map)
        """
        # ! Based on the registered Patches (Local OCGPatches per room)
        # ! a global set of bins are computed
        self.global_ocg_patch.update_bins()
        self.global_ocg_patch.update_ocg_map(binary_map=True)

        [r.set_status(ROOM_STATUS.OVERLAPPING) for r in self.list_rooms]
        for room_ocg_map, room in zip(self.global_ocg_patch.ocg_map, self.list_rooms):
            if room.status != ROOM_STATUS.OVERLAPPING:
                # ! This avoid to process merged rooms
                continue
            for tmp_ocg_map, tmp_room in zip(self.global_ocg_patch.ocg_map, self.list_rooms):
                if tmp_room is room:
                    continue
                iou = compute_iou_ocg_map(
                    ocg_map_target=room_ocg_map,
                    ocg_map_estimation=tmp_ocg_map
                )
                if iou > self.dt.cfg.get("room_id.iuo_overlapping_allowed", 0.25):
                    # ! Exits an overlapping betwen rooms
                    # *(1) merge rooms and append at the edn of list_rooms (change ready flag)
                    self.merge_rooms(room, tmp_room)

        self.delete_rooms()
        [r.set_status(False) for r in self.list_rooms]

    def delete_rooms(self):
        """
        Deletes the rooms which are defined for deletion
        """
        list_rooms = []
        [list_rooms.append(r) for r in self.list_rooms
         if r.status != ROOM_STATUS.FOR_DELETION
         ]

        if list_rooms.__len__() > 0:
            self.list_rooms = list_rooms
            self.global_ocg_patch.list_patches = [r.local_ocg_patches for r in list_rooms]

    def merge_rooms(self, room_a, room_b):
        """
        Merge the data of the passed rooms as follows

        # * (1) The rooms are set as ROOM_STATUS.FOR_DELETION
        # * (2) Create new room with merged data:
        # ** updating METADATA for new ROOM
        # ** layouts, planes, local_ocg_patches
        # *(3) Update global_ocg_map accordanally
        """
        # ! New Room in the system
        new_room = Room(self.dt)
        # ! Adding Metadata
        new_room.add_metadata(room_a)
        new_room.add_metadata(room_b)
        new_room.refresh()

        new_room.compute_orientations()
        new_room.set_status(ROOM_STATUS.MERGED)

        room_a.set_status(ROOM_STATUS.FOR_DELETION)
        room_b.set_status(ROOM_STATUS.FOR_DELETION)

        self.list_rooms.append(new_room)
        self.global_ocg_patch.list_patches.append(new_room.local_ocg_patches)

        if room_a is self.curr_room or room_b is self.curr_room:
            self.curr_room = new_room

    def compute_room_shape_all(self):
        ''' Compute the room shape for all the rooms at once '''
        room_corners = []
        for room in tqdm(self.list_rooms):
            if room.status == ROOM_STATUS.FOR_DELETION:
                continue
            out_dict = room.compute_room_shape()
            room_corners.append(out_dict['corners_xz'].T)
        return room_corners
