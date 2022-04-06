import os
import numpy as np
from tqdm import tqdm

from src.scale_recover import ScaleRecover
from src.solvers.plane_estimator import PlaneEstimator
from src.solvers.room_shape_estimator import SPAError
from src.data_structure import OCGPatches, Room
from utils.geometry_utils import find_N_peaks
from utils.ocg_utils import compute_iou_ocg_map
from utils.enum import ROOM_STATUS, CAM_REF
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

    def compute_non_sequential_fpe(self):
        """
        For debugging purposes only. This method estimates
        a FPE based on all LYs and rooms ROOM-ID knowing in advance
        """
        print("Runing Non-sequential estimation")
        list_ly = self.dt.get_list_ly(cam_ref=CAM_REF.WC_SO3)

        # ! Computes vo-scale
        if self.dt.cfg.get("scale_recover.apply_gt_scale", False):
            if not self.scale_recover.estimate_vo_and_gt_scale():
                raise ValueError("Scale recovering failed")
        else:
            if not self.scale_recover.estimate_vo_scale():
                raise ValueError("Scale recovering failed")

        self.is_initialized = True

        for kfs in self.dt.list_kf_per_room:
            print(f"Running: {self.dt.scene_name} - eval-version:{self.dt.cfg['eval_version']}")

            list_ly_per_room = [ly for ly in list_ly
                                if ly.idx in kfs
                                ]

            [self.initialize_layout(ly) for ly in list_ly_per_room]
            [self.add_layout(ly) for ly in list_ly_per_room]

           # ! Initialize current room
            self.curr_room = Room(self.dt)
            if not self.curr_room.initialize(list_ly_per_room[0]):
                raise ValueError("Somtheing is went wrong...!!!")

            self.list_rooms.append(self.curr_room)
            [self.curr_room.add_layout(ly) for ly in list_ly_per_room[1:]]

        if not self.global_ocg_patch.initialize(self.list_rooms[0].local_ocg_patches):
            raise ValueError("Somtheing is went wrong...!!!")

        [self.global_ocg_patch.list_patches.append(r.local_ocg_patches) for r in self.list_rooms[1:]]
        self.global_ocg_patch.update_bins()
        self.global_ocg_patch.update_ocg_map(binary_map=True)
        print("done")

    def estimate(self, layout):
        """
        It adds the passed Layout to the systems and estimated the floor plan
        """
        # print(f"Running: {self.dt.scene_name} - eval-version:{self.dt.cfg['eval_version']}")
        if not self.is_initialized:
            self.initialize(layout)
            return

        if not self.initialize_layout(layout):
            return layout.is_initialized

        if self.eval_new_room_creation(layout):
            self.eval_room_overlapping()
            # prev_room = self.curr_room
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
        # # plot_estimated_orientations(self.curr_room.theta_z)

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

        if self.dt.cfg.get("scale_recover.apply_gt_scale", False):
            if not self.scale_recover.estimate_vo_and_gt_scale():
                return self.is_initialized
        else:
            if not self.scale_recover.estimate_vo_scale():
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

    def compute_iou_overlapping(self, ocg_map_a, ocg_map_b):
        """
        Computes the IoU metrics for the passed ocg_map
        """
        assert self.dt.cfg["room_id.iou_overlapping_norm"] in ("min", "max", "union")
        if self.dt.cfg["room_id.iou_overlapping_norm"] == "union":
            iou = compute_iou_ocg_map(
                ocg_map_target=ocg_map_a,
                ocg_map_estimation=ocg_map_b
            )
            return iou
        elif self.dt.cfg["room_id.iou_overlapping_norm"] == "min":
            intersection = (ocg_map_a + ocg_map_b) > 1
            iou = np.sum(intersection)/np.min((np.sum(ocg_map_a), np.sum(ocg_map_b)))
            return iou
        elif self.dt.cfg["room_id.iou_overlapping_norm"] == "max":
            intersection = (ocg_map_a + ocg_map_b) > 1
            iou = np.sum(intersection)/np.max((np.sum(ocg_map_a), np.sum(ocg_map_b)))
            return iou
        else:
            raise ValueError("Error")

    def eval_room_overlapping(self):
        """
        Merges Rooms based on the overlapping of OCGPatches (local_ocg_map)
        """
        # ! Based on the registered Patches (Local OCGPatches per room)
        # ! a global set of bins are computed
        self.global_ocg_patch.update_bins()
        self.global_ocg_patch.update_ocg_map(binary_map=True)

        [r.set_status(ROOM_STATUS.OVERLAPPING) for r in self.list_rooms]
        assert len(self.global_ocg_patch.ocg_map) == len(self.list_rooms)
        for room_ocg_map, room in zip(self.global_ocg_patch.ocg_map, self.list_rooms):
            if room.status != ROOM_STATUS.OVERLAPPING:
                # ! This avoid to process merged rooms
                continue
            iou_meas = []
            rooms_candidates = []
            for tmp_ocg_map, tmp_room in zip(self.global_ocg_patch.ocg_map, self.list_rooms):
                if tmp_room is room:
                    continue
                
                iou = self.compute_iou_overlapping(room_ocg_map, tmp_ocg_map)
                iou_meas.append(iou)
                rooms_candidates.append(tmp_room)
                # if iou > self.dt.cfg.get("room_id.iou_overlapping_allowed", 0.25):
                # ! Exits an overlapping betwen rooms
                # *(1) merge rooms and append at the edn of list_rooms (change ready flag)
                # self.merge_rooms(room, tmp_room)
            if np.sum(np.array(iou_meas) > self.dt.cfg.get("room_id.iou_overlapping_allowed", 0.25)) > 0:
                self.merge_rooms(room, rooms_candidates[np.argmax(iou_meas)])

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
        assert len(self.list_rooms) == len(self.global_ocg_patch.list_patches)

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

    def compute_room_shape_all(self, plot=False):
        ''' Compute the room shape for all the rooms at once '''
        room_corners = []
        for room_idx, room in enumerate(tqdm(self.list_rooms, desc="Running iSPA...")):
            if room.status == ROOM_STATUS.FOR_DELETION:
                continue
            try:
                if plot:
                    dump_dir = os.path.join(self.dt.cfg.get("results_dir"), self.dt.scene_name)
                    os.makedirs(dump_dir, exist_ok=True)
                else:
                    dump_dir = None
                out_dict = room.compute_room_shape(
                    room_idx=room_idx,
                    dump_dir=dump_dir,
                )
                room_corners.append(out_dict['corners_xz'].T)
            except SPAError as e:
                print(e)
        return room_corners
