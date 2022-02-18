from cv2 import sepFilter2D
from src.scale_recover import ScaleRecover
from src.solvers.theta_estimator import ThetaEstimator
from src.solvers.plane_estimator import PlaneEstimator
from src.data_structure import OCGPatch
from .data_structure import Room
from utils.geometry_utils import find_N_peaks
import numpy as np


class DirectFloorPlanEstimation:

    def __init__(self, data_manager):
        self.dt = data_manager
        self.scale_recover = ScaleRecover(self.dt)
        self.theta_estimator = ThetaEstimator(self.dt)
        self.plane_estimator = PlaneEstimator(self.dt)
        self.global_ocg_patch = OCGPatch(self.dt)
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

        self.list_ly.append(layout)
        self.apply_vo_scale(layout)
        self.compute_planes(layout)

        if self.eval_new_room_creteria(layout):
            self.curr_room = self.select_room(layout)
            if self.curr_room is None:
                # ! New Room in the system
                self.curr_room = Room(self.dt)

                if not self.curr_room.initialize(layout):
                    return

        if not self.curr_room.is_initialized:
            if not self.curr_room.initialize(layout):
                return

        self.update_ocg()
        self.eval_ocg_overlapping()

    def initialize(self, layout):
        """
        Initializes the system
        """
        if self.scale_recover.estimate_vo_scale():
            # ! Create very first Room
            self.curr_room = Room(self.dt)
            self.list_rooms.append(self.curr_room)
            self.compute_planes(layout)
            self.curr_room.list_ly.append(layout)

            self.curr_room.is_initialized = True
            # TODO initialize room ID
            # TODO initialize OCG local and global?
            self.is_initialized = True

    def apply_vo_scale(self, layout):
        """
        Applies VO-scale to the passed layout
        """
        layout.apply_vo_scale(self.scale_recover.vo_scale)
        print(f"VO-scale {self.scale_recover.vo_scale} applied to Layout {layout.idx}")

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

            list_pl.append(pl)

        layout.list_pl = list_pl
