from .ocg_patch import OCGPatches
from direct_floor_plan_estimation.solvers.theta_estimator import ThetaEstimator
from direct_floor_plan_estimation.solvers.room_shape_estimator import RoomShapeEstimator
import numpy as np


class Room:
    def __init__(self, data_manager):
        self.dt = data_manager
        self.list_ly = []
        self.list_pl = []
        self.is_initialized = False
        self.local_ocg_patches = OCGPatches(self.dt)
        self.theta_estimator = ThetaEstimator(self.dt)
        self.room_shape_estimator = RoomShapeEstimator(self.dt)
        self.list_corners = []
        self.boundary = None

        # ! For Tracking pose likelihood
        self.p_pose = []
        # !Used for iSPA and overlapping rooms
        self.status = False

        self.theta_z = []
        self.room_center = None

    def set_status(self, flag):
        """
        Easy function to set ready instance
        """
        self.status = flag

    def initialize(self, layout):
        """
        Room class initializer
        """
        self.is_initialized = False
        if not layout.is_initialized:
            raise ValueError("Layout must be initialized to init a Room instance...")

        if not self.local_ocg_patches.initialize(layout.patch):
            return self.is_initialized

        self.list_ly.append(layout)
        [self.list_pl.append(pl) for pl in layout.list_pl]

        self.is_initialized = True

        return self.is_initialized

    def add_metadata(self, external_room):
        """
        Adds the METADATA from an external_room
        """
        [(self.list_ly.append(ly),
          self.local_ocg_patches.list_patches.append(patch),
          self.p_pose.append(pose)
          )for ly, patch, pose in
         zip(
             external_room.list_ly,
             external_room.local_ocg_patches.list_patches,
             external_room.p_pose
        )
        ]
        # ! Since the number of planes does not necessary matches the number of ly
        [self.list_pl.append(pl) for pl in external_room.list_pl]

    def refresh(self):
        self.is_initialized = True
        self.local_ocg_patches.is_initialized = True
        self.local_ocg_patches.update_bins()
        self.local_ocg_patches.update_ocg_map2()

    def compute_orientations(self):
        """
        Computes theta orientation using all plane registered
        """
        # return
        self.room_center = np.mean(np.vstack([ly.pose.t for ly in self.list_ly]), axis=0)

        self.theta_z = self.theta_estimator.estimate_from_list_pl(
            list_pl=self.list_pl,
            room_center=self.room_center
        )

    def add_layout(self, layout):
        """
        Adds a new layout to the ROOM
        """
        assert layout.is_initialized, "Passed layout must be initialized first... "
        # ! Adding Layouts
        self.list_ly.append(layout)
        # ! Adding Planes
        [self.list_pl.append(pl) for pl in layout.list_pl]
        # ! Adding Patch
        self.local_ocg_patches.add_patch(layout.patch)
        # ! Updating LOCAL OCG-map
        self.local_ocg_patches.update_ocg_map2()
        # ! Compute Orientations
        self.compute_orientations()

    def compute_room_shape(self, dump_dir=None, room_idx=None):
        return self.room_shape_estimator.estimate(self, dump_dir=dump_dir, room_idx=room_idx)
