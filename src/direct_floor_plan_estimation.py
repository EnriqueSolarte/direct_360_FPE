from cv2 import sepFilter2D
from src.scale_recover import ScaleRecover
from src.solvers.theta_estimator import ThetaEstimator
from src.solvers.plane_estimator import PlaneEstimator
from src.data_structure import OCGPatch
from utils.enum import CAM_REF


class DirectFloorPlanEstimation:

    def __init__(self, data_manager):
        self.dt = data_manager
        self.scale_recover = ScaleRecover(self.dt)
        self.theta_estimator = ThetaEstimator(self.dt)
        self.plane_estimator = PlaneEstimator(self.dt)
        self.ocg_map = OCGPatch(self.dt)
        self.list_ly = []
        self.list_pl = []

        self.list_rooms = []

        self.curr_room = None
        self.initialized = False

        print("DirectFloorPlanEstimation initialized successfully")

    def estimate(self, layout):
        """
        It add the passed Layout to the systems and estimated the floor plan
        """
        
        if not self.initialized:
            self.initialize()
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
                
        if not self.curr_room.initialized:
            if not self.curr_room.initialize(layout):
                return


        self.list_ly.append(layout)
        self.update_ocg()
        self.eval_ocg_overlapping()

    def initialize(self):
        """
        Initializes the system
        """
        # ! getting few lys for initialize scale
        list_ly = self.dt.get_list_ly(
            cam_ref=CAM_REF.WC_SO3,
            ratio=self.dt.cfg['scale_recover.ratio_for_initialization']
        )

        pass
