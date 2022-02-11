from src.scale_recover import ScaleRecover
from src.solvers.theta_estimator import ThetaEstimator
from src.solvers.plane_estimator import PlaneEstimator
from src.data_structure import OCGPatch
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
        pass