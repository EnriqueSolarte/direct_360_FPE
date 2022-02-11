

from cv2 import sepFilter2D


class DirectFloorPlanEstimation:

    def __init__(self, data_manager):
        self.dt = data_manager

        print("DirectFloorPlanEstimation initialized successfully")
        
    def estimate(self, layout):
        pass