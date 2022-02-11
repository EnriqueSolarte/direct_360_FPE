import numpy as np
from .data_layout import Layout
from .data_depth import Depth


class Frame:
    def __init__(self, data_manager):
                
        self.dt = data_manager
        self.est_pose = None
        self.gt_pose = None
        self.idx = None
        self.reference = None

        # * Frame data
        self.rgb_map = None
        
        # * PCL
        self.pcl_gt = None
        self.pcl_est = None

        # * Layout Info
        self.layout =  None

        # * Depth Info
        self.depth = None
        