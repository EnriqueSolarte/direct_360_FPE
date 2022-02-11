from imageio import imread
import numpy as np

class Depth:
    def __init__(self, frame):
        self.frame = frame
        self.gt_pose = self.frame.gt_pose
        self.est_pose = self.frame.est_pose
        
        idx = self.frame.dt.kf_list.index(self.frame.idx) 
        
        npy_depth = self.frame.dt.list_depth_npy[idx]
        rgb_file = self.frame.dt.list_rgb_img[idx]
        depth_file = self.frame.dt.list_depth_maps[idx]

        self.frame.rgb_map = imread(rgb_file)
        self.rgb_map = self.frame.rgb_map 
        
        self.depth_map_gt = imread(depth_file)
        self.depth_map_est =np.load(npy_depth)     
        
       
        
