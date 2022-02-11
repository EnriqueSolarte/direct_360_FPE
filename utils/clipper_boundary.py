import numpy as np 

class ClipperBoundary: 
    
    def __init__(self, data_manager):
        self.dt = data_manager
        
    
    def clip_pcl(self, pcl, return_mask=True):
        """clips the passed pcl assuming a close boundary 
        """
        
        sample = np.zeros((2, pcl.shape[1]))
        sample[0, :] = pcl[0, :]
        sample[1, :] = pcl[2, :]
        
        norm  = np.linalg.norm(sample, axis=0)
        mask = norm < self.dt.cfg['clipper.radius']
        
        if return_mask:
            return mask
        else:
            return pcl[:, mask]