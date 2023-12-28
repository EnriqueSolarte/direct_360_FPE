
from .data_visual import Visualization
from vispy import app
import sys
import numpy as np
idx = 0


def plot_dyn_from_list_ly(list_ly):
    vis = Visualization()

    all_ly = np.hstack([ly.boundary for ly in list_ly])
    
    
    def update(ev):
        global idx

        ly = list_ly[idx]        
        
        vis.pcl_visual_est.plot_pcl(
            pcl=all_ly.T,
            color=(0.5, 0.5, 1)        
        )
        vis.pcl_visual_ly.plot_pcl(
            pcl=ly.boundary.T,
            cam=ly.pose.SE3_scaled(), 
            color=(1, 0, 0),
            size=2
        )

        idx = (idx + 1) % list_ly.__len__()

    timer = app.Timer(0.1)
    timer.connect(update)
    timer.start()
    vis.canvas.show()
    if sys.flags.interactive == 0:
        app.run()
