
from .data_visual import Visualization
from vispy import app
import sys
idx = 0


def plot_pcl_from_list_fr(list_fr):
    vis = Visualization()

    def update(ev):
        global idx

        fr = list_fr[idx]        
        
        vis.pcl_visual_gt.plot_pcl(
            pcl=fr.pcl_gt.T,
            cam=fr.gt_pose.SE3)

        vis.pcl_visual_est.plot_pcl(
            pcl=fr.pcl_est.T,
            cam=fr.gt_pose.SE3)

        vis.pcl_visual_ly.plot_pcl(
            pcl=fr.layout.boundary_floor.T,
            cam=fr.est_pose.SE3_scaled()
        )

        idx = (idx + 1) % list_fr.__len__()

    timer = app.Timer(0.5)
    timer.connect(update)
    timer.start()
    vis.canvas.show()
    if sys.flags.interactive == 0:
        app.run()
