
from cv2 import sampsonDistance
from vispy.visuals.transforms import STTransform, MatrixTransform
from pyquaternion import Quaternion
import numpy as np
import vispy
from vispy import visuals, scene
import sys
from utils.image_utils import get_color_list
from zmq import DEALER


def setting_viewer(return_canvas=False, main_axis=True, bgcolor='black'):
    import vispy.scene
    from vispy.scene import visuals

    canvas = vispy.scene.SceneCanvas(keys='interactive',
                                     show=True,
                                     bgcolor=bgcolor)
    size_win = 1024
    canvas.size = 2*size_win, size_win

    view = canvas.central_widget.add_view()
    view.camera = 'arcball'  # turntable / arcball / fly / perspective

    if main_axis:
        visuals.XYZAxis(parent=view.scene)

    if return_canvas:
        return view, canvas
    return view


def setting_pcl(view, size=5, edge_width=2, antialias=0):
    from vispy.scene import visuals
    from functools import partial
    scatter = visuals.Markers()
    scatter.set_gl_state('translucent',
                         depth_test=True,
                         blend=True,
                         blend_func=('src_alpha', 'one_minus_src_alpha'))
    # scatter.set_gl_state(depth_test=True)
    scatter.antialias = 0
    view.add(scatter)
    return partial(scatter.set_data, size=size, edge_width=edge_width)


def plot_color_plc(points,
                   color=(1, 1, 1, 1),
                   return_view=False,
                   size=0.5,
                   plot_main_axis=True,
                   background="black",
                   axis_frame=None,
                   scale_factor=15
                   ):
    import vispy
    from functools import partial

    # if color == (1, 1, 1):
    #     bg = "black"
    view = setting_viewer(main_axis=plot_main_axis, bgcolor=background)

    # view.camera = vispy.scene.TurntableCamera(elevation=45,
    #                                           azimuth=-135,
    #                                           roll=0,
    #                                           fov=0,
    #                                           up='-y')
    view.camera = vispy.scene.TurntableCamera(elevation=25,
                                              azimuth=-45,
                                              roll=0,
                                              fov=0,
                                              up='-y')
    # view.camera = vispy.scene.TurntableCamera(elevation=90,
    #                                           azimuth=0,
    #                                           roll=0,
    #                                           fov=0,
    #                                           up='-y')
    # view.camera = vispy.scene.TurntableCamera(elevation=0, azimuth=180, roll=0, fov=0, up='-y')
    view.camera.scale_factor = scale_factor
    draw_pcl = setting_pcl(view=view)
    draw_pcl(points, edge_color=color, size=size)
    # pose1 = np.eye(4)
    # sphere(view, pose1, size=1, alpha=0.5)
    # pose1[0, 3] = axis_frame[0]
    # camera_frame(view, pose1, size=axis_frame, width=2)
    # pose1[0, 3] = axis_frame[1]
    # camera_frame(view, pose1, size=3, width=2)
    if not return_view:
        vispy.app.run()
    else:
        return view


def plot_list_pcl(list_pcl, size=1):
    """
    Plot the list of pcl (3, n) by the passed sizes
    """

    colors = get_color_list(number_of_colors=list_pcl.__len__())
    color_pcl = []
    for i, pcl in enumerate(list_pcl):
        color_pcl.append(np.ones_like(pcl)*colors[:, i].reshape(3, 1))
    
    plot_color_plc(
        points=np.hstack(list_pcl).T,
        color=np.hstack(color_pcl).T,
        size=size
    )
