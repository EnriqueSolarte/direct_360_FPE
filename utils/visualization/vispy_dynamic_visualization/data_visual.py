from config import *
from vispy.scene import visuals
import vispy
from vispy.visuals.transforms import STTransform
from vispy.scene.visuals import Text
from vispy import app, scene, io
import numpy as np
from utils.image_utils import get_color_list
from pyquaternion import Quaternion
from vispy.visuals.transforms import STTransform, MatrixTransform


class CameraPoseVisual:
    def __init__(self, size=0.5, width=2, view=None):
        self.initial_pose = np.eye(4)
        self.width = width
        self.size = size
        # self.axis_z = scene.visuals.create_visual_node(visuals.ArrowVisual)
        self.base = np.zeros((4, 1))
        self.base[3, 0] = 1
        self.view = view
        self.isthereCam = False
        self.prev_pose = np.eye(4)

    def add_camera(self, pose, color, plot_sphere=True):
        if plot_sphere:
            self.sphere = scene.visuals.Sphere(
                radius=self.size*0.5,
                # radius=1,
                method='latitude',
                parent=self.view.scene,
                color=color)
        else:
            self.sphere = None

        self.initial_pose = pose
        pose_w = pose
        if self.sphere is not None:
            self.sphere.transform = STTransform(translate=pose_w[0:3, 3].T)
        # scene.visuals.Arrow
        self.isthereCam = True
        x = self.base + np.array([[self.size], [0], [0], [0]])
        y = self.base + np.array([[0], [self.size], [0], [0]])
        z = self.base + np.array([[0], [0], [self.size], [0]])

        pts = np.hstack([self.base, x, y, z])
        pts = np.dot(pose_w, pts)

        pos = np.zeros((2, 3))
        pos[0, :] = pts[0:3, 0]
        pos[1, :] = pts[0:3, 3]
        self.axis_z = scene.Line(pos=pos, color=(0, 0, 1), method='gl', parent=self.view.scene)
        self.axis_z.set_data(pos=pos, color=(0, 0, 1))

        pos = np.zeros((2, 3))
        pos[0, :] = pts[0:3, 0]
        pos[1, :] = pts[0:3, 1]
        self.axis_x = scene.Line(pos=pos, color=(1, 0, 0), method='gl', parent=self.view.scene)
        self.axis_x.set_data(pos=pos, color=(1, 0, 0))

        pos = np.zeros((2, 3))
        pos[0, :] = pts[0:3, 0]
        pos[1, :] = pts[0:3, 2]
        self.axis_y = scene.Line(pos=pos, color=(0, 1, 0), method='gl', parent=self.view.scene)
        self.axis_y.set_data(pos=pos, color=(0, 1, 0))

        # self.axis_z(pos, width=self.width, color=(1, 1, 1), parent=self.view.scene)

    def transform_camera(self, pose):
        pose_w = pose @ np.linalg.inv(self.initial_pose)
        if self.sphere is not None:
            self.sphere.transform = STTransform(translate=pose[0:3, 3].T)
        q = Quaternion(matrix=pose_w[0:3, 0:3])
        trf = MatrixTransform()
        trf.rotate(angle=np.degrees(q.angle), axis=q.axis)
        trf.translate(pose_w[0:3, 3])
        self.axis_z.transform = trf
        self.axis_x.transform = trf
        self.axis_y.transform = trf


class Visualization:
    def __init__(self):
        # ! Setting up vispy
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', bgcolor='black')
        res = 1024 * 2
        self.canvas.size = res, res//2
        self.canvas.show()

        # ! for visualization of more than one grid
        self.vb1 = scene.widgets.ViewBox(parent=self.canvas.scene)
        # self.vb2 = scene.widgets.ViewBox(parent=self.canvas.scene)
        # self.vb3 = scene.widgets.ViewBox(parent=self.canvas.scene)

        visuals.XYZAxis(parent=self.vb1.scene)
        grid = self.canvas.central_widget.add_grid()
        grid.padding = 6
        grid.add_widget(self.vb1, 0, 0)
        # grid.add_widget(self.vb2, 0, 1)
        # grid.add_widget(self.vb3, 0, 2)

        # self.vb1.camera = vispy.scene.TurntableCamera(elevation=35,
        #                                               azimuth=-165,
        #                                               roll=0,
        #                                               fov=10,
        #                                               up='-y')
        self.vb1.camera = vispy.scene.TurntableCamera(elevation=35,
                                                      azimuth=150,
                                                      roll=0,
                                                      fov=10,
                                                      up='-y')
        # self.vb2.camera = vispy.scene.TurntableCamera(elevation=35,
        #                                               azimuth=150,
        #                                               roll=0,
        #                                               fov=10,
        #                                               up='-y')

        # self.vb3.camera = vispy.scene.TurntableCamera(elevation=35,
                                                    #   azimuth=150,
                                                    #   roll=0,
                                                    #   fov=10,
                                                    #   up='-y')

        # self.vb1.camera = vispy.scene.TurntableCamera(elevation=-90,
        #                                               azimuth=90,
        #                                               roll=0,
        #                                               fov=10,
        #                                               up='-y')
        # # # self.vb1.camera = vispy.scene.TurntableCamera(elevation=25,
        #                                               azimuth=-10,
        #                                               roll=0,
        #                                               fov=0,
        #                                               up='-y')
        # self.vb1.camera.scale_factor = 100
        # self.vb1.camera.scale_factor = 10

        # self.vb2.camera = vispy.scene.TurntableCamera(elevation=90,
        #                                               azimuth=-165,
        #                                               roll=0,
        #                                               fov=10,
        #                                               up='-y')

        self.vb1.camera.scale_factor = 20
        # self.vb2.camera.scale_factor = 20
        # self.vb3.camera.scale_factor = 20

        # self.vb1.camera.link(self.vb2.camera)
        # self.vb1.camera.link(self.vb3.camera)

        self.list_visuals = []
        self.pcl_visual_gt = DataVisual()
        self.pcl_visual_est = DataVisual()
        self.pcl_visual_ly = DataVisual()

        self.pcl_visual_gt.set_view(self.vb1)
        self.pcl_visual_est.set_view(self.vb1)
        self.pcl_visual_ly.set_view(self.vb1)

        self.colors_list = get_color_list(number_of_colors=100)


class DataVisual:
    def __init__(self):
        self.cam_color = np.array((0, 255, 255))/255
        self.second_color = np.array((1, 0, 1))
        self.red_color = np.array((1, 0.1, 0.1))
        self.gt_color = np.array((0, 255, 0))/255
        self.yellow_color = np.array((255, 255, 0))/255
        self.white_color = np.array((255, 255, 255))/255
        self.black_color = np.array((0, 0, 0))
        self.scatter_bounds = visuals.Markers()
        self.scatter_bounds.set_gl_state('translucent',
                                         depth_test=True,
                                         blend=True,
                                         blend_func=('src_alpha', 'one_minus_src_alpha'))
        # scatter.set_gl_state(depth_test=True)
        self.scatter_bounds.antialias = 0

        self.scatter_normals = visuals.Markers()
        self.scatter_normals.set_gl_state('translucent',
                                          depth_test=True,
                                          blend=True,
                                          blend_func=('src_alpha', 'one_minus_src_alpha'))
        # scatter.set_gl_state(depth_test=True)
        self.scatter_normals.antialias = 0

        self.scatter_position = visuals.Markers()
        self.scatter_position.set_gl_state('translucent',
                                           depth_test=True,
                                           blend=True,
                                           blend_func=('src_alpha', 'one_minus_src_alpha'))
        # scatter.set_gl_state(depth_test=True)
        self.scatter_position.antialias = 0

        self.scatter_planes = visuals.Markers()
        self.scatter_planes.set_gl_state('translucent',
                                         depth_test=True,
                                         blend=True,
                                         blend_func=('src_alpha', 'one_minus_src_alpha'))
        # scatter.set_gl_state(depth_test=True)
        self.scatter_planes.antialias = 0

        self.scatter_corners = visuals.Markers()
        self.scatter_corners.set_gl_state('translucent',
                                          depth_test=True,
                                          blend=True,
                                          blend_func=('src_alpha', 'one_minus_src_alpha'))
        # scatter.set_gl_state(depth_test=True)
        self.scatter_corners.antialias = 0

        self.list_scatter = []
        self.cam = CameraPoseVisual()
        self.aux_cams = [CameraPoseVisual()]

    @staticmethod
    def set_cam_view(cam, view):
        cam.view = view

    def set_view(self, view):
        self.set_cam_view(self.cam, view)
        [self.set_cam_view(cam, view) for cam in self.aux_cams]

        view.add(self.scatter_bounds)
        view.add(self.scatter_corners)
        view.add(self.scatter_planes)
        view.add(self.scatter_normals)
        view.add(self.scatter_position)

    def plot_pcl(self, pcl, cam=np.eye(4), size=0.5, color=None):
        if color is None:
            if pcl.shape[1] > 3:
                color = pcl[:, 3:]
            else:
                color = self.white_color

        self.scatter_corners.set_data(pcl[:, 0:3], edge_color=color,
                                      size=size)

        if not self.cam.isthereCam:
            self.cam.add_camera(cam, self.cam_color)
        else:
            self.cam.transform_camera(cam)
