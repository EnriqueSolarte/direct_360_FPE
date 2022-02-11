import os
import glob
import numpy as np
from utils.io import *
from tqdm import tqdm
from utils.geometry_utils import get_bearings_from_phi_coords
from utils.geometry_utils import extend_array_to_homogeneous
from utils.camera_models.sphere import Sphere
from imageio import imread
from utils import ScaleRecover


class DataManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_paths()
        self.load_data()
        self.list_frames = []
        print("DataManager successfully loaded...")

    def set_paths(self):
        """
        Sets all paths necessary for the DataManager
        """
        self.scene_name = self.cfg['scene'] + '_' + self.cfg['scene_version']
        self.mp3d_fpe_dir = self.cfg["mp3d_fpe_dir"]
        self.vo_dir = glob.glob(os.path.join(self.mp3d_fpe_dir, 'vo_*'))[0]

    def load_data(self):
        """
        Loads all data for DataManager
        """
        # ! List of Kf
        with open(os.path.join(self.vo_dir, 'keyframe_list.txt'), 'r') as f:
            self.kf_list = sorted([int(kf) for kf in f.read().splitlines()])

        # ! List of camera poses
        self.load_camera_poses()

        #! List of LY estimations
        self.list_ly_npy = [os.path.join(self.vo_dir, self.cfg['ly_model'], f'{f}.npy') for f in self.kf_list]

        # ! List of RGB images
        self.list_rgb_img = [os.path.join(self.mp3d_fpe_dir, f'rgb/{f}.png') for f in self.kf_list]

        # !List of DepthGT maps
        self.list_depth_maps = [os.path.join(self.mp3d_fpe_dir, f'depth/tiff/{f}.tiff') for f in self.kf_list]

        self.cam = Sphere(shape=self.cfg['image_resolution'])

    def load_camera_poses(self):
        """
        Load both GT and estimated camera poses
        """
        # ! Loading estimated poses
        estimated_poses_file = os.path.join(
            self.vo_dir,
            'cam_pose_estimated.csv')

        assert os.path.isfile(
            estimated_poses_file
        ), f'Cam pose file {estimated_poses_file} does not exist'
        self.est_poses = np.stack(
            list(read_trajectory(estimated_poses_file).values()))

        # ! Loading GT camera poses
        gt_poses_file = os.path.join(
            self.mp3d_fpe_dir,
            'frm_ref.txt')

        assert os.path.isfile(
            gt_poses_file
        ), f'Cam pose file {gt_poses_file} does not exist'

        idx = np.array(self.kf_list)-1
        self.gt_poses = np.stack(
            list(read_trajectory(gt_poses_file).values()))[idx, :, :]

    def get_list_ly(self, clipped=False):
        """
        Returns a list the ly objects. It indirectly sets the list of frames
        However, only Ly data is set on that list.
        """
        list_ly = []
        for idx_kf in tqdm(self.kf_list, desc=f'Loading data Frames...'):
            idx = self.kf_list.index(idx_kf)
            # ! npy file
            npy_ly = self.list_ly_npy[idx]

            est_pose = CamPose(
                config=self.cfg,
                pose=self.est_poses[idx, :, :],
                vo_scale=self.cfg["vo_scale"]
            )

            gt_pose = CamPose(
                config=self.cfg,
                pose=self.gt_poses[idx, :, :])
            est_pose.idx = gt_pose.idx = idx_kf

            # * Every npy file content data estimated from CNN layout estimation in camera coordinates
            # * (NO WC--> no world coordinates)

            data_layout = np.load(npy_ly)
            # > data[0] is floor
            # > data[1] is ceiling
            # > data[2] are corners

            # ! Note: HorizonNet defines in other reference floor & ceiling (sign are different)
            # ! Possible BUG in HorizonNet floor--> ceiling
            data_layout[(0, 1), :] = -data_layout[(1, 0), :]

            bearings_floor = get_bearings_from_phi_coords(
                phi_coords=data_layout[0, :])
            bearings_ceiling = get_bearings_from_phi_coords(
                phi_coords=data_layout[1, :])

            # ! Projecting bearing to 3D as pcl --> boundary
            # > Forcing ly-scale = 1
            ly_scale = self.cfg["ly_scale"] / bearings_floor[1, :]

            pcl_floor = est_pose.rot @ (ly_scale * bearings_floor)
            pcl_ceiling = est_pose.rot @ (ly_scale * bearings_ceiling)

            if clipped:
                # ! Clipping boundary. Only the closset point are accepted
                mask = self.clipper.clip_pcl(pcl_ceiling)
                if np.sum(mask) < 10:
                    continue
                pcl_floor = pcl_floor[:, mask]
                pcl_ceiling = pcl_ceiling[:, mask]

            fr = Frame(self)
            fr.est_pose = est_pose
            fr.gt_pose = gt_pose
            fr.idx = idx_kf

            fr.layout = Layout(fr)
            fr.layout.boundary_floor = pcl_floor
            fr.layout.boundary_ceiling = pcl_ceiling
            fr.layout.bearings_floor = bearings_floor
            fr.layout.bearings_ceiling = bearings_ceiling
            fr.layout.reference = "WC_SO3"
            # * Data stored in npy file (estimated data)
            fr.layout.ly_data = data_layout

            self.list_frames.append(fr)
            list_ly.append(fr.layout)

        self.scale_recover.fully_vo_scale_estimation(list_ly)
        return list_ly

    def get_point_cloud(self, label='gt', clipped=False):
        """
        Returns the stacked PCL for the current scene (6, n). Colors included
        """
        if self.list_frames.__len__() == 0:
            raise NotImplemented()
            return

        pcl = []
        for fr in tqdm(self.list_frames, desc="Projecting depth to pcl..."):
            if fr.depth is None:
                fr.depth = Depth(fr)

            if label == 'gt':
                depth_map = fr.depth.depth_map_gt
            elif label == 'estimated':
                depth_map = fr.depth.depth_map_est
            else:
                raise NotImplemented()

            pcl_ = self.cam.project_pcl(
                color_map=fr.rgb_map,
                depth_map=depth_map,
            )

            if clipped:
                # ! Clipping boundary. Only the closset point are accepted
                mask = self.clipper.clip_pcl(pcl_[0])
                if np.sum(mask) < 10:
                    continue
                pcl_ = [pcl_[0][:, mask], pcl_[1][:, mask]]

            pcl_[0] = fr.gt_pose.SE3[
                0:3, :] @ extend_array_to_homogeneous(pcl_[0])

            pcl.append(np.vstack(pcl_))

        return np.hstack(pcl)

    def get_list_frames(self):
        """
        Returns a list of Frames, w/o Layout information 
        """
        list_frames = []

        for idx_kf in tqdm(self.kf_list, desc=f'Loading data Frames...'):

            fr = Frame(self)

            idx = self.kf_list.index(idx_kf)

            npy_depth = self.list_depth_npy[idx]
            rgb_file = self.list_rgb_img[idx]
            depth_file = self.list_depth_maps[idx]

            rgb_image = imread(rgb_file)
            depth_map_gt = imread(depth_file)

            est_pose = CamPose(
                config=self.cfg,
                pose=self.est_poses[idx, :, :],
                vo_scale=self.cfg["vo_scale"]
            )

            gt_pose = CamPose(
                config=self.cfg,
                pose=self.gt_poses[idx, :, :])
            est_pose.idx = gt_pose.idx = idx_kf

            depth_map_est = np.load(npy_depth)
            pcl_est = self.cam.project_pcl(
                color_map=rgb_image,
                depth_map=depth_map_est,
            )
            pcl_est[0] = gt_pose.SE3_scaled()[
                0:3, :] @ extend_array_to_homogeneous(pcl_est[0])

            pcl_gt = self.cam.project_pcl(
                color_map=rgb_image,
                depth_map=depth_map_gt,
            )

            pcl_gt[0] = gt_pose.SE3[
                0:3, :] @ extend_array_to_homogeneous(pcl_gt[0])

            fr = Frame(self)
            fr.est_pose = est_pose
            fr.gt_pose = gt_pose
            fr.idx = idx_kf

            fr.pcl_est = np.vstack(pcl_est)
            fr.pcl_gt = np.vstack(pcl_gt)
            fr.depth = Depth(fr)
            fr.rgb_map = rgb_image

            list_frames.append(fr)

        self.list_frames = list_frames
        return list_frames

    def set_layout_data(self):

        if self.list_frames.__len__() == 0:
            raise NotImplemented()
            return

        list_ly = []
        for fr in tqdm(self.list_frames, desc="Loading data layout..."):
            if fr.depth is None:
                fr.depth = Depth(fr)

            idx = self.kf_list.index(fr.idx)
            # ! npy file
            npy_ly = self.list_ly_npy[idx]

            data_layout = np.load(npy_ly)
            # > data[0] is floor
            # > data[1] is ceiling
            # > data[2] are corners

            # ! Note: HorizonNet defines in other reference floor & ceiling (sign are different)
            # ! Possible BUG in HorizonNet floor--> ceiling
            data_layout[(0, 1), :] = -data_layout[(1, 0), :]

            bearings_floor = get_bearings_from_phi_coords(
                phi_coords=data_layout[0, :])
            bearings_ceiling = get_bearings_from_phi_coords(
                phi_coords=data_layout[1, :])

            # ! Projecting bearing to 3D as pcl --> boundary
            # > Forcing ly-scale = 1
            ly_scale = self.cfg["ly_scale"] / bearings_floor[1, :]

            pcl_floor = fr.est_pose.rot @ (ly_scale * bearings_floor)
            pcl_ceiling = fr.est_pose.rot @ (ly_scale * bearings_ceiling)

            fr.layout = Layout(fr)
            fr.layout.boundary_floor = pcl_floor
            fr.layout.boundary_ceiling = pcl_ceiling
            fr.layout.bearings_floor = bearings_floor
            fr.layout.bearings_ceiling = bearings_ceiling
            fr.layout.reference = "WC_SO3"
            # * Data stored in npy file (estimated data)
            fr.layout.ly_data = data_layout
            list_ly.append(fr.layout)

        self.scale_recover.fully_vo_scale_estimation(list_ly)
