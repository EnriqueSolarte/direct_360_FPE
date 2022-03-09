import os
import glob
import json
import numpy as np
from tqdm import tqdm
from utils.camera_models.sphere import Sphere
from utils.enum import *
from src.data_structure import CamPose
from utils.io import read_trajectory, read_ply
from utils.geometry_utils import get_bearings_from_phi_coords, extend_array_to_homogeneous
from src.data_structure import Layout
from skimage.measure import points_in_poly


class DataManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_paths()
        self.load_data()

        self.list_ly = []
        print("DataManager successfully loaded...")
        print(f"Scene Category: {self.cfg['scene_category']}")
        print(f"Scene: {self.scene_name}")

    def set_paths(self):
        """
        Sets all paths necessary for the DataManager
        """
        try:
            self.scene_name = self.cfg['scene'] + '_' + self.cfg['scene_version']
            self.mp3d_fpe_dir = self.cfg["mp3d_fpe_dir"]
            self.vo_dir = glob.glob(os.path.join(self.mp3d_fpe_dir, 'vo*'))[0]
        except:
            print(f"ERROR AT READING SCENE --> {self.mp3d_fpe_dir}")
            raise ValueError("Data_manager couldn't access to the data..")

    def load_data(self):
        """
        Loads all data for DataManager
        """
        # ! List of Kf
        try:
            with open(os.path.join(self.vo_dir, 'keyframe_list.txt'), 'r') as f:
                self.list_kf = sorted([int(kf) for kf in f.read().splitlines()])

            # ! List of camera poses
            self.load_camera_poses()

            # ! List of LY estimations
            self.list_ly_npy = [os.path.join(self.vo_dir, self.cfg['ly_model'], f'{f}.npy') for f in self.list_kf]

            # ! List of RGB images
            self.list_rgb_img = [os.path.join(self.mp3d_fpe_dir, f'rgb/{f}.png') for f in self.list_kf]

            # ! List of DepthGT maps
            self.list_depth_maps = [os.path.join(self.mp3d_fpe_dir, f'depth/tiff/{f}.tiff') for f in self.list_kf]

            # ! Load GT floor plan & point cloud
            self.room_corners, self.axis_corners = self.load_fp_gt(os.path.join(self.mp3d_fpe_dir, 'label.json'))

            # NOTE: pcl_gt is (N, 3) and z-axis is the height
            self.pcl_gt = read_ply(os.path.join(self.mp3d_fpe_dir, 'pcl.ply'))

            self.compute_kf_per_room()
            self.cam = Sphere(shape=self.cfg['image_resolution'])
        except:
            raise ValueError("Data_manager couldn't access to the data..")

    def compute_kf_per_room(self):
        """
        Computes the kf stored per room for evaluation purposes
        """
        label = os.path.join(self.mp3d_fpe_dir, 'label.json')

        with open(label) as data_file:
            data_loaded = json.load(data_file)

        if 'kf_per_room' in data_loaded.keys():
            # ! Loading list KF per ROOM
            self.list_kf_per_room = data_loaded['kf_per_room']

        else:
            # ! Computes Kf per room and save them into JSON file
            corners = data_loaded["room_corners"]
            all_cam_poses_t = self.poses_gt[:, 0:3, 3]
            self.list_kf_per_room = []
            for room_id, local_corners_gt in enumerate(corners):

                mask = points_in_poly(all_cam_poses_t[:, (0, 2)],
                                      local_corners_gt)

                if np.sum(mask) == 0:
                    print('Room GT {} does not have any Kf'.format(room_id))
                    continue
                kf = [f for f, m in zip(self.list_kf, mask) if m]
                self.list_kf_per_room.append(kf)

            data_loaded['kf_per_room'] = self.list_kf_per_room
            with open(f"{self.mp3d_fpe_dir}/label.json", 'w', encoding='utf-8') as f:
                json.dump(data_loaded, f, ensure_ascii=False, indent=4)

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
        self.poses_est = np.stack(
            list(read_trajectory(estimated_poses_file).values()))

        # ! Loading GT camera poses
        gt_poses_file = os.path.join(
            self.mp3d_fpe_dir,
            'frm_ref.txt')

        assert os.path.isfile(
            gt_poses_file
        ), f'Cam pose file {gt_poses_file} does not exist'

        idx = np.array(self.list_kf)-1
        self.poses_gt = np.stack(
            list(read_trajectory(gt_poses_file).values()))[idx, :, :]

    def load_fp_gt(self, fn):
        with open(fn, 'r') as f:
            d = json.load(f)
            room_list = d['room_corners']
            room_corners = []
            for corners in room_list:
                corners = np.asarray([[float(x[0]), float(x[1])] for x in corners])
                room_corners.append(corners)
            axis_corners = d['axis_corners']
            axis_corners = np.asarray([[float(x[0]), float(x[1])]
                                       for x in axis_corners])
        return room_corners, axis_corners

    def load_points_gt(self, fn):
        plydata = PlyData.read(fn)
        v = np.array([list(x) for x in plydata.elements[0]])
        points = np.ascontiguousarray(v[:, :3])
        points[:, 0:3] = points[:, [0, 2, 1]]
        colors = np.ascontiguousarray(v[:, 3:6], dtype=np.float32) / 255
        return np.concatenate((points, colors), axis=1)

    def get_list_ly(self, cam_ref=CAM_REF.CC):
        """
        Returns a list of layouts (Layout class) for the scene 
        """
        list_ly = []
        for idx_kf in tqdm(self.list_kf, desc="Loading Layouts..."):
            idx = self.list_kf.index(idx_kf)

            pose_est = CamPose(self, pose=self.poses_est[idx])
            pose_est.idx = idx_kf

            pose_gt = CamPose(self, pose=self.poses_gt[idx])
            pose_gt.idx = idx_kf

            # * Every npy file content data estimated from CNN layout estimation in camera coordinates
            # *(NO WC--> no world coordinates)
            data_ly = np.load(self.list_ly_npy[idx])
            # > data[0] is floor
            # > data[1] is ceiling,
            # > data[2] are corners
            # ! Note: HorizonNet defines in other reference floor & ceiling (sign are different)
            # ! Possible BUG in HorizonNet floor--> ceiling
            data_ly[(0, 1), :] = -data_ly[(1, 0), :]

            bearings_phi = data_ly[0, :]

            bearings = get_bearings_from_phi_coords(phi_coords=bearings_phi)

            # !Projecting bearing to 3D as pcl --> boundary
            ly_scale = self.cfg['ly_scale'] / bearings[1, :]
            # pcl = (1-pose.t[1])*ly_scale * bearings
            pcl = ly_scale * bearings

            if cam_ref == CAM_REF.WC_SO3:
                pcl = pose_est.rot @ pcl
            elif cam_ref == CAM_REF.WC:
                pcl = pose_est.SE3_scaled()[0:3, :] @ extend_array_to_homogeneous(pcl)

            # > Projecting PCL into zero-plane
            # pcl[1, :] = 0  # TODO verify if this is really needed

            ly = Layout(self)
            ly.bearings = bearings
            ly.boundary = pcl
            ly.pose_est = pose_est
            ly.pose_gt = pose_gt
            ly.idx = pose_est.idx
            ly.ly_data = data_ly
            ly.cam_ref = cam_ref
            # ly.estimate_height_ratio()
            ly.compute_cam2boundary()
            list_ly.append(ly)

        self.list_ly = list_ly
        return list_ly
