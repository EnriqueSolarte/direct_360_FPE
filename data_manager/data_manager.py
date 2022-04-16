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
import datetime
import sys
import yaml
import matplotlib.pyplot as plt



class DataManager:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_paths()
        self.load_data()

        self.list_ly = []
        print("DataManager successfully loaded...")
        print(f"Scene Category: {self.cfg['data.scene_category']}")
        print(f"Scene: {self.scene_name}")

    def set_paths(self):
        """
        Sets all paths necessary for the DataManager
        """
        try:
            self.scene_name = self.cfg['data.scene'] + '_' + self.cfg['data.scene_version']
            self.mp3d_fpe_scene_dir = os.path.join(
                self.cfg['path.mp3d_fpe_dir'],
                self.cfg['data.scene_category'],
                self.cfg['data.scene'],
                self.cfg['data.scene_version']
            )
            self.mp3d_fpe_scene_vo_dir = glob.glob(os.path.join(self.mp3d_fpe_scene_dir, 'vo*'))[0]
        except:
            print(f"ERROR AT READING SCENE --> {self.mp3d_fpe_scene_dir}")
            raise ValueError("Data_manager couldn't access to the data..")

    def load_data(self):
        """
        Loads all data for DataManager
        """
        # ! List of Kf
        try:
            with open(os.path.join(self.mp3d_fpe_scene_vo_dir, 'keyframe_list.txt'), 'r') as f:
                self.list_kf = sorted([int(kf) for kf in f.read().splitlines()])

            # ! List of camera poses
            self.load_camera_poses()

            # ! List of LY estimations
            self.list_ly_npy = [os.path.join(self.mp3d_fpe_scene_vo_dir, self.cfg['data.ly_model'], f'{f}.npy') for f in self.list_kf]

            # ! List of RGB images
            self.list_rgb_img = [os.path.join(self.mp3d_fpe_scene_dir, f'rgb/{f}.png') for f in self.list_kf]

            # ! List of DepthGT maps
            self.list_depth_maps = [os.path.join(self.mp3d_fpe_scene_dir, f'depth/tiff/{f}.tiff') for f in self.list_kf]

            # ! Load GT floor plan & point cloud
            self.room_corners, self.axis_corners, self.list_kf_per_room = \
                self.load_fp_gt(os.path.join(self.mp3d_fpe_scene_dir, 'label.json'))

            # NOTE: pcl_gt is (N, 3) and z-axis is the height
            self.pcl_gt = read_ply(os.path.join(self.mp3d_fpe_scene_dir, 'pcl.ply'))

            self.cam = Sphere(shape=self.cfg['data.image_resolution'])
        except:
            raise ValueError("Data_manager couldn't access to the data..")

    def load_camera_poses(self):
        """
        Load both GT and estimated camera poses
        """
        # ! Loading estimated poses
        estimated_poses_file = os.path.join(
            self.mp3d_fpe_scene_vo_dir,
            'cam_pose_estimated.csv')

        assert os.path.isfile(
            estimated_poses_file
        ), f'Cam pose file {estimated_poses_file} does not exist'
        self.poses_est = np.stack(
            list(read_trajectory(estimated_poses_file).values()))

        # ! Loading GT camera poses
        gt_poses_file = os.path.join(
            self.mp3d_fpe_scene_dir,
            'frm_ref.txt')

        assert os.path.isfile(
            gt_poses_file
        ), f'Cam pose file {gt_poses_file} does not exist'

        idx = np.array(self.list_kf)-1
        self.poses_gt = np.stack(
            list(read_trajectory(gt_poses_file).values()))[idx, :, :]

    def load_fp_gt(self, fn):
        """
        Load GT information defined in the label.json file. Additionally, it clusters kf per room
        and masks out room which are defined without frame inside
        """
        with open(fn, 'r') as f:
            d = json.load(f)
            self.cfg['data.label_version'] = d['version']
            room_list = d['room_corners']
            room_corners = []
            for corners in room_list:
                corners = np.asarray([[float(x[0]), float(x[1])] for x in corners])
                room_corners.append(corners)
            axis_corners = d['axis_corners']
            axis_corners = np.asarray([[float(x[0]), float(x[1])]
                                       for x in axis_corners])

        # Compute kf per room and mask out unvisited GT rooms
        poses_gt_xyz = self.poses_gt[:, :3, 3]
        kf_per_room = []
        masked_room_corners = []
        for corners in room_corners:
            cr = np.vstack((corners, corners[0, :]))
            mask = points_in_poly(poses_gt_xyz[:, [0, 2]], cr)
            if np.sum(mask) == 0:
                continue
            kf = [f for f, m in zip(self.list_kf, mask) if m]
            kf_per_room.append(kf)
            masked_room_corners.append(corners)
        assert len(masked_room_corners) > 0
        return masked_room_corners, axis_corners, kf_per_room

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
            ly_scale = self.cfg['data.ly_scale'] / bearings[1, :]
            # pcl = (1-pose.t[1])*ly_scale * bearings
            pcl = ly_scale * bearings

            if cam_ref == CAM_REF.WC_SO3:
                pcl = pose_est.rot @ pcl
            elif cam_ref == CAM_REF.WC:
                pcl = pose_est.SE3_scaled()[0:3, :] @ extend_array_to_homogeneous(pcl)

            # > Projecting PCL into zero-plane
            pcl[1, :] = 0  # TODO verify if this is really needed
            cov = pcl @ pcl.T
            _, s, _ = np.linalg.svd(cov/(pcl.size - 1))

            ly = Layout(self)
            ly.bearings = bearings
            ly.boundary = pcl
            ly.pose_est = pose_est
            ly.pose_gt = pose_gt
            ly.idx = pose_est.idx
            ly.ly_data = data_ly
            ly.cam_ref = cam_ref
            ly.sigma_ratio = (s[1]/s[0])
            # ly.estimate_height_ratio()
            ly.compute_cam2boundary()
            list_ly.append(ly)

        self.list_ly = list_ly
        return list_ly

    def save_config(self, filename=None):
        """
        Saves the current configuration (settings) in a yaml file
        """
        time = datetime.datetime.now()

        timestamp = str(time.year) + "-" + str(time.month) + "-" + str(time.day) + \
            "." + str(time.hour) + '.' + str(time.minute) + '.' + str(time.second)
        original_stdout = sys.stdout  # Save a reference to the original standard output

        filename = os.path.join(self.cfg.get("results_dir"), "saved_config.yaml")
        with open(filename, "w") as file:
            yaml.dump(self.cfg, file)

            sys.stdout = file  # Change the standard output to the file we created.

            # ! This is the comment at the end of every generated config file
            print('\n\n# VSLAB @ National Tsing Hua University')
            print("# Direct Floor Plan Estimation")
            print("# This config file has been generated automatically")
            print("#")
            print('# {}'.format(timestamp))
            sys.stdout = original_stdout

    def save_gt_rooms(self, output_dir):
        """
        Saves the gt rooms defined for the scene
        """
        data = dict()
        assert self.list_kf_per_room.__len__() == self.room_corners.__len__()
        plt.figure("GT-Rooms")
        plt.clf()
        plt.title(f"GT-Rooms - label version:{self.cfg['data.label_version']}")
        for idx, list_kf, corners in zip(range(self.room_corners.__len__()), self.list_kf_per_room, self.room_corners):
            cr = [[float(c[0]), float(c[1])] for c in corners]
            data[f"room.{idx}"] = dict(
                list_kf=list_kf,
                corners=cr
            )
            vis = np.vstack((corners, corners[0, :]))
            plt.plot(vis[:, 0], vis[:, 1])
            
            # ! plotting KF poses
            kf_poses = np.vstack([self.poses_gt[self.list_kf.index(i)][(0, 2), 3] for i in list_kf])
                # ! Plotting rooms
            plt.scatter(kf_poses[:, 0], kf_poses[:, 1])
        
        plt.axis('equal')
        plt.draw()
        plt.savefig(os.path.join(output_dir, f"gt_rooms_{self.cfg['data.label_version']}.jpg"), bbox_inches='tight') 
        data["data.scene"] = self.cfg["data.scene"]
        data["data.scene_version"] = self.cfg["data.scene_version"]
        data["data.scene_category"] = self.cfg["data.scene_category"]
        data["data.label_version"] = self.cfg["data.label_version"]
        
        filename = os.path.join(output_dir, f"room_gt_{data['data.label_version']}.yaml")
        with open(filename, "w") as file:
            yaml.dump(data, file)
        
       