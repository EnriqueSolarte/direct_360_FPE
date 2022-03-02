import sys
import os
import glob
import json
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from pyquaternion import Quaternion
from skimage.measure import points_in_poly


def read_json_label(fn):
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


def read_ply(fn):
    plydata = PlyData.read(fn)
    v = np.array([list(x) for x in plydata.elements[0]])
    points = np.ascontiguousarray(v[:, :3])
    points[:, 0:3] = points[:, [0, 2, 1]]
    colors = np.ascontiguousarray(v[:, 3:6], dtype=np.float32) / 255
    return np.concatenate((points, colors), axis=1)


def write_ply(points, output_path):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'),
             ('green', 'u1'), ('blue', 'u1')]
    dtype.extend([('nx', '<f4'), ('ny', '<f4'),
                  ('nz', '<f4')]) if points.shape[1] == 9 else None
    if (points[:, 3:6] <= 1).all():
        points[:, 3:6] = points[:, 3:6] * 255
    vertex = np.array([tuple(x) for x in points], dtype=dtype)
    vertex_el = PlyElement.describe(vertex, 'vertex')
    PlyData([vertex_el]).write(output_path)


def read_floorsp_pred(fn):
    x = np.load(fn, allow_pickle=True).item()['vectorized_preds']
    return [[corner['corner'][0], corner['corner'][1]] for corner in x]


def read_scene_list(fn):
    with open(fn, 'r') as f:
        return sorted(f.read().strip().split('\n'))


def estimate_normals(points):
    xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals()
    normals = pcd.normals
    return np.concatenate((points, normals), axis=1)


def rotate_by_axis_corners(points, axis_corners):
    points = points.copy()
    x, y = axis_corners[1] - axis_corners[0]
    yaw_angle = np.arctan(y / x)
    yaw_2d = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle)],
                       [np.sin(yaw_angle),
                        np.cos(yaw_angle)]])

    points[:, :2] = np.matmul(points[:, :2], yaw_2d)
    if points.shape[-1] > 6:
        # has normal
        yaw_3d = np.eye(3)
        yaw_3d[:2, :2] = yaw_2d
        points[:, 6:] = np.matmul(points[:, 6:], yaw_3d)
    return points


def merge_neighbor_corners(corners, dist_threshold):
    '''
        Naively merge all corners by DFS
        corners: shape (N, 2)
        dist_threshold: float

        return: merged corners with shape (M, 2)
    '''
    dist_matrix = np.expand_dims(corners, axis=1) - np.expand_dims(
        corners, axis=0)  # N, N, 2
    dist_matrix = np.sqrt(np.sum(np.power(dist_matrix, 2), axis=-1))  # N, N
    mask = dist_matrix < dist_threshold

    def dfs(i, graph, used_node):
        collected = set()
        used_node[i] = True
        collected.add(i)

        for j in range(graph.shape[1]):
            if graph[i, j] and not used_node[j]:
                collected = collected | dfs(j, graph, used_node)
        return collected

    used_node = np.zeros((corners.shape[0], ))
    merged_corners = []
    for i in range(corners.shape[0]):
        if not used_node[i]:
            corners_ids = list(dfs(i, mask, used_node))
            merged_corners.append(np.mean(corners[corners_ids, :], axis=0))

    merged_corners = np.stack(merged_corners, axis=0)
    return merged_corners


class FloorNetCornerAlignment:
    '''
        Align the corners based on the FloorNet and Floor-SP style.
    '''
    def __init__(
        self,
        points,
        axis_corners=None,
        padding_ratio=0.05,
    ):
        self.axis_corners = axis_corners
        if self.axis_corners is not None:
            points = rotate_by_axis_corners(points, self.axis_corners)

        assert len(points.shape) == 2
        if points.shape[1] > 2:
            # Use only the x, y coordinate
            points = points[:, :2]

        # Map the point coordinate into 0-1
        # where the point cloud is in the middle with a padding
        self.max_edge = np.max(points.max(0) - points.min(0))
        self.padding = self.max_edge * padding_ratio
        self.mins = (points.max(0) + points.min(0)) / 2 - self.max_edge / 2
        self.mins -= self.padding
        self.max_edge += self.padding * 2

        self.aligned_points = (points - self.mins) / self.max_edge

    def align_corners(self, corners):
        if self.axis_corners is not None:
            corners = rotate_by_axis_corners(corners, self.axis_corners)
        return (corners - self.mins) / self.max_edge

    def inverse_align_corners(self, corners):
        corners = corners * self.max_edge + self.mins
        if self.axis_corners is not None:
            # Inverse rotate
            corners = corners.copy()
            x, y = self.axis_corners[1] - self.axis_corners[0]
            yaw_angle = np.arctan(y / x)
            yaw_2d = np.array([[np.cos(yaw_angle), -np.sin(yaw_angle)],
                               [np.sin(yaw_angle),
                                np.cos(yaw_angle)]])
            corners[:, :2] = np.matmul(corners[:, :2], yaw_2d.T)
        return corners


def mytransform44(l, seq="xyzw"):
    """
    Generate a 4x4 homogeneous transformation matrix from a 3D point and unit quaternion.

    Input:
    l -- tuple consisting of (stamp,tx,ty,tz,qx,qy,qz,qw) where
         (tx,ty,tz) is the 3D position and (qx,qy,qz,qw) is the unit quaternion.

    Output:
    matrix -- 4x4 homogeneous transformation matrix
    """
    t = l[1:4]
    q = np.array(l[4:8], dtype=np.float64, copy=True)
    if seq == 'wxyz':
        if q[0] < 0:
            q *= -1
        q = Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    else:
        if q[3] < 0:
            q *= -1
        q = Quaternion(
            x=q[0],
            y=q[1],
            z=q[2],
            w=q[3],
        )
    trasnform = np.eye(4)
    trasnform[0:3, 0:3] = q.rotation_matrix
    trasnform[0:3, 3] = np.array(t)

    return trasnform


def read_trajectory(filename, matrix=True, traj_gt_keys_sorted=[], seq="xyzw"):
    """
    Read a trajectory from a text file. 
    
    Input:
    filename -- file to be read_datasets
    matrix -- convert poses to 4x4 matrices
    
    Output:
    dictionary of stamped 3D poses
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    list = [[float(v.strip()) for v in line.split(" ") if v.strip() != ""]
            for line in lines if len(line) > 0 and line[0] != "#"]
    list_ok = []
    for i, l in enumerate(list):
        if l[4:8] == [0, 0, 0, 0]:
            continue
        isnan = False
        for v in l:
            if np.isnan(v):
                isnan = True
                break
        if isnan:
            sys.stderr.write(
                'Warning: line {} of file {} has NaNs, skipping line\n'.format(
                    i, filename))
            continue
        list_ok.append(l)
    if matrix:
        traj = dict([(l[0], mytransform44(l[0:], seq=seq)) for l in list_ok])
    else:
        traj = dict([(l[0], l[1:8]) for l in list_ok])

    return traj


def make_unvisited_area(base_dir,
                        scene_name,
                        gt_corners,
                        min_key_frames_per_room=5):
    """Mask out unvisited room masks and corners."""
    scene, version = scene_name.split('_')
    pose_fn = os.path.join(base_dir, scene, version, 'frm_ref.txt')
    kf_fn = os.path.join(base_dir, scene, version + '/*/keyframe_list.txt')
    kf_fn = glob.glob(kf_fn)

    assert len(kf_fn) == 1, 'Keyframe list found error. Size: ' + str(len(kf_fn))
    with open(kf_fn[0], 'r') as f:
        kf_list = [int(kf_id) for kf_id in f.read().split()]

    poses_gt = read_trajectory(pose_fn)
    gt_poses = [list(poses_gt.values())[idx - 1] for idx in kf_list]

    all_cam_poses = np.vstack([pose[0:3, 3] for pose in gt_poses])
    masked_in_corners = []
    masked_out_corners = []
    for local_corners_gt in gt_corners:
        mask = points_in_poly(all_cam_poses[:, (0, 2)], local_corners_gt)
        if np.sum(mask) <= min_key_frames_per_room:
            masked_out_corners.append(local_corners_gt)
            continue
        masked_in_corners.append(local_corners_gt)
    return masked_in_corners, masked_out_corners
