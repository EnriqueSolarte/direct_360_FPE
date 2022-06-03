import sys
import json
import numpy as np
from plyfile import PlyData
from pyquaternion import Quaternion
import csv
import os


def get_list_scenes(data_dir, flag_file, exclude=""):
    """
    Searches in the the @data_dir, recurrently, the sub-directories which content the @flag_file.
    exclude directories can be passed to speed up the rearching
    """
    scenes_paths = []
    for root, dirs, files in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        [scenes_paths.append(root) for f in files if flag_file in f]
    return scenes_paths


def read_csv_file(filename):
    with open(filename) as f:
        csvreader = csv.reader(f)

        lines = []
        for row in csvreader:
            lines.append(row[0])
    return lines


def save_csv_file(filename, data, flag='w'):
    with open(filename, flag) as f:
        writer = csv.writer(f)
        for line in data:
            writer.writerow([l for l in line])
    f.close()


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
    transform = np.eye(4)
    transform[0:3, 0:3] = q.rotation_matrix
    transform[0:3, 3] = np.array(t)

    return transform


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


def read_ply(fn):
    plydata = PlyData.read(fn)
    v = np.array([list(x) for x in plydata.elements[0]])
    points = np.ascontiguousarray(v[:, :3])
    # points[:, 0:3] = points[:, [0, 2, 1]]
    colors = np.ascontiguousarray(v[:, 3:6], dtype=np.float32) / 255
    return np.concatenate((points, colors), axis=1)


def read_scene_list(fn):
    with open(fn, 'r') as f:
        return f.read().strip().split('\n')
