import os
import argparse
import open3d as o3d
from utils.visualization.open3d_utils import visualize_geometry


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pkl', required=True, help='path to load pickle file.')
    parser.add_argument('--texture_dir', required=False, default=None, help='path to load config file.')

    args = parser.parse_args()
    
    dfpe_layout = visualize_geometry(pkl_geometry_path=args.pkl)
    o3d.io.write_triangle_mesh(os.path.join(os.path.dirname(args.pkl), "3D_layout.obj"), dfpe_layout)
