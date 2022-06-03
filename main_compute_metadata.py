import os
import numpy as np
import matplotlib.pyplot as plt
from config import read_config
from data_manager import DataManager
from src import DirectFloorPlanEstimation
from utils.visualization.vispy_utils import plot_color_plc
from utils.enum import CAM_REF
from utils.data_utils import flatten_lists_of_lists
from utils.visualization.room_utils import plot_curr_room_by_patches, plot_all_rooms_by_patches
from utils.visualization.room_utils import plot_floor_plan, plot_all_planes, plot_planes_rooms_patches
# from utils.metric import evaluate_rooms_pr
from utils.eval_utils import evaluate_corners_pr, evaluate_rooms_pr
from utils.io import read_csv_file, read_scene_list
import yaml


def compute_metadata(config_file, scene_list_file, output_dir):
    """
    Computes the metadata for the scene listed in @scene_list_file using the configuration file @config_file 
    """
    
    cfg = read_config(config_file=config_file)
    scene_list = read_csv_file(scene_list_file)
    for scene in scene_list:
     
        # ! Overrite data.* values using scene_list (iter)
        if scene.split('/').__len__() > 1:
            cfg['data.scene_category'], cfg['data.scene'], cfg['data.scene_version'] = scene.split('/')
        else:
            cfg['data.scene'], cfg['data.scene_version'] = scene.split('_')

        dt = DataManager(cfg)
        fpe = DirectFloorPlanEstimation(dt)
        # ! Set the list_ly into dt class
        dt.get_list_ly(cam_ref=CAM_REF.WC_SO3)

        metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(metadata_dir, exist_ok=True)
       
        # ! Saving GT Room data
        dt.save_gt_rooms(metadata_dir)
        
        if not fpe.scale_recover.estimate_vo_and_gt_scale():
            raise ValueError("Scale recovering failed")

        # ! Saving VO-Scale recovery
        fpe.scale_recover.save_estimation(metadata_dir)


if __name__ == '__main__':
    # TODO read from passed args
    config_file = "./config/config.yaml"
    scene_list_file = './data/all_scenes_list.csv'
    output_dir = "./test/metadata"
    compute_metadata(config_file, scene_list_file, output_dir)
