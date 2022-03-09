from config import read_config, overwrite_scene_data
from data_manager import DataManager
from src import DirectFloorPlanEstimation
from utils.visualization.vispy_utils import plot_color_plc
from utils.enum import CAM_REF
import numpy as np
from utils.data_utils import flatten_lists_of_lists
import matplotlib.pyplot as plt
from utils.visualization.room_utils import plot_curr_room_by_patches, plot_all_rooms_by_patches
from utils.visualization.room_utils import plot_floor_plan
from utils.room_id_eval_utils import eval_2D_room_id_iou, restults_2D_room_id_iou
from utils.io import read_csv_file, save_csv_file
import os


def main(config_file):
    list_scenes = read_csv_file(
        os.path.join(os.path.dirname(__file__), "data", "scene_list.csv")
    )
    for scene in list_scenes:
        cfg = read_config(config_file=config_file)
        overwrite_scene_data(cfg, scene)
        dt = DataManager(cfg)

        fpe = DirectFloorPlanEstimation(dt)
        list_ly = dt.get_list_ly(cam_ref=CAM_REF.WC_SO3)

        for ly in list_ly:
            fpe.estimate(ly)

        fpe.global_ocg_patch.update_bins()
        fpe.global_ocg_patch.update_ocg_map()
        eval_2D_room_id_iou(fpe)
    
    restults_2D_room_id_iou(fpe)
    
        

if __name__ == '__main__':
    # TODO read from  passed args
    config_file = "./config/config.yaml"
    main(config_file=config_file)
