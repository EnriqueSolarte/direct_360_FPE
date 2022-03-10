from config import read_config, overwrite_scene_data
from config.config import overwite_version
from data_manager import DataManager
from src import DirectFloorPlanEstimation
from utils.visualization.vispy_utils import plot_color_plc
from utils.enum import CAM_REF
import numpy as np
from utils.data_utils import flatten_lists_of_lists
import matplotlib.pyplot as plt
from utils.visualization.room_utils import plot_curr_room_by_patches, plot_all_rooms_by_patches
from utils.visualization.room_utils import plot_floor_plan
from utils.room_id_eval_utils import eval_2D_room_id_iou, sumarize_restults_room_id_iou
from utils.io import read_csv_file, save_csv_file
import os


def main(config_file, scene_list_file, dump_dir, version):
    # ! Reading list of scenes
    list_scenes = read_csv_file(scene_list_file)

    # ! creating dump_dir for save results
    os.makedirs(dump_dir, exist_ok=True)
    cfg = read_config(config_file=config_file)

    # for thr in (0.1, 0.25, 0.5, 0.7, 0.8):
    #     version += str(thr)
    #     cfg["room_id.ocg_threshold"] = thr
    # for overl in (0.1, 0.25, 0.5, 0.7, 0.8):
    #     version += str(overl)
    #     cfg["room_id.iuo_overlapping_allowed"] = overl
    # for clip_dis in (1, 2, 5, 10, 15, 20, 30):
    #     version += str(clip_dis)
    #     cfg["room_id.clipped_ratio"] = clip_dis
        
    # ! Running every scene
    for scene in list_scenes:
        overwrite_scene_data(cfg, scene)
        overwite_version(cfg, version)
        
        dt = DataManager(cfg)

        fpe = DirectFloorPlanEstimation(dt)
        list_ly = dt.get_list_ly(cam_ref=CAM_REF.WC_SO3)

        for ly in list_ly:
            fpe.estimate(ly)

        fpe.eval_room_overlapping()
        
        # fpe.global_ocg_patch.update_bins()
        # fpe.global_ocg_patch.update_ocg_map()
        # plot_all_rooms_by_patches(fpe, only_save=True)

        # exit()
        eval_2D_room_id_iou(fpe)
        fpe.dt.save_config()
        
        # sumarize_restults_room_id_iou(fpe)


if __name__ == '__main__':
    # TODO read from  passed args
    config_file = "./config/config.yaml"
    scene_list_file = './data/all_scenes_list.csv'
    dump_dir = './fpe_results'
    version = 'eval_thr0.8_ov0.8_clip10_max_inter'

    main(config_file, scene_list_file, dump_dir, version)
