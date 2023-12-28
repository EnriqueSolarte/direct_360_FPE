from config import read_config, overwrite_scene_data
from config.config import overwite_version
from data_manager import DataManager
from direct_floor_plan_estimation import DirectFloorPlanEstimation
from utils.visualization.vispy_utils import plot_color_plc
from utils.enum import CAM_REF
import numpy as np
from utils.data_utils import flatten_lists_of_lists
import matplotlib.pyplot as plt
from utils.visualization.room_utils import plot_curr_room_by_patches, plot_all_rooms_by_patches
from utils.visualization.room_utils import plot_floor_plan
from utils.room_id_eval_utils import eval_2D_room_id_iou, summarize_results_room_id_iou
from utils.io import read_csv_file, save_csv_file
import os
from utils.eval_utils import evaluate_scene, dump_images, dump_result
from main_eval_list_scenes import get_passed_args


def main(config_file, scene_list_file, output_dir):
    # ! Reading list of scenes
    list_scenes = read_csv_file(scene_list_file)

    cfg = read_config(config_file=config_file)

    all_result = []
    # ! Running every scene
    for scene in list_scenes:
        cfg['data.scene'], cfg['data.scene_version'] = scene.split('_')

        dt = DataManager(cfg)

        fpe = DirectFloorPlanEstimation(dt)

        fpe.compute_non_sequential_fpe()

        fpe.masking_ocg_map()
        points_gt = fpe.dt.pcl_gt      # (3, N)

        room_corner_list = fpe.compute_room_shape_all()
        image_room_id = plot_all_rooms_by_patches(fpe)
        image_final_fp = plot_floor_plan(room_corner_list, fpe.global_ocg_patch)
        room_corner_list = [x.T for x in room_corner_list]  # Make it (N, 2)
        result_dict, images_dict = evaluate_scene(
            room_corner_list,
            fpe.dt.room_corners,
            points_gt,
            axis_corners=fpe.dt.axis_corners
        )
        result_dict['scene'] = scene
        images_dict['scene'] = scene
        images_dict['room_id'] = image_room_id
        images_dict['final_fp'] = image_final_fp

        # Saving the results
        results_dir = os.path.join(output_dir, f"{scene}")
        os.makedirs(results_dir, exist_ok=True)

        # GT data for references
        dt.save_gt_rooms(results_dir)

        # Estimated VO-SCALE and density 2d function
        fpe.scale_recover.save_estimation(results_dir)

        #  Estimated results
        dump_images(images_dict, results_dir)

        # writing results
        all_result.append(result_dict)
        dump_result(all_result, output_dir)


if __name__ == '__main__':
    opt = get_passed_args()
    main(config_file=opt.cfg, scene_list_file=opt.scene_list, output_dir=opt.results)
