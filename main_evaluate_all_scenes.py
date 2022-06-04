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
from utils.io import read_scene_list
from utils.eval_utils import evaluate_scene, dump_images, dump_result


def main(config_file, scene_list_file, output_dir):
    cfg = read_config(config_file=config_file)

    scene_list = read_scene_list(scene_list_file)
    all_result = []
    for i, scene in enumerate(scene_list):
        cfg['data.scene'], cfg['data.scene_version'] = scene.split('_')

        dt = DataManager(cfg)
        fpe = DirectFloorPlanEstimation(dt)
        list_ly = dt.get_list_ly(cam_ref=CAM_REF.WC_SO3)

        for ly in list_ly:
            fpe.estimate(ly)

        fpe.global_ocg_patch.update_bins()
        fpe.global_ocg_patch.update_ocg_map()
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
    # TODO read from passed args
    config_file = "./config/config.yaml"
    scene_list_file = './data/scene_list_50_multi_room.txt'
    output_dir = './test/eval_all'
    main(config_file, scene_list_file, output_dir)
