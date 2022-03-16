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


def evaluate_scene(room_corners_pred, room_corners_gt, points, axis_corners=None):
    '''
    Evaluate the scene with room and corner metric
        corners_pred: list of predicted room corners
        corners_gt: list of GT room corners
        gt_scale:
    '''
    return_dict = {}
    images_dict = {}
    for iou_threshold in [0.3, 0.5, 0.7]:
        num_match, size_pred, size_gt, image_pred, image_gt = evaluate_rooms_pr(
            room_corners_pred, room_corners_gt,
            points, axis_corners,
            grid_size=512,
            iou_threshold=iou_threshold,
            room_name_list=None,
        )
        print(f'Room@{iou_threshold} recall: {num_match / size_gt}, precision: {num_match / size_pred}')
        return_dict[f'room@{iou_threshold}'] = dict(
            num_match=num_match,
            size_pred=size_pred,
            size_gt=size_gt,
        )
        images_dict[f'room@{iou_threshold}'] = dict(
            image_pred=image_pred,
            image_gt=image_gt,
        )

    corners_pred = np.concatenate(room_corners_pred, axis=0)
    corners_gt = np.concatenate(room_corners_gt, axis=0)
    for do_merge in [True, False]:
        num_match, size_pred, size_gt, image_pred, image_gt = evaluate_corners_pr(
            corners_pred, corners_gt,
            points, axis_corners,
            grid_size=256,
            merge_corners=do_merge,
            merge_dist=0.5,
            dist_threshold=10,
        )
        print(f'Corner recall: {num_match / size_gt}, precision: {num_match / size_pred}')
        key = 'corner_merge' if do_merge else 'corner_raw'
        return_dict[key] = dict(
            num_match=num_match,
            size_pred=size_pred,
            size_gt=size_gt,
        )
        images_dict[key] = dict(
            image_pred=image_pred,
            image_gt=image_gt,
        )
    return return_dict, images_dict


def dump_images(result_dict, save_dir):
    # Dump images
    scene = result_dict['scene']
    keys = sorted(list(result_dict.keys()))
    keys.remove('scene')
    keys.remove('patches')
    plt.imsave(os.path.join(save_dir, f'{scene}_patches.png'), result_dict['patches'])
    for key in keys:
        plt.imsave(os.path.join(save_dir, f'{scene}_{key}_gt.png'), result_dict[key]['image_gt'])
        plt.imsave(os.path.join(save_dir, f'{scene}_{key}_pred.png'), result_dict[key]['image_pred'])


def dump_result(result_list, save_dir):
    # Dump result to a csv file
    with open(os.path.join(save_dir, '360_dfpe_result.csv'), 'w') as f:
        keys = sorted(list(result_list[0].keys()))
        keys.remove('scene')
        # Write header
        f.write('scene')
        for key in keys:
            f.write(f',{key}_recall,{key}_prec')
        f.write('\n')

        for result_dict in result_list:
            # Write each row (repective to the result of a scene)
            scene = result_dict['scene']
            f.write(f'{scene}')
            for key in keys:
                recall = result_dict[key]['num_match'] / result_dict[key]['size_gt']
                prec = result_dict[key]['num_match'] / result_dict[key]['size_pred']
                f.write(f',{recall:.4f},{prec:.4f}')
            f.write('\n')

        # Compute average result for last row
        f.write('mean')
        for key in keys:
            all_recall = []
            all_prec = []
            for result_dict in result_list:
                recall = result_dict[key]['num_match'] / result_dict[key]['size_gt']
                prec = result_dict[key]['num_match'] / result_dict[key]['size_pred']
                all_recall.append(recall)
                all_prec.append(prec)
            f.write(f',{np.mean(all_recall):.4f},{np.mean(all_prec):.4f}')
        f.write('\n')


def main(config_file, scene_list_file):
    cfg = read_config(config_file=config_file)
    dump_dir = os.path.join(os.getenv("RESULTS_DIR"), cfg.get("version", "test_evaluation"))
    os.makedirs(dump_dir, exist_ok=True)
    scene_list = read_scene_list(scene_list_file)
    all_result = []
    try:
        for i, scene in enumerate(scene_list):
            cfg['scene'], cfg['scene_version'] = scene.split('_')

            dt = DataManager(cfg)
            fpe = DirectFloorPlanEstimation(dt)
            list_ly = dt.get_list_ly(cam_ref=CAM_REF.WC_SO3)

            for ly in list_ly:
                fpe.estimate(ly)

            fpe.global_ocg_patch.update_bins()
            fpe.global_ocg_patch.update_ocg_map()
            points_gt = fpe.dt.pcl_gt      # (3, N)
            debug_image1 = plot_planes_rooms_patches(fpe, points_gt)
            # plot_all_planes(fpe)

            # list_pl = flatten_lists_of_lists([ly.list_pl for ly in list_ly if ly.list_pl.__len__() > 0])
            # plot_color_plc(np.hstack([ly.boundary for ly in list_pl]).T)

            # plot_floor_plan(room_corner_list, fpe.global_ocg_patch, points_gt=points_gt)

            room_corner_list = fpe.compute_room_shape_all()
            debug_image2 = plot_planes_rooms_patches(fpe, points_gt, room_corner_list=room_corner_list, draw_plane=False)
            room_corner_list = [x.T for x in room_corner_list]  # Make it (N, 2)
            result_dict, images_dict = evaluate_scene(
                room_corner_list,
                fpe.dt.room_corners,
                points_gt,
                axis_corners=fpe.dt.axis_corners
            )
            result_dict['scene'] = scene
            images_dict['scene'] = scene
            images_dict['patches'] = plot_all_rooms_by_patches(fpe)
            all_result.append(result_dict)

            plt.figure("Debug floor plan plot")
            plt.subplot(211)
            plt.imshow(debug_image1)
            plt.subplot(212)
            plt.imshow(debug_image2)
            plt.savefig(os.path.join(dump_dir, scene, 'deubug.png'))
            plt.draw()
            plt.waitforbuttonpress(1)
            dump_images(images_dict, dump_dir)
            dump_result(all_result, dump_dir)

    except Exception as e:
        # Dump result before raising the error
        dump_result(all_result, dump_dir)
        raise e

    print('Saving the result')
    dump_result(all_result, dump_dir)
    print('done')


if __name__ == '__main__':
    # TODO read from passed args
    config_file = "./config/config.yaml"
    scene_list_file = './data/scene_list_50.txt'
    main(config_file, scene_list_file)
