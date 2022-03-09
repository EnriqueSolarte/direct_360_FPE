from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.geometry_utils import extend_array_to_homogeneous
from utils.ocg_utils import compute_iou_ocg_map
from skimage.color import rgb2hsv, hsv2rgb
from utils.io import read_csv_file, save_csv_file
import os
import pandas as pd


def eval_2D_room_id_iou(fpe, save=True):
    """
    computes 2D IoU per estimated ROOM
    * For debugging purposes only
    """
    results = []

    global_map = np.ones((fpe.global_ocg_patch.H, fpe.global_ocg_patch.W*2, 3))
    global_map[:, :, 1] = 0
    colors = np.linspace(0, 0.9, fpe.dt.room_corners.__len__())

    for idx, cr in enumerate(fpe.dt.room_corners):
        # ! GT rooms
        gt_room = np.zeros(fpe.global_ocg_patch.get_shape())
        cr_xyz = extend_array_to_homogeneous(cr.T)[(0, 2, 1), :]
        cr_px = fpe.global_ocg_patch.project_xyz_to_uv(cr_xyz)
        cv2.fillPoly(gt_room, [cr_px.T], (1, 1, 1))

        # ! Estimated rooms
        eval_iou = []
        for est_room in fpe.global_ocg_patch.ocg_map:
            # plt.imshow(est_room)
            est_room[est_room/est_room.max() < fpe.dt.cfg.get("room_id.ocg_threshold")] = 0
            est_room[est_room > 0] = 1
            iou = compute_iou_ocg_map(
                ocg_map_target=gt_room,
                ocg_map_estimation=est_room
            )
            eval_iou.append(iou)

        best_iou = np.max(eval_iou)
        est_room = fpe.global_ocg_patch.ocg_map[np.argmax(eval_iou), :, :]
        est_room[est_room/est_room.max() < fpe.dt.cfg.get("room_id.ocg_threshold")] = 0
        est_room[est_room > 0] = 1

        results.append(
            (f"{fpe.dt.cfg['scene']}_{fpe.dt.cfg['scene_version']}_room{idx}",
             best_iou
             )
        )

        # ! Plotting GT and estimation figure
        comb_map = np.hstack((gt_room, est_room))
        mask = comb_map > 0
        global_map[mask, 0] = colors[idx]
        global_map[mask, 1] = 1
        global_map[mask, 2] = comb_map[mask]

    metadata = f"{fpe.dt.cfg.get('room_id.ocg_threshold')}_{fpe.dt.cfg.get('room_id.clipped_ratio')}_{fpe.dt.cfg.get('room_id.iuo_overlapping_allowed')}"
    metadata += "_non_iso_norm_per_esp"
    metadata += "_no_wtemp"
    file_results = os.path.join(fpe.dt.cfg.get("results_dir"), f"room_id_iou_results_{metadata}.csv")
    figure_results = os.path.join(fpe.dt.cfg.get("results_dir"), f"{fpe.dt.scene_name}_{metadata}.jpg")

    # ! Saving filename results
    fpe.dt.cfg['results.room_id_iou'] = file_results
    
    os.makedirs(fpe.dt.cfg.get("results_dir"), exist_ok=True)
    
    
    # ! Save results
    if os.path.exists(file_results):
        # ! Eval pre-exist results
        eval_data = pd.read_csv(fpe.dt.cfg['results.room_id_iou'], header=None, delimiter=',').values
        if f"{fpe.dt.cfg['scene']}_{fpe.dt.cfg['scene_version']}_room{idx}" in eval_data[:, 0]:
            return
        save_csv_file(f"{file_results}", results, flag="a")
    else:
        save_csv_file(f"{file_results}", results, flag="w+")

    global_map = hsv2rgb(global_map)
    plt.figure("room_id_eval")
    plt.clf()
    plt.title(f"{fpe.dt.scene_name}_{metadata}")
    plt.imshow(global_map)
    plt.savefig(figure_results)



def restults_2D_room_id_iou(fpe):
    results = pd.read_csv(fpe.dt.cfg['results.room_id_iou'], header=None, delimiter=',').values
    q25 = np.quantile(results[:, 1], 0.25)
    q50 = np.quantile(results[:, 1], 0.5)
    q75 =np.quantile(results[:, 1], 0.75)
    mean_ = np.mean(results[:, 1])
    print("TOTAL RESULTS")
    print(f"Q25: {q25}")
    print(f"Q50: {q50}")
    print(f"Q75: {q75}")
    print(f"mean: {mean_}")
    