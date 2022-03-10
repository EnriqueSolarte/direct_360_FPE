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
        corner_xyz = extend_array_to_homogeneous(cr.T)[(0, 2, 1), :]
        corner_uv = fpe.global_ocg_patch.project_xyz_to_uv(corner_xyz)
        cv2.fillPoly(gt_room, [corner_uv.T], (1, 1, 1))

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


    file_results = os.path.join(fpe.dt.cfg.get("results_dir"), 
                                f"results_room_id_iou_{fpe.dt.cfg.get('eval_version')}.csv")
    figure_results = os.path.join(fpe.dt.cfg.get("results_dir"), 
                                  f"{fpe.dt.scene_name}_{fpe.dt.cfg.get('eval_version')}.jpg")

    # ! Saving filename results
    fpe.dt.cfg['results.room_id_iou'] = file_results
    
    os.makedirs(fpe.dt.cfg.get("results_dir"), exist_ok=True)
    

    global_map = hsv2rgb(global_map)
    plt.figure("room_id_eval")
    plt.clf()
    plt.title(f"{fpe.dt.scene_name}_{fpe.dt.cfg.get('eval_version')}")
    plt.imshow(global_map)
    
    if not save:
        plt.draw()
        plt.waitforbuttonpress(0.01)
        return 
    
    # ! Save results
    if os.path.exists(file_results):
        # ! Append data
        save_csv_file(f"{file_results}", results, flag="a")
    else:
        # ! Create and write data
        save_csv_file(f"{file_results}", results, flag="w+")
    
    plt.savefig(figure_results)


def sumarize_restults_room_id_iou(fpe):
    assert os.path.exists(fpe.dt.cfg['results.room_id_iou'])
    
    file_results = os.path.join(fpe.dt.cfg.get("results_dir"), "results.room_id_iou.csv")
    results = pd.read_csv(fpe.dt.cfg['results.room_id_iou'], header=None, delimiter=',').values
    data = []
    q25 = np.quantile(results[:, 1], 0.25)
    data.append(("Q25", q25))
    q50 = np.quantile(results[:, 1], 0.5)
    data.append(("Q50", q50))
    q75 =np.quantile(results[:, 1], 0.75)
    data.append(("Q75", q75))
    mean_ = np.mean(results[:, 1])
    data.append(("mean", mean_))
    
    print("TOTAL RESULTS")
    print(f"Q25: {q25}")
    print(f"Q50: {q50}")
    print(f"Q75: {q75}")
    print(f"mean: {mean_}")
    save_csv_file(f"{file_results}", data, flag="w+")
    