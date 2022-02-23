import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
import copy

def plot_curr_room_by_patches(fpe):
    """
    Plots de current ocg-map relate to the current room
    """
    room_ocg_map = fpe.curr_room.local_ocg_patches.ocg_map
    plt.figure("plot_curr_room_by_patches")
    plt.clf()
    plt.subplot(121)
    plt.imshow(room_ocg_map)
    plt.subplot(122)
    plt.plot(fpe.curr_room.p_pose)
    plt.draw()
    plt.waitforbuttonpress(0.1)


def plot_all_rooms_by_patches(fpe):
    """
    Plots all rooms by ocg-maps
    """
    fpe = copy.deepcopy(fpe)
    fpe.global_ocg_patch.update_bins()
    fpe.global_ocg_patch.update_ocg_map()
    
    global_map = np.ones((fpe.global_ocg_patch.H,fpe.global_ocg_patch.W, 3))
    global_map[:, :, 1] = 0
    
    colors = np.linspace(0, 0.5, fpe.list_rooms.__len__())
    for idx, ocg_map in enumerate(fpe.global_ocg_patch.ocg_map):
        ocg_map = ocg_map/np.max(ocg_map)
        mask = ocg_map > fpe.dt.cfg.get("room_id.ocg_threshold", 0.5)

        global_map[mask, 0] = colors[idx]
        global_map[mask, 1] = 1
        global_map[mask, 2] = ocg_map[mask]

    global_map = hsv2rgb(global_map)   
    plt.figure("plot_all_rooms_by_patches")
    plt.clf()
    plt.imshow(global_map)
    plt.draw()
    plt.waitforbuttonpress(0.1)
