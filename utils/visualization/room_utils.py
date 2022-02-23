import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
import copy
from src.solvers.theta_estimator import GaussianModel_1D


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

    global_map = np.ones((fpe.global_ocg_patch.H, fpe.global_ocg_patch.W, 3))
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


def plot_gaussian_estimations(list_theta_z, block=True):
    x = np.linspace(-1.1 * np.pi, 1.1 * np.pi, 1000)
    plt.figure("Estimated Gaussian Orientations")
    for z in list_theta_z:
        plt.plot(np.degrees(x),
                 GaussianModel_1D.visual_model(x, mean=z.mean, sigma=z.sigma),
                 label=r'$\mu$:{0:.2f} - $\sigma$:{1:.02f}'.format(np.degrees(z.mean), np.degrees(z.sigma)))
    plt.title("Estimated Gaussian Orientations")
    if block:
        plt.show()
    else:
        plt.draw()
        plt.waitforbuttonpress(0.01)


def plot_estimated_orientations(list_theta_z, block=False, caption="Estimated Orientations"):
    fig = plt.figure(caption)
    plt.clf()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(-np.pi/10)
    for z in list_theta_z:
        theta = z.mean
        sigma2 = z.sigma/2
        if z.sigma > np.radians(60):
            continue
        # ax.scatter(np.ones((100,)) * z.mean, np.linspace(0, 1, 100))
        ax.plot([theta, theta], [0, 1], lw=1, marker="*", color="black")
        plt.fill_between(
            np.linspace(theta - sigma2, theta + sigma2, 100),
            0,
            0.95,
            label="sigma:{0:2.2f}".format(np.degrees(z.sigma))
        )

    ax.set_title(caption)
    plt.legend()
    if block:
        plt.show()
    else:
        plt.draw()
        plt.waitforbuttonpress(0.001)
