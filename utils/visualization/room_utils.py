import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv, hsv2rgb
from src.solvers.theta_estimator import GaussianModel_1D
from utils.eval_utils import rotate_by_axis_corners
import os


def plot_curr_room_by_patches(fpe):
    """
    Plots de current ocg-map relate to the current room
    """
    room_ocg_map = fpe.curr_room.local_ocg_patches.ocg_map.copy()
    pose_uv = fpe.curr_room.local_ocg_patches.project_xyz_to_uv(
        fpe.curr_room.list_ly[-1].pose_est.t.reshape(3, 1)
    )

    room_ocg_map[pose_uv[1, :], pose_uv[0, :]] = -1
    plt.figure("plot_curr_room_by_patches")
    plt.clf()
    plt.subplot(121)
    plt.imshow(room_ocg_map)
    plt.subplot(122)
    plt.plot(fpe.curr_room.p_pose)
    plt.draw()
    plt.waitforbuttonpress(0.1)


def plot_all_rooms_by_patches(fpe, only_save=False):
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
    # if only_save:
    #     plt.savefig(os.path.join(fpe.dt.cfg.get("results_dir"), f"{fpe.dt.scene_name}.jpg"))
    #     return
    return global_map


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


def get_colors(num_colors):
    ''' return colors with shape (num_colors, 3) '''
    colors = np.linspace(0, 0.5, num_colors)
    ones = np.ones_like(colors)
    colors = hsv2rgb(np.stack([colors, ones, ones], axis=1))
    colors = np.clip(colors * 255, 0, 255).astype(np.int32)
    return colors


def plot_all_planes(fpe, axis_align=True):
    # Draw planes and gt point cloud together
    planes = []
    for room in fpe.list_rooms:
        planes.extend([pl.boundary for pl in room.list_pl])
    planes = np.concatenate(planes, axis=1)     # (3, N)
    planes = planes[[0, 2], :].T
    points = fpe.dt.pcl_gt[:, [0, 2]]

    if axis_align:
        axis_corners = fpe.dt.axis_corners
        planes = rotate_by_axis_corners(planes, axis_corners)
        points = rotate_by_axis_corners(points, axis_corners)

    plt.figure("Planes & points")
    plt.clf()
    plt.scatter(planes[:, 0], planes[:, 1], c='blue')
    plt.scatter(points[:, 0], points[:, 1], c='red')
    plt.show()


def plot_floor_plan(room_list, ocg, grid_size=512, points_gt=None, planes=None):
    '''
        room_list: list of room_corners with shape (2, N)
        ocg: OCGPatches objects for mapping 3D into 2D space
    '''

    height, width = ocg.get_shape()
    if grid_size is not None:
        ocg = copy.deepcopy(ocg)
        ocg.resize(grid_size / max(height, width))
        height, width = ocg.get_shape()
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image.fill(255)
    if points_gt is not None:
        density_map = ocg.project_xyz_points_to_hist(points_gt[:, :3].T)
        density_map = np.stack([density_map, density_map, density_map], axis=-1)
        density_map /= density_map.max()
        density_map *= 255
        density_map = np.clip(np.round(density_map), 0, 255).astype(np.uint8)
        density_map = 255 - density_map
        image += density_map
    # if planes is not None:
    #     density_map = ocg.project_xyz_points_to_hist(planes.T)

    colors = get_colors(len(room_list))
    for room_idx, room_corners in enumerate(room_list):
        # TODO: Deal with the scale difference between GT points and corners
        # room_corners *= scale
        N = room_corners.shape[1]
        if room_corners.shape[0] == 2:
            # Make xz (2, N) -> xyz (3, N)
            ones = np.ones_like(room_corners[0, :])
            room_corners = np.stack([room_corners[0, :], ones, room_corners[1, :]], axis=0)
        room_corners = ocg.project_xyz_to_uv(room_corners)
        # color = tuple((colors[room_idx][0], colors[room_idx][1], colors[room_idx][2]))
        color = colors[room_idx].tolist()
        for i in range(N):
            u1, v1 = room_corners[:, i]
            u2, v2 = room_corners[:, (i+1) % N]
            cv2.line(image, (u1, v1), (u2, v2), color, 3)
            cv2.circle(image, (u1, v1), 5, color, -1)
            cv2.circle(image, (u2, v2), 5, color, -1)
    return image


def plot_planes_rooms_patches(fpe, points_gt=None, room_corner_list=None, draw_plane=True):
    '''
        fpe:            FPE object
        pcl_gt:         GT point cloud
        room_list:      Estimated room corners. Draw rooms if not None
        room_list_gt:      GT room corners. Draw rooms if not None
        scale:          if None, will use fpe.gt_scale
        draw_plane:     boolean
    '''
    # Prepare background
    ocg = fpe.global_ocg_patch
    height, width = ocg.get_shape()
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image.fill(255)
    if points_gt is not None:
        density_map = ocg.project_xyz_points_to_hist(points_gt[:, :3].T)
        density_map = np.stack([density_map, density_map, density_map], axis=-1)
        density_map /= density_map.max()
        density_map *= 255
        density_map = np.clip(np.round(density_map), 0, 255).astype(np.uint8)
        density_map = 255 - density_map
        image += density_map

    if draw_plane:
        planes = []
        for room in fpe.list_rooms:
            planes.extend([pl.boundary for pl in room.list_pl])
        planes = np.concatenate(planes, axis=1)     # (3, N)
        planes = ocg.project_xyz_to_uv(planes)
        mask = (planes[0, :] >= 0) & (planes[0, :] < width) & (planes[1, :] >= 0) & (planes[1, :] < height)
        image[planes[1, mask], planes[0, mask], :] = np.array([128, 128, 128])

    colors = get_colors(len(fpe.global_ocg_patch.ocg_map))
    for idx, ocg_map in enumerate(fpe.global_ocg_patch.ocg_map):
        mask = ocg_map > fpe.dt.cfg["room_shape_opt.ocg_threshold"]
        image[mask, :] = colors[idx, :]

    if room_corner_list is not None:
        for room_idx, room_corners in enumerate(room_corner_list):
            N = room_corners.shape[1]
            if room_corners.shape[0] == 2:
                # Make xz (2, N) -> xyz (3, N)
                ones = np.ones_like(room_corners[0, :])
                room_corners = np.stack([room_corners[0, :], ones, room_corners[1, :]], axis=0)
            room_corners = ocg.project_xyz_to_uv(room_corners)
            # color = colors[room_idx].tolist()
            color = (255, 0, 255)
            for i in range(N):
                u1, v1 = room_corners[:, i]
                u2, v2 = room_corners[:, (i+1) % N]
                cv2.line(image, (u1, v1), (u2, v2), color, 2)
                cv2.circle(image, (u1, v1), 1, color, -1)
                cv2.circle(image, (u2, v2), 1, color, -1)

    return image
