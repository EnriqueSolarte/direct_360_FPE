import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils.ocg_utils import get_line_pixels


def visualize_spa_info(
    spa,
    draw_plane=False,
    corners=None,
    start_uv=None,
    end_uv=None,
    break_line=None
):

    H, W = spa.ocg.get_shape()
    image = np.zeros((H, W, 3), dtype=np.uint8)
    if draw_plane:
        plane_map = spa.get_plane_density()
        plane_map[plane_map > 0] = 1
        vs, us = plane_map.nonzero()
        image[vs, us, :] = np.array([127, 127, 127])
    else:
        all_nodes = spa.get_valid_pixels()
        image[all_nodes[:, 1], all_nodes[:, 0], :] = np.array([127, 127, 127])

    area = spa.get_interior_area()
    image[area] = np.array([0, 127, 0])

    if start_uv is not None:
        # Pink
        cv2.circle(image, (start_uv[0], start_uv[1]), 1, (254, 140, 173), -1)
    if end_uv is not None:
        # Green
        cv2.circle(image, (end_uv[0], end_uv[1]), 1, (0, 255, 0), -1)
    if break_line is not None:
        line_u, line_v = get_line_pixels(break_line[0], break_line[1])
        image[line_v, line_u, :] = np.array([255, 0, 0])

    return image


def visualize_room_result(spa, corners_uv, start_uv=None, end_uv=None, draw_plane=False):
    ocg = spa.ocg
    H, W = ocg.get_shape()
    image = np.zeros((H, W, 3), dtype=np.uint8)

    # Draw valid pixels
    if draw_plane:
        plane_map = spa.get_plane_density()
        plane_map[plane_map > 0] = 1
        vs, us = plane_map.nonzero()
        image[vs, us, :] = np.array([127, 127, 127])
    else:
        all_nodes = spa.get_valid_pixels()
        image[all_nodes[:, 1], all_nodes[:, 0], :] = np.array([127, 127, 127])

    # Draw interior area
    area = spa.get_interior_area()
    image[area] = np.array([0, 127, 0])

    # Draw result room shape
    size = corners_uv.shape[0]
    for i in range(size):
        x_u, x_v = corners_uv[i, :]
        y_u, y_v = corners_uv[(i+1) % size, :]

        cv2.line(image, (x_u, x_v), (y_u, y_v), (255, 255, 255), 1)
        cv2.circle(image, (x_u, x_v), 1, (255, 0, 0), -1)
        cv2.circle(image, (y_u, y_v), 1, (255, 0, 0), -1)
    if start_uv is not None:
        # Pink
        cv2.circle(image, (start_uv[0], start_uv[1]), 1, (254, 140, 173), -1)
    if end_uv is not None:
        # Green
        cv2.circle(image, (end_uv[0], end_uv[1]), 1, (0, 255, 0), -1)
    return image


def plot_room_result(spa, corners_uv, start_uv=None, end_uv=None):
    image_spa = visualize_room_result(spa, corners_uv, start_uv, end_uv, draw_plane=False)
    image_plane = visualize_room_result(spa, corners_uv, start_uv, end_uv, draw_plane=True)
    plt.figure("Room shape result")
    plt.clf()
    plt.subplot(121)
    plt.imshow(image_spa)
    plt.subplot(122)
    plt.imshow(image_plane)
    plt.draw()
    plt.waitforbuttonpress(0.1)

