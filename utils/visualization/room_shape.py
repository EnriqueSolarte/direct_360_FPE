import cv2
import matplotlib.pyplot as plt
import numpy as np
# from src.solvers.room_shape_estimator import RoomShapeEstimator
from utils.ocg_utils import get_line_pixels


def plot_spa_info(
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

    plt.figure("spa info")
    plt.clf()
    plt.imshow(image)
    plt.draw()
    plt.waitforbuttonpress(0.1)


def vis_room_result(spa, corners_uv, start_uv=None, end_uv=None, draw_plane=False):
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
    image_spa = vis_room_result(spa, corners_uv, start_uv, end_uv, draw_plane=False)
    image_plane = vis_room_result(spa, corners_uv, start_uv, end_uv, draw_plane=True)
    plt.figure("Room shape result")
    plt.clf()
    plt.subplot(121)
    plt.imshow(image_spa)
    plt.subplot(122)
    plt.imshow(image_plane)
    plt.draw()
    plt.waitforbuttonpress(0.1)


def visualize_spa_info(spa, draw_plane=False, corners=None, start_uv=None, end_uv=None, break_line=None):
    '''
        Draw information in spa (input for spa)
        If plane is False, draw valid_pixels as background.

    '''
    # RSO = RoomShapeEstimator(room)
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

    # Draw corners
    if corners is None:
        # Try get corners from spa object
        corners = spa.get_corners()
    if corners is not None:
        uv_corners = ocg_xyz_to_uv(spa.ocg, corners)
        # uv_corners = np.round(uv_corners).astype(np.int32)
        for i in range(uv_corners.shape[1]):
            cv2.circle(image, (uv_corners[0, i], uv_corners[1, i]), 1, (255, 0, 0), -1)

    # Find center
    center = spa.get_center()
    uv_center = ocg_xyz_to_uv(spa.ocg, center)

    # Draw theta orientation
    theta_vecs = spa.get_theta_vecs()
    length = 5
    for i in range(theta_vecs.shape[0]):
        k = uv_center + theta_vecs[i, :] * length
        k = np.round(k).astype(np.int32)
        # c = round(127 +  127 * i / theta_vecs.shape[0])
        c = 255
        cv2.line(image, (uv_center[0], uv_center[1]), (k[0], k[1]), (c, c, c), 1)

    # uv_center = np.round(uv_center).astype(np.int32)
    cv2.circle(image, (uv_center[0], uv_center[1]), 1, (255, 255, 0), -1)

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


def visualize_edge(spa, graph, node_to_pixel, start_node_idx=None, end_node_idx=None):
    if end_node_idx is not None:
        node_idx = end_node_idx
        uv = node_to_pixel[node_idx]
        edges = {}
        for k, v in graph.items():
            if end_node_idx in v:
                edges[k] = v[end_node_idx]
    else:
        if start_node_idx is None:
            # Pick random start node idx
            node_idx = np.random.choice(node_to_pixel.shape[0])

        else:
            node_idx = start_node_idx
        uv = node_to_pixel[node_idx]
        edges = graph[node_idx]

    images = []
    H, W = spa.ocg.get_shape()
    for j in range(6):
        image = np.zeros((H, W), dtype=np.float32)
        # image.fill(255)

        all_weights = []
        for (next_node_idx, weights) in edges.items():
            all_weights.append(weights[j])
        # if len(all_weights) > 0:
        #     max_weight = max(all_weights)
        #     min_weight = min(all_weights)
        for (next_node_idx, weights) in edges.items():
            next_uv = node_to_pixel[next_node_idx]
            image[next_uv[1], next_uv[0]] = weights[j]

        # cv2.circle(image, (uv[1], uv[0]), 2, (255, 0, 0), thickness=-1)
        images.append(image)
    return images


def visualize_room_result(spa, uv_corners, draw_plane=False):
    H, W = spa.ocg.get_shape()
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
    size = uv_corners.shape[0]
    for i in range(size):
        x_u, x_v = uv_corners[i, :]
        y_u, y_v = uv_corners[(i+1) % size, :]

        cv2.line(image, (x_u, x_v), (y_u, y_v), (255, 255, 255), 1)
        cv2.circle(image, (x_u, x_v), 1, (255, 0, 0), -1)
        cv2.circle(image, (y_u, y_v), 1, (255, 0, 0), -1)

    return image


def visualize_debug_spa(cfg, room):
    spa = SPAPatch(cfg, room)
    input_image = visualize_spa_info(spa)
    plane_image = visualize_spa_info(spa, draw_plane=True)
    return input_image, plane_image
