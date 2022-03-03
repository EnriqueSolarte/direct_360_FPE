import numpy as np
from skimage import measure


def dist2d(x1, x2):
    return max(abs(x1[0] - x2[0]), abs(x1[1] - x2[1]))


def norm(x, axis):
    return np.sqrt(np.sum(np.power(x, 2), axis=axis))


def calc_mask_iou(mask1, mask2):
    # mask shape: (H, W), 0: background, 1: target
    intersection = (mask1 * mask2).sum()
    union = mask1.sum() + mask2.sum() - intersection
    return float(intersection) / max(union, 1)


def calc_polygon_iou(polygon1, polygon2, sample_size):
    '''
        Calculate iou ratio between two polygons by uniform sampling over space
    '''
    if isinstance(polygon1, list):
        points1 = np.concatenate(polygon1, axis=0)
    else:
        points1 = polygon1
    if isinstance(polygon2, list):
        points2 = np.concatenate(polygon2, axis=0)
    else:
        points2 = polygon2
    points = np.concatenate([points1, points2], axis=0)

    # Uniformly sample points in space
    mins = points.min(0)
    maxs = points.max(0)
    x = np.linspace(mins[0], maxs[0], sample_size)
    y = np.linspace(mins[1], maxs[1], sample_size)
    xv, yv = np.meshgrid(x, y)
    sample_points = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)

    if isinstance(polygon1, list):
        mask1 = np.zeros_like(xv.reshape(-1), dtype=np.bool)
        for p in polygon1:
            mask1 |= measure.points_in_poly(sample_points, p)
    else:
        mask1 = measure.points_in_poly(sample_points, polygon1)
    if isinstance(polygon2, list):
        mask2 = np.zeros_like(xv.reshape(-1), dtype=np.bool)
        for p in polygon2:
            mask2 |= measure.points_in_poly(sample_points, p)
    else:
        mask2 = measure.points_in_poly(sample_points, polygon2)
    inter = mask1 & mask2
    union = mask1 | mask2
    iou = np.sum(inter) * 1.0 / np.sum(union)
    return iou


def corner_chamfer_dist(points_pred, points_gt):
    points_pred = np.expand_dims(points_pred, axis=1)       # N, 1, 2
    points_gt = np.expand_dims(points_gt, axis=0)           # 1, M, 2
    dist_matrix = norm(points_pred - points_gt, axis=-1)    # N, M
    dist_matrix = np.power(dist_matrix, 2)
    return np.mean(np.min(dist_matrix, axis=1)) + np.mean(np.min(dist_matrix, axis=0))


def calc_corners_pr(points_pred, points_gt, dist_threshold):
    '''
    Floor-SP paper:
        We declare that a corner is successfully reconstructed
        if there is a ground-truth room corner within 10 pixels.
        When multiple corners are detected around a single ground-truth corner,
        we only take the closest one as correct and treat the others
        as false-positives (-> precision).
        # TODO: What about (the other way) when multiple ground-truth corners
        # match the same prediction. Is it considered as a hit for recall?
    inputs:
        points: shape (N, 2), x-y coordinates of each corner
    '''
    dist_map_gt = {}
    for i, point_gt in enumerate(points_gt):
        dist_map_gt[i] = []
        for j, point_pred in enumerate(points_pred):
            d = dist2d(point_pred[0:2], point_gt[0:2])
            if d <= dist_threshold:
                dist_map_gt[i].append((d, j))
        if len(dist_map_gt[i]) > 0:
            dist_map_gt[i].sort(key=lambda x: x[0])

    total_hits = 0.0
    visit_pred = set()
    for i, point_gt in enumerate(points_gt):
        if len(dist_map_gt[i]) > 0:
            # The closest prediction point
            (d, pred_idx) = dist_map_gt[i][0]
            if pred_idx not in visit_pred:
                total_hits += 1
                visit_pred.add(pred_idx)
    return total_hits, len(points_gt), len(points_pred)


def evaluate_edges(points_pred, points_gt, walls_pred, walls_gt, dist_threshold):
    '''
    Floor-SP paper:
        We declare that an edge of a graph is successfully reconstructed
        if its two end-points pass the corner test described above (corner test)
        and the corresponding edge belongs to the ground-truth.
    inputs:
        points: shape (N, 2), x-y coordinates of each corner
        walls: shape (M, 2), corner index of the two end-points of each wall
    '''

    total_hits = 0.0
    for i, wall_gt in enumerate(walls_gt):
        x1 = points_gt[wall_gt[0]]
        x2 = points_gt[wall_gt[1]]

        hit = False
        for j, wall_pred in enumerate(walls_pred):
            y1 = points_pred[wall_pred[0]]
            y2 = points_pred[wall_pred[1]]

            # x1 matches y1, x2 matches y2
            if (dist2d(x1[0:2], y1[0:2]) <= dist_threshold
                and dist2d(x2[0:2], y2[0:2]) <= dist_threshold):
                hit = True
                break
            # Or the other way matches
            if (dist2d(x1[0:2], y2[0:2]) <= dist_threshold
                and dist2d(x2[0:2], y1[0:2]) <= dist_threshold):
                hit = True
                break

        if hit:
            total_hits += 1

    total_gt = len(walls_gt)
    total_pred = len(walls_pred)
    return total_hits, total_gt, total_pred


def calc_room_pr_from_mask(room_masks_pred, room_masks_gt, iou_threshold):
    '''
    In this function, we don't consider room type for evaluation.
    Floor-SP paper:
        1) it does not overlap with any other room, and
        2) there exists a room in the ground-truth with intersection-over-union (IOU) score more than 0.7
    input:
        room_masks: list of mask with shape (H, W)
    '''
    total_hits = 0.0
    pred_to_gt = {}
    for i, mask_pred in enumerate(room_masks_pred):
        for j, mask_gt in enumerate(room_masks_gt):
            # NOTE: Original code consider room type with hasCommonLabel condition
            # However, this implies each room could have multiple types given by its mutiple walls.
            # This could be a bug, UNLESS for a room, each of its wall will always output the same label.
            # > hasCommonLabel = False
            # > for labelGT in labelPred:
            # >     if labelGT in labelPred:
            # >     hasCommonLabel = True
            # >     break

            iou = calc_mask_iou(mask_gt, mask_pred)
            # TODO: Make sure each pred only match one GT.
            # If iou_threshold > 0.5, maybe no problem about this.
            # (since no intersection between output rooms.)

            # TODO: Check predition overlap to other prediction.
            # (This won't happend for FloorNet but maybe other methods will.)
            if iou >= iou_threshold:
                pred_to_gt[i] = j
                total_hits += 1
    # Check duplicate matching (multiple preds match the same GT)
    x = [v for k, v in pred_to_gt.items()]
    assert len(x) == len(set(x))
    return total_hits, len(room_masks_gt), len(room_masks_pred)


def evaluate_room_plus_plus(
    room_masks_pred,
    room_adj_pred,
    room_masks_gt,
    room_adj_gt,
    iou_threshold
):
    '''
    Room++ metric
        We declare that a room is successfully reconstructed in this metric,
        if the room is connected (i.e., sharing edges) to the correct set of successfully
        reconstructed rooms as in the ground-truth, besides passing the above two room conditions.
    input:
        room_masks: list of mask with shape (H, W)
        room_adj_dict: dictionary, room_idx -> set of adjacent room indices
    '''
    pred_to_gt = {}
    for i, mask_pred in enumerate(room_masks_pred):
        for j, mask_gt in enumerate(room_masks_gt):
            iou = calc_mask_iou(mask_gt, mask_pred)
            if iou >= iou_threshold:
                pred_to_gt[i] = j

    total_hits = 0.0
    for i, room_pred in enumerate(room_masks_pred):
        if i not in pred_to_gt:
            # Target room not matches in ground truth
            continue

        # Find the mapping for the adjacent rooms of "i"
        target_room_idx = pred_to_gt[i]
        target_adj_room = set()
        all_neighbors_found = True

        for room_idx in room_adj_pred[i]:
            room_idx_gt = pred_to_gt.get(room_idx, None)
            if room_idx_gt is None:
                all_neighbors_found = False
                break
            target_adj_room.add(room_idx_gt)

        if not all_neighbors_found:
            continue

        adj_room_gt = set(room_adj_gt[target_room_idx])

        if target_adj_room == adj_room_gt:
            total_hits += 1

    return total_hits, len(room_masks_gt), len(room_masks_pred)