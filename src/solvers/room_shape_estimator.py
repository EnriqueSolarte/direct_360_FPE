import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import square, dilation, erosion

# from solvers.SPA import SPA
from utils.ocg_utils import get_line_pixels
from utils.graph_utils import dijkstra
from utils.visualization.room_shape import visualize_spa_info
from utils.visualization.room_shape import visualize_room_result


def ocg_xyz_to_uv(ocg, xyz: np.ndarray):
    '''
        Convert xyz with shape (3,) or (N, 3) to uv with shape (2,) or (N, 2)
    '''
    if len(xyz.shape) == 1:
        if xyz.shape[0] == 2:
            # (x, z) => (x, x, z)
            xyz = np.array([xyz[0], 0, xyz[1]], dtype=xyz.dtype)
        xyz = xyz.reshape(3, 1)
        uv = ocg.project_xyz_to_uv(xyz)
        uv = uv.reshape(-1)
        uv = np.round(uv).astype(np.int32)
        return uv
    elif len(xyz.shape) == 2:
        uv = ocg.project_xyz_to_uv(xyz)
        uv = np.round(uv).astype(np.int32)
        return uv
    else:
        raise NotImplementedError


def ocg_uv_to_xz(ocg, uv: np.ndarray):
    '''
        Convert uv with shape (2,) or (N, 2) to xz with shape (2,) or (N, 2)
    '''
    if len(uv.shape) == 1:
        uv = uv.reshape(2, 1)
        xyz = ocg.project_uv_to_xyz(uv)
        xyz = xyz.reshape(-1)
        return xyz[[0, 2]]
    elif len(uv.shape) == 2:
        xyz = ocg.project_uv_to_xyz(uv)
        return xyz[[0, 2], :]
    else:
        raise NotImplementedError


def fit_size_range(height, width, min_edge, max_edge):
    scale1 = 1.0
    scale2 = 1.0
    if min_edge is not None and min(height, width) < min_edge:
        if height <= width:
            scale1 = min_edge / height
        else:
            scale1 = min_edge / width

    height = round(height * scale1)
    width = round(width * scale1)
    if max_edge is not None and max(height, width) > max_edge:
        if height <= width:
            scale2 = max_edge / width
        else:
            scale2 = max_edge / height
    height = round(height * scale2)
    width = round(width * scale2)
    return height, width, scale1*scale2


class SPAError(Exception):
    pass


class RoomShapeEstimator:
    def __init__(self, dt):
        self.dt = dt
        self.cfg = dt.cfg

    def estimate(self, room, room_idx=0, dump_dir=None):
        '''
            Estimate initial room shape with spa_basic and run multiple spa_refine for better room shape
        '''
        # Deep copy so that the estimator can adjust the size without effecting room_id
        ocg_patch = copy.deepcopy(room.local_ocg_patches)
        height, width = ocg_patch.get_shape()
        min_edge = self.cfg['room_shape_opt.spa_min_edge']
        max_edge = self.cfg['room_shape_opt.spa_max_edge']
        new_height, new_width, scale = fit_size_range(height, width, min_edge, max_edge)
        # ocg_patch.ocg_map = (ocg_patch.get_mask()).astype(np.int32)
        # ocg_patch.ocg_map = erosion(ocg_patch.ocg_map, square(3))
        ocg_patch.resize(scale)     # TODO: Check why apply erosion before resize

        # Fit room and estimate room shape
        spa_first = SPABasic(self.cfg, room, ocg_patch)
        out_dict = spa_first.estimate_shape(os.path.join(dump_dir, f'{room_idx}_0.png'))
        # plot_room_result(
        #     self.spa_first,
        #     out_dict['corners_uv'],
        #     start_uv=out_dict['start_uv'],
        #     end_uv=out_dict['end_uv'],
        #     # draw_plane=False
        # )

        all_result = [out_dict]

        # Start refine SPA
        if self.cfg['room_shape_opt.spa_refine_times'] > 0:
            for iter_idx in range(self.cfg['room_shape_opt.spa_refine_times']):
                if all_result[-1]['corners_xz'].shape[0] <= 3:
                    break
                refine_size = self.cfg['room_shape_opt.refine_sizes'][iter_idx]
                ocg_patch = copy.deepcopy(room.local_ocg_patches)       # Copy again from original size
                height, width = ocg_patch.get_shape()
                new_height, new_width, scale = fit_size_range(height, width, refine_size, refine_size)
                # ocg_patch.ocg_map = (ocg_patch.get_mask()).astype(np.int32)
                # ocg_patch.ocg_map = erosion(ocg_patch.ocg_map, square(3))
                ocg_patch.resize(scale)
                spa_refine = SPARefine(
                    self.cfg, room, ocg_patch,
                    all_result[-1]['corners_xz'].T,
                    prev_start_corner=all_result[-1]['start_uv'],
                    prev_end_corner=all_result[-1]['end_uv'],
                )
                out_dict = spa_refine.estimate_shape(os.path.join(dump_dir, f'{room_idx}_{iter_idx+1}.png'))
                all_result.append(out_dict)
                # plot_room_result(
                #     self.spa_refine,
                #     out_dict['corners_uv'],
                #     start_uv=out_dict['start_uv'],
                #     end_uv=out_dict['end_uv'],
                # )
        return all_result[-1]


class SPABasic:
    def __init__(self, cfg, room, ocg_patch):
        '''
            room: LocalRoom object
        '''
        self.cfg = cfg
        self.room = room
        self.ocg = ocg_patch

        if len(self.room.list_pl) == 0:
            raise SPAError('No plane info found')

    def get_plane_pcl(self, plane_type='all'):
        assert plane_type in ['optimal', 'all']
        if plane_type == 'optimal':
            planes = [pl.boundary for pl in self.room.list_pl if pl.isOptimal]
        elif plane_type == 'all':
            planes = [pl.boundary for pl in self.room.list_pl]

        return np.concatenate(planes, axis=1)

    def get_plane_density(self, plane_type='all'):
        planes = self.get_plane_pcl(plane_type)
        plane_density_map = self.ocg.project_xyz_points_to_hist(planes)
        plane_density_map[plane_density_map > 0] = 1
        return plane_density_map

    def get_corners(self):
        corner_list = [c.position for c in self.room.list_corners]
        if len(corner_list) == 0:
            return None
        corners = np.stack(corner_list, axis=1)
        return corners

    def get_corner_density(self):
        corners = self.get_corners()
        if corners is None:
            # Return density map with all zeros
            H, W = self.ocg.get_shape()
            corner_density_map = np.zeros((H, W), dtype=np.int32)
            return corner_density_map
        corner_density_map = self.ocg.project_xyz_points_to_hist(corners)
        corner_density_map[corner_density_map > 0] = 1
        corner_density_map = dilation(corner_density_map, square(3))
        return corner_density_map

    def get_patch(self):
        # TODO: Check how to adapt original mask
        # return room.local_ocg_patches.get_mask()
        ocg_map = self.ocg.ocg_map
        ocg_map = ocg_map / np.max(ocg_map)
        mask = ocg_map > self.cfg["room_shape_opt.ocg_threshold"]
        return mask

    def get_interior_area(self):
        # Use patch to create binary mask for interior area
        ocg_map = self.get_patch()
        ocg_map = erosion(ocg_map, square(3))
        # ocg_map = erosion(ocg_map, square(5))
        # all_nodes = self.get_valid_pixels()
        # ocg_map[all_nodes[:, 1], all_nodes[:, 0]] = 0
        return ocg_map

    def get_center(self):
        planes = self.get_plane_pcl()
        center = np.mean(planes, axis=1)
        return center

    def get_major_orientation(self):
        return np.array([a.mean for a in self.room.theta_z])

    def get_theta_vecs(self):
        theta_angles = self.get_major_orientation()
        theta_cos = np.cos(theta_angles)
        theta_sin = np.sin(theta_angles)
        theta_vecs = np.stack([theta_sin, theta_cos], axis=1)      # (N, 2)
        return theta_vecs

    def get_camera_poses(self):
        poses = [ly.pose_est.t for ly in self.room.list_ly]
        poses = np.stack(poses, axis=1)[[0, 2], :]
        return poses

    def get_valid_pixels(self):
        '''
            Return valid pixels for graph nodes with shape (n, 2)
        '''
        if self.cfg['room_shape_opt.valid_type'] == 'edge':
            planes = self.get_plane_pcl()
            ocg_map = self.ocg.project_xyz_points_to_hist(planes)
            ocg_map[ocg_map > 0] = 1

            ocg_map = dilation(ocg_map, square(3))
            ocg_map = erosion(ocg_map, square(3))
            inter_mask = self.get_interior_area()
            # patch = erosion(patch, square(5))
            ocg_map[inter_mask] = 0

        elif self.cfg['room_shape_opt.valid_type'] == 'patch_contour':
            # TODO: Need to fix
            ocg_map = self.get_patch()
            contours = measure.find_contours(ocg_map)
            image = np.zeros_like(ocg_map)
            for contour in contours:
                contour = contour.astype(np.int32)
                image[contour[:, 0], contour[:, 1]] = 1
            ocg_map = image
            ocg_map = dilation(ocg_map, square(3))
            # ocg_map = dilation(ocg_map, square(3))
            # ocg_map[inter_area] = 0

        elif self.cfg['room_shape_opt.valid_type'] == 'patch_boundary':
            inter_mask = self.get_interior_area()
            ocg_map = inter_mask.copy()

            for _ in range(4):
                ocg_map = dilation(ocg_map, square(3))
            # patch = erosion(patch, square(5))
            ocg_map[inter_mask] = 0

        elif self.cfg['room_shape_opt.valid_type'] == 'corner':
            raise NotImplementedError
        else:
            raise NotImplementedError
        v_idxs, u_idxs = ocg_map.nonzero()
        all_nodes = np.stack([u_idxs, v_idxs], axis=1)
        return all_nodes

    def get_start_edge(self, all_nodes, corners, theta_vecs, center, inter_area):
        '''
        The starting edge is provided as follows
            1. Take one valid plane
            2. Take any two corners
        However, the starting edge need to satisfy the valid edge condition:
            a. major orientation
            b. Not crossing the inter_area
            c. Cross product assumption
        '''
        # Return the index of start_node and end_node
        valid_edges = []
        # corners = self.get_corners().T

        def find_nearest_corner(x, corners, threshold=1000):
            # NOTE: We are not using corners anymore
            return x
            v = corners - np.expand_dims(x, axis=1)    # 2, N
            dist = np.linalg.norm(v, axis=0)
            idx = np.argmin(dist)
            if dist[idx] > threshold:
                return x
            return corners[:, idx]

        for plane in self.room.list_pl:
            # import pdb; pdb.set_trace()
            start_node = plane.boundary[[0, 2], 0]
            end_node = plane.boundary[[0, 2], -1]

            if corners is not None:
                start_node = find_nearest_corner(start_node, corners)
                end_node = find_nearest_corner(end_node, corners)

            if np.linalg.norm(start_node - end_node) < 1e-6:
                # If start_node and end_node are the same
                continue

            # Check if the line in uv space will cross inter_area
            start_uv = ocg_xyz_to_uv(self.ocg, start_node)
            end_uv = ocg_xyz_to_uv(self.ocg, end_node)
            line_u, line_v = get_line_pixels(start_uv, end_uv)
            if inter_area[line_v, line_u].sum() > 0:
                continue
            if self.is_edge(end_node, start_node, theta_vecs, center):
                valid_edges.append([start_node, end_node])
            if self.is_edge(start_node, end_node, theta_vecs, center):
                valid_edges.append([end_node, start_node])

        # Test all pairs of corners
        # if corners is not None:
        #     for i in range(corners.shape[1]):
        #         for j in range(corners.shape[1]):
        #             if i == j:
        #                 continue
        #             start_node = corners[:, i]
        #             end_node = corners[:, j]

        #             # Check if the line in uv space will cross inter_area
        #             start_uv = ocg_xyz_to_uv(self.ocg, start_node)
        #             end_uv = ocg_xyz_to_uv(self.ocg, end_node)
        #             line_u, line_v = get_line_pixels(start_uv, end_uv)
        #             if inter_area[line_v, line_u].sum() > 0:
        #                 continue
        #             if self.is_edge(end_node, start_node, theta_vecs, center):
        #                 valid_edges.append([start_node, end_node])
        #             if self.is_edge(start_node, end_node, theta_vecs, center):
        #                 valid_edges.append([end_node, start_node])

        if len(valid_edges) == 0:
            raise SPAError("Cannot find a valid starting edge")

        # start_node, end_node = valid_edges[0]
        # Find breaking line
        valid_edges_filtered = []
        for start_node, end_node in valid_edges:
            start_uv = ocg_xyz_to_uv(self.ocg, start_node)
            end_uv = ocg_xyz_to_uv(self.ocg, end_node)

            # In case the start_uv is not in valid pixels, find the closest valid start node (same for end node)
            start_node_idx = np.argmin(np.sum(np.abs(all_nodes - np.expand_dims(start_uv, axis=0)), axis=1))
            end_node_idx = np.argmin(np.sum(np.abs(all_nodes - np.expand_dims(end_uv, axis=0)), axis=1))

            start_uv = all_nodes[start_node_idx, :]
            end_uv = all_nodes[end_node_idx, :]
            edge_length = np.sum(np.abs(start_uv - end_uv))

            found_valid, break_line = self.find_break_line(start_uv, end_uv, inter_area)
            if found_valid:
                valid_edges_filtered.append(
                    (start_node_idx, end_node_idx, break_line, edge_length)
                )

        if len(valid_edges_filtered) == 0:
            raise SPAError("Cannot find a starting edge with a valid break line")
        valid_edges_filtered.sort(key=lambda x: x[-1], reverse=True)     # Sort by edge length
        (start_node_idx, end_node_idx, break_line, edge_length) = valid_edges_filtered[0]
        break_line = (np.array(break_line[0]), np.array(break_line[1]))

        return start_node_idx, end_node_idx, break_line

    def find_break_line(self, start_uv, end_uv, inter_area, oppose_search_length=12):
        # OPPOSE_SEARCH_LENGTH = 6
        found_valid = False
        break_line = None
        # Find four direction: up, down, left, right
        mid_uv = np.round((start_uv + end_uv) / 2).astype(np.int32)
        H, W = inter_area.shape
        u, v = mid_uv

        # Test four directions to the boundary and its opposite side need to touch inter_area
        # 1. (+1, 0): from mid_uv[0] to (W - mid_uv[0])
        # 2. (-1, 0): from 0 to mid_uv[0]
        # 3. (0, +1): from mid_uv[1] to (H - mid_uv[1])
        # 4. (0, -1): from 0 to mid_uv[1]

        theta = np.arctan2(start_uv[0] - end_uv[0], start_uv[1] - end_uv[1]) * 180 / np.pi
        if theta < 0:
            theta += 180

        if np.abs(theta - 90) > 35:
            # TODO: Test 45 degree
            # The plane is not fully horizontal
            us = np.arange(0, u+1)
            for i in range(1, oppose_search_length+1):
                us_opp = np.arange(u+1, u+i+1)
                us_opp = np.clip(us_opp, 0, W-1)

                if not np.any(inter_area[v, us]) and np.sum(inter_area[v, us_opp]) > 1:
                    break_line = ((us[0], v), (us_opp[-1], v))
                    found_valid = True
                    break

            us = np.arange(u, W)
            for i in range(1, oppose_search_length+1):
                us_opp = np.arange(u-i, u)
                us_opp = np.clip(us_opp, 0, W-1)
                if not np.any(inter_area[v, us]) and np.sum(inter_area[v, us_opp]) > 1:
                    break_line = ((us_opp[0], v), (us[-1], v))
                    found_valid = True
                    break

        if np.abs(theta - 0) > 35 and np.abs(theta - 180) > 35:
            # TODO: Test 45 degree
            # The plane is not fully vertical
            vs = np.arange(0, v+1)
            for i in range(1, oppose_search_length+1):
                vs_opp = np.arange(v+1, v+i+1)
                vs_opp = np.clip(vs_opp, 0, H-1)
                if not np.any(inter_area[vs, u]) and np.sum(inter_area[vs_opp, u]) > 1:
                    break_line = ((u, vs[0]), (u, vs_opp[-1]))
                    found_valid = True
                    break

            vs = np.arange(v, H)
            for i in range(1, oppose_search_length+1):
                vs_opp = np.arange(v-i, v)
                vs_opp = np.clip(vs_opp, 0, H-1)
                if not np.any(inter_area[vs, u]) and np.sum(inter_area[vs_opp, u]) > 1:
                    break_line = ((u, vs_opp[0]), (u, vs[-1]))
                    found_valid = True
                    break
        return found_valid, break_line

    def is_edge(self, x, y, theta_vecs, room_center):
        '''
            Input:
                - Edge from x to y, where x and y are in world coordinate
                - theta_vecs: the major orientation. shape: (N, 2)
                - room_center: in world coordinate

            A valid edge must satisfy the three conditions:
                1. Distance between x, y > t
                2. Angle (x, y) close to one of the theta angle
                3. Cross product x->y and room_center->x < 0
        '''
        d = np.linalg.norm(x - y)
        if d < self.cfg['room_shape_opt.min_edge_dist']:
            return False
        v = (y - x) / d
        min_cos_angle = np.min(np.abs(np.dot(theta_vecs, v)))
        if min_cos_angle > np.abs(np.cos(np.pi / 2 - self.cfg['room_shape_opt.min_edge_angle_diff'])):
            return False
        if self.cfg['room_shape_opt.use_cross_product']:
            cv = np.cross(x - room_center, y - x)
            if cv >= 0:
                return False
        return True

    def build_edges_fast(self, uv_nodes, theta_vecs, room_center):
        '''
            Build edges for nodes using tensor operation by replacing the loop
                "for i in range(n) for j in range(n)"
        '''
        xy_nodes = ocg_uv_to_xz(self.ocg, uv_nodes.T).T         # size, 2
        size = uv_nodes.shape[0]

        s_nodes = xy_nodes.reshape(size, 1, -1)
        t_nodes = xy_nodes.reshape(1, size, -1)
        vector_matrix = -(s_nodes - t_nodes)                    # size, size, 2

        # Distance condition
        dist_matrix = np.linalg.norm(vector_matrix, axis=-1)    # size, size
        valid_mask = dist_matrix > self.cfg['room_shape_opt.min_edge_dist']

        # Cross product condition
        if self.cfg['room_shape_opt.use_cross_product']:
            tile = np.tile(s_nodes - room_center, (1, size, 1))      # size, size, 2
            direct_matrix = np.cross(tile, vector_matrix, axis=-1)  # size, size
            valid_mask &= (direct_matrix < 0)

        # Major plane orientation condition
        if self.cfg['room_shape_opt.use_angle_constraint']:
            unit_vecs = vector_matrix / np.expand_dims(dist_matrix+1e-8, -1)
            theta_vecs = theta_vecs.reshape(-1, 2, 1)
            dot = np.dot(unit_vecs, theta_vecs).squeeze(-1)     # size, size, theta_size
            min_cos_angle = np.min(np.abs(dot), axis=-1)        # size*size
            theta_mask = min_cos_angle < np.cos(np.pi / 2 - self.cfg['room_shape_opt.min_edge_angle_diff'])
            valid_mask &= theta_mask

        if self.cfg['room_shape_opt.max_edge_dist'] > 0:
            s_nodes_uv = uv_nodes.reshape(size, 1, -1)
            t_nodes_uv = uv_nodes.reshape(1, size, -1)
            matrix_uv = -(s_nodes_uv - t_nodes_uv)    # size, size, 2
            dist_uv = np.sum(np.abs(matrix_uv), axis=-1)
            max_mask = dist_uv <= self.cfg['room_shape_opt.max_edge_dist']
            valid_mask &= max_mask

        return valid_mask.nonzero()

    def compute_edge_weight(
        self,
        x0, x1,     # in uv space
        p0, p1,     # in xy space
        plane_density_map,
        corner_density_map,
        theta_vecs,
        pose_path_map,
        room_center,
    ):
        '''
            Compute weight for an edge
                1. Theta error
                2. Density function cost
                3. Corner function cost
                4. GoP intersect
                5. (Deprecated) GoP center cost
                6. Number of corner = constant
            Return:
                - scalar weight
                - list of the six weights seperately
        '''
        total_cost = 0
        all_cost = []
        # Theta angular error
        d = np.linalg.norm(p1 - p0)
        v = (p1 - p0) / d
        angles = np.arccos(np.dot(theta_vecs, v)) - np.pi / 2
        angles = angles / np.pi * 180
        min_angle = np.min(np.abs(angles))
        total_cost += self.cfg['room_shape_opt.weight_angle_diff'] * min_angle
        all_cost.append(min_angle)

        # Plane density function
        line_u, line_v = get_line_pixels(x0, x1)
        if self.cfg['room_shape_opt.average_plane_cost']:
            cost = (1 - plane_density_map[line_v, line_u]).mean()
        else:
            cost = (1 - plane_density_map[line_v, line_u]).sum()
        total_cost += self.cfg['room_shape_opt.weight_plane_cost'] * cost
        all_cost.append(cost)

        # Corner density function
        cost = (
            (1 - corner_density_map[x0[1], x0[0]]) / 2 +
            (1 - corner_density_map[x1[1], x1[0]]) / 2
        )
        total_cost += self.cfg['room_shape_opt.weight_corner_cost'] * cost
        all_cost.append(cost)

        # Gop intersect
        num_inter = pose_path_map[line_v, line_u].sum()
        if num_inter > 0:
            if self.cfg['room_shape_opt.use_constant_intersec']:
                total_cost += self.cfg['room_shape_opt.weight_gop_intersec']
            else:
                total_cost += self.cfg['room_shape_opt.weight_gop_intersec'] * num_inter
            all_cost.append(1)
        else:
            all_cost.append(0)

        d = np.linalg.norm(p1 - room_center)
        # TODO: To be removed
        total_cost += self.cfg['room_shape_opt.weight_gop_closeness'] * d
        all_cost.append(d)

        # Number of corners (model simplicity)
        total_cost += self.cfg['room_shape_opt.weight_complex_cost']
        all_cost.append(1)

        assert total_cost >= 0, f'{all_cost}, {total_cost}'
        return total_cost, all_cost

    def build_graph(self):
        '''
            Build the graph for iSPA
        '''
        # TODO: Get rid of the room center
        center = self.get_center()[[0, 2]]

        # Get ocg_map from planes for search space
        all_nodes = self.get_valid_pixels()
        if all_nodes.shape[0] == 0:
            raise SPAError('No valid pixels')

        # Convert theta_angles to unit vectors
        theta_vecs = self.get_theta_vecs()
        if theta_vecs.shape[0] == 0:
            raise SPAError('No valid orientations')

        # Build plane density map
        plane_density_map = self.get_plane_density()
        # Build corner density map
        corner_density_map = self.get_corner_density()

        # Compuate interior area
        inter_map = self.get_interior_area()
        corners = self.get_corners()
        if corners is not None:
            corners = corners[[0, 2], :]
        start_node_idx, end_node_idx, break_line = \
            self.get_start_edge(all_nodes, corners, theta_vecs, center, inter_map)

        x_ids, y_ids = self.build_edges_fast(all_nodes, theta_vecs, center)
        invalid_area = np.zeros_like(inter_map)
        line_u, line_v = get_line_pixels(break_line[0], break_line[1])
        invalid_area[line_v, line_u] = 1

        # Compute edge weights
        graph = {node_idx: dict() for node_idx in range(all_nodes.shape[0])}
        detailed_graph = {node_idx: dict() for node_idx in range(all_nodes.shape[0])}
        pixel_to_node = np.zeros_like(plane_density_map, dtype=np.int32)
        pixel_to_node.fill(-1)
        pixel_to_node[all_nodes[:, 1], all_nodes[:, 0]] = np.arange(0, all_nodes.shape[0], 1)

        for i in range(x_ids.shape[0]):
            xi = x_ids[i]
            yi = y_ids[i]

            x0 = all_nodes[xi, :]
            x1 = all_nodes[yi, :]
            if not self.cfg['room_shape_opt.use_cross_product']:
                line_u, line_v = get_line_pixels(x0, x1)
                if invalid_area[line_v, line_u].sum() > 0:
                    continue

            p0 = ocg_uv_to_xz(self.ocg, x0)
            p1 = ocg_uv_to_xz(self.ocg, x1)
            weight, weights = self.compute_edge_weight(
                x0, x1, p0, p1,
                plane_density_map,
                corner_density_map,
                theta_vecs,
                inter_map,
                center)

            graph[xi][yi] = weight
            detailed_graph[xi][yi] = weights

        start_uv = all_nodes[start_node_idx, :]
        end_uv = all_nodes[end_node_idx, :]
        start_xy = ocg_uv_to_xz(self.ocg, start_uv)
        end_xy = ocg_uv_to_xz(self.ocg, end_uv)

        return dict(
            graph=graph,
            detailed_graph=detailed_graph,
            node_to_pixel=all_nodes,
            pixel_to_node=pixel_to_node,
            start_edge_idx=(start_node_idx, end_node_idx),
            start_edge_xy=(start_xy, end_xy),
            break_line=break_line,
        )

    def solve_graph(self, graph, node_to_pixel, start_idx, end_idx):
        # Solve SPA
        path, dists = dijkstra(graph, start_idx, end_idx, node_to_pixel)
        path.reverse()

        path = np.array(path, dtype=np.int32)
        corners_uv = node_to_pixel[path, :]
        corners_xz = ocg_uv_to_xz(self.ocg, corners_uv.T).T
        return corners_uv, corners_xz

    def estimate_shape(self, save_path=None):
        ''' Estimate the room shape '''
        fig, axs = plt.subplots(2, 2)
        axs[0][0].imshow(
            visualize_spa_info(self)
        )
        axs[0][1].imshow(
            visualize_spa_info(self, draw_plane=True)
        )
        if save_path is not None:
            fig.savefig(save_path)

        start_time = time.time()
        graph_dict = self.build_graph()
        build_time = time.time() - start_time

        graph = graph_dict['graph']
        # detailed_graph = graph_dict['detailed_graph']
        node_to_pixel = graph_dict['node_to_pixel']
        start_idx, end_idx = graph_dict['start_edge_idx']
        break_line = graph_dict['break_line']

        start_uv = node_to_pixel[start_idx, :]
        end_uv = node_to_pixel[end_idx, :]
        start_xz = ocg_uv_to_xz(self.ocg, start_uv.T).T
        end_xz = ocg_uv_to_xz(self.ocg, end_uv.T).T
        axs[0][0].imshow(
            visualize_spa_info(
                self, start_uv=start_uv, end_uv=end_uv, break_line=break_line)
        )
        axs[0][1].imshow(
            visualize_spa_info(
                self, start_uv=start_uv, end_uv=end_uv, break_line=break_line, draw_plane=True)
        )

        start_time = time.time()
        corners_uv, corners_xz = self.solve_graph(graph, node_to_pixel, start_idx, end_idx)
        solve_time = time.time() - start_time
        axs[1][0].imshow(
            visualize_room_result(self, corners_uv=corners_uv, start_uv=start_uv, end_uv=end_uv, draw_plane=False)
        )
        axs[1][1].imshow(
            visualize_room_result(self, corners_uv=corners_uv, start_uv=start_uv, end_uv=end_uv, draw_plane=True)
        )
        if save_path is not None:
            fig.savefig(save_path)
        plt.close(fig)

        return dict(
            start_uv=start_uv,
            end_uv=end_uv,
            start_xz=start_xz,
            end_xz=end_xz,
            corners_uv=corners_uv,
            corners_xz=corners_xz,
            time=(build_time, solve_time),
        )


class SPARefine(SPABasic):
    def __init__(self, cfg, room, ocg_patch, corners_xz, prev_start_corner, prev_end_corner):
        '''
            room: LocalRoom object
        '''
        self.cfg = cfg
        self.room = room
        self.ocg = ocg_patch
        self.corners = corners_xz
        self.prev_start_corner = prev_start_corner
        self.prev_end_corner = prev_end_corner

    def estimate_shape(self, save_path=None):
        ''' Estimate the room shape '''
        fig, axs = plt.subplots(2, 2)
        axs[0][0].imshow(
            visualize_spa_info(self)
        )
        axs[0][1].imshow(
            visualize_spa_info(self, draw_plane=True)
        )
        if save_path is not None:
            fig.savefig(save_path)

        start_time = time.time()
        graph_dict = self.build_graph()
        build_time = time.time() - start_time

        graph = graph_dict['graph']
        node_to_pixel = graph_dict['node_to_pixel']
        start_idx, end_idx = graph_dict['start_edge_idx']
        break_line = graph_dict['break_line']
        start_uv = node_to_pixel[start_idx, :]
        end_uv = node_to_pixel[end_idx, :]
        start_xz = ocg_uv_to_xz(self.ocg, start_uv.T).T
        end_xz = ocg_uv_to_xz(self.ocg, end_uv.T).T

        axs[0][0].imshow(
            visualize_spa_info(
                self, start_uv=start_uv, end_uv=end_uv, break_line=break_line)
        )
        axs[0][1].imshow(
            visualize_spa_info(
                self, start_uv=start_uv, end_uv=end_uv, break_line=break_line, draw_plane=True)
        )
        if save_path is not None:
            fig.savefig(save_path)
        # break_line = graph_dict['break_line']
        # start_uv = node_to_pixel[start_idx, :]
        # end_uv = node_to_pixel[end_idx, :]
        # corners_uv = ocg_xyz_to_uv(self.ocg, corners_xz.T).T
        start_time = time.time()
        corners_uv, corners_xz = self.solve_graph(graph, node_to_pixel, start_idx, end_idx)
        solve_time = time.time() - start_time
        axs[1][0].imshow(
            visualize_room_result(self, corners_uv=corners_uv, start_uv=start_uv, end_uv=end_uv, draw_plane=False)
        )
        axs[1][1].imshow(
            visualize_room_result(self, corners_uv=corners_uv, start_uv=start_uv, end_uv=end_uv, draw_plane=True)
        )
        if save_path is not None:
            fig.savefig(save_path)
        plt.close(fig)

        return dict(
            start_uv=start_uv,
            end_uv=end_uv,
            start_xz=start_xz,
            end_xz=end_xz,
            corners_uv=corners_uv,
            corners_xz=corners_xz,
            time=(build_time, solve_time),
        )

    def get_valid_pixels(self):
        ocg_map = self.ocg.project_xyz_points_to_hist(self.corners)
        ocg_map[ocg_map > 0] = 1

        for i in range(self.cfg['room_shape_opt.refine_box_size']):
            ocg_map = dilation(ocg_map)

        inter_mask = self.get_interior_area()
        # patch = erosion(patch, square(5))
        ocg_map[inter_mask] = 0

        v_idxs, u_idxs = ocg_map.nonzero()
        all_nodes = np.stack([u_idxs, v_idxs], axis=1)
        return all_nodes

    def get_start_edge(self, all_nodes, corners, theta_vecs, center, inter_area):
        valid_edges = []

        N = self.corners.shape[1]
        for i in range(N):
            start_xy = self.corners[:, i]
            end_xy = self.corners[:, (i+1) % N]
            start_uv = ocg_xyz_to_uv(self.ocg, start_xy)
            end_uv = ocg_xyz_to_uv(self.ocg, end_xy)
            line_u, line_v = get_line_pixels(start_uv, end_uv)

            if inter_area[line_v, line_u].sum() > 0:
                continue
            if self.is_edge(end_xy, start_xy, theta_vecs, center):
                valid_edges.append([start_xy, end_xy])

            if self.is_edge(start_xy, end_xy, theta_vecs, center):
                valid_edges.append([end_xy, start_xy])

        if len(valid_edges) == 0:
            raise SPAError('No valid edges find in SPARefine')
        np.random.shuffle(valid_edges)

        def score_similar_to_previous(start_xy, end_xy, threshold=1.0):
            penalty = 0
            if np.linalg.norm(start_xy - self.prev_start_corner) < threshold:
                penalty += 1
            if np.linalg.norm(start_xy - self.prev_end_corner) < threshold:
                penalty += 1
            if np.linalg.norm(end_xy - self.prev_start_corner) < threshold:
                penalty += 1
            if np.linalg.norm(end_xy - self.prev_end_corner) < threshold:
                penalty += 1
            return penalty
        if self.prev_start_corner is not None and self.prev_end_corner is not None:
            # Avoid choosing the same start and end corners if they are given
            valid_edges.sort(
                key=lambda x: score_similar_to_previous(x[0], x[1])
            )

        valid_edges_filtered = []
        for start_node, end_node in valid_edges:
            start_uv = ocg_xyz_to_uv(self.ocg, start_node)
            end_uv = ocg_xyz_to_uv(self.ocg, end_node)

            # In case the start_uv is not in valid pixels, find the closest valid start node (same for end node)
            start_node_idx = np.argmin(np.sum(np.abs(all_nodes - np.expand_dims(start_uv, axis=0)), axis=1))
            end_node_idx = np.argmin(np.sum(np.abs(all_nodes - np.expand_dims(end_uv, axis=0)), axis=1))

            start_uv = all_nodes[start_node_idx, :]
            end_uv = all_nodes[end_node_idx, :]
            edge_length = np.sum(np.abs(start_uv - end_uv))

            found_valid, break_line = self.find_break_line(start_uv, end_uv, inter_area)
            if found_valid:
                valid_edges_filtered.append(
                    (start_node_idx, end_node_idx, break_line, edge_length)
                )

        if len(valid_edges_filtered) == 0:
            raise SPAError("Cannot find a starting edge with a valid break line")
        valid_edges_filtered.sort(key=lambda x: x[-1], reverse=True)     # Sort by edge length
        (start_node_idx, end_node_idx, break_line, edge_length) = valid_edges_filtered[0]
        break_line = (np.array(break_line[0]), np.array(break_line[1]))

        return start_node_idx, end_node_idx, break_line

    def build_edges_fast(self, uv_nodes, theta_vecs, room_center):
        '''
            Build edges for nodes using tensor operation by replacing the loop
                "for i in range(n) for j in range(n)"
        '''
        xy_nodes = ocg_uv_to_xz(self.ocg, uv_nodes.T).T         # size, 2
        size = uv_nodes.shape[0]

        s_nodes = xy_nodes.reshape(size, 1, -1)
        t_nodes = xy_nodes.reshape(1, size, -1)
        vector_matrix = -(s_nodes - t_nodes)                    # size, size, 2
        # Distance condition
        dist_matrix = np.linalg.norm(vector_matrix, axis=-1)    # size, size
        valid_mask = dist_matrix > 0

        # Major plane orientation condition
        if self.cfg['room_shape_opt.use_angle_constraint']:
            unit_vecs = vector_matrix / np.expand_dims(dist_matrix+1e-8, -1)
            theta_vecs = theta_vecs.reshape(-1, 2, 1)
            dot = np.dot(unit_vecs, theta_vecs).squeeze(-1)     # size, size, theta_size
            min_cos_angle = np.min(np.abs(dot), axis=-1)        # size*size
            theta_mask = min_cos_angle < np.cos(np.pi / 2 - self.cfg['room_shape_opt.min_edge_angle_diff'])
            valid_mask &= theta_mask

        s_nodes_uv = uv_nodes.reshape(size, 1, -1)
        t_nodes_uv = uv_nodes.reshape(1, size, -1)
        matrix_uv = -(s_nodes_uv - t_nodes_uv)    # size, size, 2
        dist_uv = np.sum(np.abs(matrix_uv), axis=-1)
        max_mask = dist_uv >= self.cfg['room_shape_opt.refine_box_size'] * 2
        valid_mask &= max_mask

        return valid_mask.nonzero()
