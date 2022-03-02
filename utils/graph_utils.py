"""
    Modified from Floor-SP
    https://github.com/woodfrog/floor-sp/blob/master/floor-sp/utils/floorplan_utils/graph_algorithms.py
    Graph algorithms for solving floorplan structures
"""

import numpy as np


def dijkstra(graph, source_idx, end_idx, all_nodes):
    dists = [np.inf for _ in range(len(graph))]
    prevs = [None for _ in range(len(graph))]
    unvisited = [i for i in range(len(graph))]

    dists[source_idx] = 0

    while len(unvisited) > 0:
        min_node = _find_min_node(dists, unvisited)
        unvisited.remove(min_node)

        for neighbour_node in graph[min_node].keys():
            prev_node_idx = prevs[min_node]
            if neighbour_node == prev_node_idx:
                continue  # skip the incoming path, it's not possible to go back to prev again
            # if min_node == source_idx:
            #     # Special case: the first edge must consider the edge from end_node->start_node
            #     if check_go_back(
            #         all_nodes[end_idx],
            #         all_nodes[min_node],
            #         all_nodes[neighbour_node]
            #     ):
            #         continue
            # if neighbour_node == end_idx:
            #     # Special case: the last edge must consider the edge from end_node->start_node
            #     if check_go_back(
            #         all_nodes[min_node],
            #         all_nodes[neighbour_node],
            #         all_nodes[source_idx]
            #     ):
            #         continue

            if prev_node_idx is not None:
                # if check_same_line(
                #     all_nodes[prev_node_idx],
                #     all_nodes[min_node],
                #     all_nodes[neighbour_node]
                # ):
                #     continue
                if check_go_back(
                    all_nodes[prev_node_idx],
                    all_nodes[min_node],
                    all_nodes[neighbour_node]
                ):
                    continue
            dist = graph[min_node][neighbour_node]
            if dist + dists[min_node] < dists[neighbour_node]:
                dists[neighbour_node] = dist + dists[min_node]
                prevs[neighbour_node] = min_node

    path = list()
    next_node = end_idx

    while next_node is not None:
        path.append(next_node)
        next_node = prevs[next_node]
        # if next_node == source_idx:
        #     print('Find path back to starting node')
        #     break
        # try:
        #     next_node = prevs[next_node]
        # except Exception as e:
        #     print(prevs, next_node)
        #     raise(e)

    path = refine_path(path, all_nodes)
    return path, dists


def _find_min_node(dists, unvisited):
    min_dist = None
    min_node = None
    for node in unvisited:
        if min_node is None:
            min_node = node
            min_dist = dists[node]
        elif dists[node] < min_dist:
            min_node = node
            min_dist = dists[node]
    return min_node


def check_same_line(source, mid, end):
    # Ensure the path source->mid->end won't be on the same line forward
    v1 = mid - source
    v2 = end - mid
    dist1 = np.linalg.norm(v1)
    dist2 = np.linalg.norm(v2)
    if dist1 == 0 or dist2 == 0:
        import pdb
        pdb.set_trace()

    cos = np.dot(v1, v2) / (dist1 * dist2)
    if cos >= 0.9396:
        # < 20 degree
        return True
    else:
        return False


def check_go_back(source, mid, end):
    # Ensure the path source->mid->end won't go backward
    v1 = mid - source
    v2 = end - mid
    dist1 = np.linalg.norm(v1)
    dist2 = np.linalg.norm(v2)
    if dist1 == 0 or dist2 == 0:
        import pdb
        pdb.set_trace()

    cos = np.dot(v1, v2) / (dist1 * dist2)
    if cos <= -0.866:
        # 150 degree
        return True
    else:
        return False


def refine_path(path, all_nodes):
    if len(path) < 4:
        return path

    # Currently, it only check the starting and the ending node
    new_path = [x for x in path]
    if check_same_line(all_nodes[path[-1]], all_nodes[path[0]], all_nodes[path[1]]):
        # path[0] is useless
        new_path.pop(0)
    elif check_go_back(all_nodes[path[-1]], all_nodes[path[0]], all_nodes[path[1]]):
        # path[0] is useless
        new_path.pop(0)

    if check_same_line(all_nodes[path[-2]], all_nodes[path[-1]], all_nodes[path[0]]):
        # path[-1] is useless
        new_path.pop(-1)
    elif check_go_back(all_nodes[path[-2]], all_nodes[path[-1]], all_nodes[path[0]]):
        # path[-1] is useless
        new_path.pop(-1)

    # TODO: Check closed areas and save only the largest one
    return new_path
