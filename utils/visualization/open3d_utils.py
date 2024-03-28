import pickle
import numpy as np
import open3d as o3d

def visualize_geometry(pkl_geometry_path):
    try:
        fpe = pickle.load(open(pkl_geometry_path, 'rb')) 
    except:
        print("No pickle file found")
        
    all_rooms_corners = fpe["corners_list"]

    R = 0.8
    cam_height = 1.5
    rooms_mesh = []
    layout_mesh = o3d.geometry.TriangleMesh()
    room_mesh_subdivided = o3d.geometry.TriangleMesh()

    pts = []
    li = []
    triangles = []
    for room_corners in all_rooms_corners:
        x_list = room_corners[:, 0].tolist()
        z_list = room_corners[:, 1].tolist()
        k = len(pts)
        n = len(x_list)
        centroid_up = [(np.sum(x_list)/len(x_list), cam_height, np.sum(z_list)/len(z_list))]
        centroid_down = [(np.sum(x_list)/len(x_list), -R * cam_height, np.sum(z_list)/len(z_list))]
        for x,z in zip(x_list, z_list):
            # Despite of what's on the literature, cam_height * R gives height that looks better when textured
            pts += [(x, cam_height, z), (x, - cam_height * R, z)]
            for i in range(0, 2*len(x_list), 2):
                triangles.append([k+(i+0)%(2*n), k+(i+1)%(2*n), k+(i+2)%(2*n)])
                triangles.append([k+(i+1)%(2*n), k+(i+3)%(2*n), k+(i+2)%(2*n)])
                triangles.append([k+(i+0)%(2*n), k+(i+2)%(2*n), 2*len(x_list)])

                li.append([k+i, k+(i+2)%(2*n)])
                li.append([k+i, k+(i+1)])
                li.append([k+i+1, k+(i+3)%(2*n)])
                li.append([k+i+1, 2*len(x_list)+1])
                li.append([k+i, 2*len(x_list)])

        pts = pts+centroid_up+centroid_down

        room_mesh = o3d.geometry.TriangleMesh()
        room_mesh.vertices = o3d.utility.Vector3dVector(pts)
        room_mesh.triangles = o3d.utility.Vector3iVector(triangles)

        rooms_mesh.append(room_mesh)
        room_mesh_subdivided = room_mesh.subdivide_midpoint(3)
        layout_mesh += room_mesh_subdivided

    o3d.visualization.draw(room_mesh_subdivided)
    return room_mesh_subdivided
