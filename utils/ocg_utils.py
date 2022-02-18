
import numpy as np

def compute_uv_bins(pcl, grid_size=None, padding=100):
    """
    Computes a tuple of bins needed to represet the passed PCL
    """
    assert pcl.shape[0] == 3
    
    x = pcl[0, :]
    z = pcl[2, :]

    u_edges = np.mgrid[np.min(x)-padding*grid_size:np.max(x)+padding*grid_size:grid_size]
    v_edges = np.mgrid[np.min(z)-padding*grid_size:np.max(z)+padding*grid_size:grid_size]
    return u_edges, v_edges



def project_xyz_to_uv(xyz_points, u_bins, v_bins):
    """
    Projects a set of point xyz (3, n) into a grid map defined by the passed bins
    """
    grid_size = abs(u_bins[1] - u_bins[0])
    shape = (v_bins.size-1, u_bins.size-1)

    x_cell_u = [np.argmin(abs(p - u_bins-grid_size*0.5)) % shape[1]
                for p in xyz_points[0, :]]
    z_cell_v = [np.argmin(abs(p - v_bins-grid_size*0.5)) % shape[0]
                for p in xyz_points[2, :]]
    
    return np.vstack((x_cell_u, z_cell_v))
