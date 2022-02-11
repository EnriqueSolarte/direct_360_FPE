from config import read_config
from data_manager import DataManager
from utils.geometry_utils import extend_array_to_homogeneous
from visualization.vispy_utils import plot_color_plc
import numpy as np
from test import plotting_data, reproject_frames
from visualization.vispy_dynamic_visualization.dynamic_vis_utils import plot_pcl_from_list_fr
import matplotlib.pyplot as plt
# plotting_data()


if __name__ == '__main__':
   reproject_frames()