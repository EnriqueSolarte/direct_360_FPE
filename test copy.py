from config import read_config
from data_manager import DataManager
from src import DirectFloorPlanEstimation
from utils.visualization.vispy_utils import plot_color_plc
from utils.enum import CAM_REF
import numpy as np
from utils.data_utils import flatten_lists_of_lists
import matplotlib.pyplot as plt


def main(config_file):
    cfg = read_config(config_file=config_file)
    dt = DataManager(cfg)
    # fpe = DirectFloorPlanEstimation(dt)
    list_ly = dt.get_list_ly(cam_ref=CAM_REF.WC)

    plot_color_plc(np.hstack([ly.boundary for ly in list_ly]).T)

    print('done')


if __name__ == '__main__':
    # TODO read from  passed args
    config_file = "./config/config.yaml"
    main(config_file=config_file)
