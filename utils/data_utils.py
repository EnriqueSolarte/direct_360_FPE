from utils.io import get_list_scenes, save_csv_file
from config import read_config
import os


def list_and_save_scenes():

    list_scene = get_list_scenes(
        data_dir=os.getenv("MP3D_FPE_DIR"),
        flag_file='minos_poses.txt',
        exclude=('rgb', 'depth', 'hn_mp3d', 'hoho_mp3d')
    )
    save_csv_file("./data/scene_list.csv", list_scene)


def flatten_lists_of_lists(list_of_list):
    flat_list = [item for sublist in list_of_list for item in sublist]
    return flat_list

