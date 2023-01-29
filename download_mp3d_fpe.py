import argparse
import os

import gdown
import pandas as pd


def download_scenes(opt):

    output_dir = os.path.join(opt.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # list_google_scenes = "./data/pilot_scenes_google_ids.csv"
    list_google_scenes = opt.ids_file
    scenes_ids = pd.read_csv(list_google_scenes)

    for zip_fn, gd_id in zip(scenes_ids.Id, scenes_ids.Path):
        print(f"Downloading... {zip_fn}")
        url = f"https://drive.google.com/uc?id={gd_id}"
        output_file = os.path.join(output_dir, zip_fn)
        gdown.download(url, output_file, quiet=False)


def get_passed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="./dataset", help='Output dataset directory...')
    parser.add_argument('--ids_file', type=str, default="./data/pilot_scenes_google_ids.csv", help="lists of IDS to download from GoogleDrive")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = get_passed_args()
    download_scenes(opt)
