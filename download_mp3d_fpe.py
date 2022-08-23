import argparse
import os
import pandas as pd
import gdown


def download_scenes(opt):

    output_dir = os.path.join(opt.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # list_google_scenes = "./data/pilot_scenes_google_ids.csv"
    list_google_scenes = opt.list_google_scenes
    scenes_ids = pd.read_csv(list_google_scenes).values

    for scene in scenes_ids:
        print(f"Downloading... {scene[0]}")
        url = f"https://drive.google.com/uc?id={scene[1]}"
        output_file = os.path.join(output_dir, scene[0])
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
