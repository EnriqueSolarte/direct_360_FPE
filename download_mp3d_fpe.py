import argparse
import os
import zipfile
import gdown
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def download_scenes(opt):

    output_dir = os.path.join(opt.output_dir, "downloaded")
    os.makedirs(output_dir, exist_ok=True)

    # list_google_scenes = "./data/pilot_scenes_google_ids.csv"
    list_google_scenes = opt.ids_file
    scenes_ids = pd.read_csv(list_google_scenes)

    for gd_id, zip_fn in zip(scenes_ids.Id, scenes_ids.Path):
        print(f"Downloading... {zip_fn}")
        url = f"https://drive.google.com/uc?id={gd_id}"
        output_file = os.path.join(output_dir, zip_fn)
        gdown.download(url, output_file, quiet=False)

    mp3d_fpe_dir = os.path.join(Path(output_dir).parent, "mp3d_fpe_dir")
    
    for zip_fn in tqdm(os.listdir(output_dir)):
        if "_npy.zip" in zip_fn:
            continue 
        print(f"Unzipping... {zip_fn}")
        zip_file = os.path.join(output_dir, zip_fn)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(mp3d_fpe_dir)
    
    list_npy_files = [fn for fn in os.listdir(output_dir) if "_npy.zip" in fn]
    for zip_fn in tqdm(list_npy_files):
        print(f"Unzipping... {zip_fn}")
        path = "/".join(zip_fn.split("_")[:2])
        zip_file = os.path.join(output_dir, zip_fn)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(mp3d_fpe_dir, path))
    
    
def get_passed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default="./dataset", help='Output dataset directory...')
    parser.add_argument('--ids_file', type=str, default="./data/mp3d_fpe_google_drive_ids.csv", help="lists of IDS to download from GoogleDrive")
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = get_passed_args()
    download_scenes(opt)
