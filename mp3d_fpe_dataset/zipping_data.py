import zipfile
import os
from tqdm import tqdm
from pathlib import Path
from geometry_perception_utils.io_utils import create_directory, process_arcname, get_abs_path
from geometry_perception_utils.config_utils import save_cfg
import hydra
import argparse


def zip_data(root_dir, zf, list_fn):
    list_arc_fn = process_arcname(list_fn, root_dir)
    [
        (
            print(f"zipping {fn}"),
            zf.write(
                os.path.join(root_dir, fn),
                compress_type=zipfile.ZIP_DEFLATED,
                arcname=fn,
            ),
        )
        for fn in tqdm(list_arc_fn)
    ]


def zip_directory(src_dir, dst_dir, root, scene_version):
    assert os.path.exists(
        src_dir), f"Source directory {src_dir} does not exist"

    create_directory(dst_dir, delete_prev=False)
    list_all_fn = [f for f in os.listdir(
        src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    zip_filename = os.path.join(dst_dir, f"{scene_version}.zip")
    with zipfile.ZipFile(file=zip_filename, mode="w") as zf:
        list_fn = [
            os.path.join(src_dir, f"{fn}")
            for fn in list_all_fn
        ]
        zip_data(root, zf, list_fn)


def get_passed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=f"/media/datasets/mp3d_fpe/MULTI_ROOM_SCENES",
                        help='Output directory for results')
    parser.add_argument('--zip_dir', type=str, default=f"/media/datasets/mp3d_fpe/zipped_data",
                        help='Output directory for results')
    opt = parser.parse_args()
    return opt


def main(opt):
    data_dir = Path(opt.data_dir)
    zip_dir = Path(opt.zip_dir)

    for scene in data_dir.iterdir():
        _scene = scene.stem
        if ".DS_Store" in _scene:
                continue
        for scene_version in scene.iterdir():
            _version = scene_version.stem
            if ".DS_Store" in _version:
                continue
            o_dir = create_directory(zip_dir / _scene / _version)

            # images
            src_dir = f'{scene_version}/rgb'
            zip_directory(src_dir, Path(o_dir), root=data_dir,
                          scene_version=f"{_scene}_{_version}_rgb")

            # depth maps
            src_dir = f'{scene_version}/depth/tiff'
            zip_directory(src_dir, Path(o_dir), root=data_dir,
                          scene_version=f"{_scene}_{_version}_depth")

            # hn-mp3d
            src_dir = f'{[f for f in scene_version.glob("vo*")][0]}/hn_mp3d'
            zip_directory(src_dir, Path(o_dir), root=data_dir,
                          scene_version=f"{_scene}_{_version}_hn_mp3d")

            # labels
            est_poses = [f for f in [f for f in scene_version.glob(
                "vo*")][0].iterdir() if f.is_file()]
            fpe_gt = [f for f in scene_version.iterdir() if f.is_file()]
            list_labels = est_poses + fpe_gt
            zip_filename = os.path.join(
                o_dir, f"{_scene}_{_version}_labels.zip")
            with zipfile.ZipFile(file=zip_filename, mode="w") as zf:
                zip_data(data_dir, zf, list_labels)


if __name__ == "__main__":
    opt = get_passed_args()
    main(opt)
