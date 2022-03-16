import os
import yaml
from dotenv import load_dotenv
import git

load_dotenv('env.env')


def read_config(config_file):
    assert os.path.isfile(
        config_file), f"Config file {config_file} does not exit"

    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    mp3d_fpe_dir = os.path.join(os.getenv('MP3D_FPE_DIR'), cfg['scene_category'], cfg['scene'], cfg['scene_version'])
    cfg['mp3d_fpe_dir'] = mp3d_fpe_dir

    dir_results = os.path.join(os.getenv("RESULTS_DIR"), cfg.get("version", "test_evaluation"))
    cfg['results_dir'] = dir_results

    repo = git.Repo(search_parent_directories=True)
    cfg['git.commit_brach'] = repo.head._get_commit().name_rev
    print(f"Config file {config_file} loaded successfully")
    return cfg


def overwrite_scene_data(cfg, scene):
    """
    Change the relevant scene data given a Path scene 
    """
    scene_split = scene.split("/")
    cfg["scene"] = scene_split[-2]
    cfg["scene_version"] = scene_split[-1]
    cfg["scene_category"] = scene_split[-3]
    cfg["mp3d_fpe_dir"] = os.path.join(os.getenv("MP3D_FPE_DIR"), scene)

    dir_results = os.path.join(os.getenv("RESULTS_DIR"), cfg.get("version", "test_evaluation"))
    cfg['results_dir'] = dir_results

    return cfg


def overwite_version(cfg, version):
    cfg['eval_version'] = version
    dir_results = os.path.join(os.getenv("RESULTS_DIR"), cfg.get("eval_version", "test_evaluation"))
    cfg['results_dir'] = dir_results
    os.makedirs(dir_results, exist_ok=True)
