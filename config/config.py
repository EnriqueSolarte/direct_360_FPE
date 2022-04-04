import os
import yaml
import git

def read_config(config_file):
    assert os.path.isfile(
        config_file), f"Config file {config_file} does not exit"

    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    # cfg['path.results_dir'] = os.path.join(cfg.get("path.results_dir"), cfg.get("data.eval_version", "test_evaluation"))

    repo = git.Repo(search_parent_directories=True)
    cfg['git.commit_brach'] = repo.head._get_commit().name_rev
    print(f"Config file {config_file} loaded successfully")
    return cfg


def overwrite_scene_data(cfg, scene):
    """
    Change the relevant scene data given a Path scene 
    """
    scene_split = scene.split("/")
    cfg["data.scene"] = scene_split[-2]
    cfg["data.scene_version"] = scene_split[-1]
    cfg["data.scene_category"] = scene_split[-3]

    return cfg


def overwite_version(cfg, version):
    cfg['data.eval_version'] = version
    dir_results = os.path.join(cfg.get("path.results_dir"), cfg.get("data.eval_version"))
    # cfg['path.results_dir'] = dir_results
    os.makedirs(dir_results, exist_ok=True)
