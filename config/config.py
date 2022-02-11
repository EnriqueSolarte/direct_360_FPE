import os
import yaml
from dotenv import load_dotenv

load_dotenv('env.env')

def read_config(config_file):
    assert os.path.isfile(
        config_file), f"Config file {config_file} does not exit"

    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    mp3d_fpe_dir = os.path.join(os.getenv('MP3D_FPE_DIR'), cfg['scene_category'], cfg['scene'], cfg['scene_version'])
    cfg['mp3d_fpe_dir'] = mp3d_fpe_dir

    print(f"Config file {config_file} loaded successfully")
    return cfg
