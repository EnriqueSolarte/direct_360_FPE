import os
import yaml
from dotenv import load_dotenv

load_dotenv('env.env')

def read_config(config_file):
    assert os.path.isfile(
        config_file), f"Config file {config_file} does not exit"

    with open(config_file, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Config file {config_file} loaded successfully")
    return cfg
