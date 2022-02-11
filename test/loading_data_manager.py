
from torch import import_ir_module
from config import read_config
from data_manager import DataManager

def test_data_manager():
    print("TESTING DataManager...")
    config_file = "./config/config.yaml"
    cfg = read_config(config_file)
    dt = DataManager(cfg)
    return True

if __name__ == '__main__':
    test_data_manager()