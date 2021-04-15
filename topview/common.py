from pathlib import Path
import os
import numpy as np


def get_big_data_path():
    # Important: keep absolute path
    path_of_file = Path(__file__).parent.absolute()

    # Paths for data repository
    DATA_PATH = "/home/Datasets"

    if os.path.isdir(DATA_PATH):
        return DATA_PATH
    else:
        print(f"path {DATA_PATH} is not valid")
        return None
