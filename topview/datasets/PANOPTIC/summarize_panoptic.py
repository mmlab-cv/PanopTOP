import numpy as np
import os
import gzip
import sys
from collections import Counter

sys.path.append("../../..")
from topview.common import get_big_data_path

DATA_DIR = f"{get_big_data_path()}/PANOPTIC"
cloud_path = f"{DATA_DIR}/clouds/"
joints_path = f"{DATA_DIR}/joints/"

clouds_files = os.listdir(cloud_path)
joints_files = os.listdir(joints_path)


#assert(sorted(clouds_files) == sorted(joints_files))

def get_filename_person_id(filename):
    return filename[:2]

ids = map(get_filename_person_id, clouds_files)
counter = Counter(ids)

print(dict(counter))
print(sorted(counter.keys()))

