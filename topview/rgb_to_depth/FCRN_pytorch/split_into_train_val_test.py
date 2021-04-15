import os
from shutil import copyfile

path = "../../../external_volume/PANOPTIC/01"
indices = os.path.join(path, "indices.py")

exec(open(indices).read())
# good_indices
good_indices = sorted(good_indices)
num = len(good_indices)
train_num = int(0.7 * num)
val_num = int(0.15 * num)
test_num = num - train_num - val_num

train_indices = good_indices[:train_num]
val_indices = good_indices[train_num: train_num + val_num]
test_indices = good_indices[train_num + val_num: ]

'''
os.mkdir(os.path.join(path, "train"))
os.mkdir(os.path.join(path, "validation"))
os.mkdir(os.path.join(path, "test"))

os.mkdir(os.path.join(path, "train", "rgb_textured"))
os.mkdir(os.path.join(path, "validation", "rgb_textured"))
os.mkdir(os.path.join(path, "test", "rgb_textured"))

os.mkdir(os.path.join(path, "train", "depth"))
os.mkdir(os.path.join(path, "validation", "depth"))
os.mkdir(os.path.join(path, "test", "depth"))
'''

# depth example
# 00024980.png

# rgb example
# 00024210_1.png
# 00024210_2.png
# 00024210_3.png
# 00024210_4.png

depth_path = "depth"
rgb_path = "rgb_textured"

indices_dict = {
    "train": train_indices,
    "validation": val_indices,
    "test": test_indices
}

for mode, indices in indices_dict.items():
    for ind in indices:
        ind_str = f'{ind:08}'

        for rgb_suffix in ["_1", "_2", "_3", "_4"]:
            src = f"{path}/rgb_textured/{ind_str}{rgb_suffix}.png"
            dst = f"{path}/{mode}/rgb_textured/{ind_str}{rgb_suffix}.png"
            copyfile(src, dst)

        src = f"{path}/depth/{ind_str}.png"
        dst = f"{path}/{mode}/depth/{ind_str}.png"
        copyfile(src, dst)