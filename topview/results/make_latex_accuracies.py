import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

from accuracy import *
from plot import *

def get_accuracy_for_joints(experiment, needed_acc = 0.1):
    current_file_path = pathlib.Path(__file__).parent.absolute()
    gt_file = f'{current_file_path}/res_files/{experiment}_gt.txt'
    pred_file = f'{current_file_path}/res_files/{experiment}_predictions.txt'

    gt = np.loadtxt(gt_file)
    gt = gt.reshape(gt.shape[0], -1, 3)

    pred = np.loadtxt(pred_file)
    pred = pred.reshape(pred.shape[0], -1, 3)

    dist, acc = compute_dist_acc_wrapper(pred, gt, max_dist=0.3, num=100)
    acc_ind = np.where(dist == needed_acc)
    return acc[:, acc_ind].flatten()

def create_accuracy_df_for_experiments(needed_acc = 0.1):
    results_acc = []
    for exp in ["itop_itop_itop", "itop_itop_panoptic", "itop_both_panoptic", "panoptic_panoptic_panoptic", "panoptic_panoptic_itop", "panoptic_both_itop", "both_both_itop", "both_both_panoptic"]:
        exp_acc = get_accuracy_for_joints(exp, needed_acc)
        res_acc = {f"j{i+1}": el for i, el in enumerate(exp_acc)}
        
        res_acc = {
            "experiment": exp,
            **res_acc,
        }

        results_acc.append(res_acc)
    df = pd.DataFrame(results_acc)
    df = df.round(3)
    return df

print("0.1m")
print(create_accuracy_df_for_experiments(0.1).to_latex())
print("0.2m")
print(create_accuracy_df_for_experiments(0.2).to_latex())