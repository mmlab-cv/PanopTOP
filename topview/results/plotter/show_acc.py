# %%
import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

from accuracy import *
from plot import *
# %%

def plot_results_of_experiment(experiment):
    current_file_path = pathlib.Path(__file__).parent.absolute()
<<<<<<< HEAD
    gt_file = f'{current_file_path}/../res_files/{experiment}_gt.txt'
    pred_file = f'{current_file_path}/../res_files/{experiment}_predictions.txt'
=======
    gt_file = f'{current_file_path}/res_files/{experiment}_gt.txt'
    pred_file = f'{current_file_path}/res_files/{experiment}_predictions.txt'
>>>>>>> brodkap/reverted_results

    gt = np.loadtxt(gt_file)
    gt = gt.reshape(gt.shape[0], -1, 3)

    pred = np.loadtxt(pred_file)
    pred = pred.reshape(pred.shape[0], -1, 3)

    keypoints_num = 15
    names = ['j'+str(i+1) for i in range(keypoints_num)]

    dist, acc = compute_dist_acc_wrapper(pred, gt, max_dist=0.3, num=100)
    
    fig, ax = plt.subplots()
    plot_acc(ax, dist, acc, names)
    fig.savefig(f'plots/{experiment}_accuracy.png')
    plt.show()

    mean_err = compute_mean_err(pred, gt)
    fig, ax = plt.subplots()
    plot_mean_err(ax, mean_err, names)
    fig.savefig(f'plots/{experiment}_joints.png')
    plt.show()

    print('mean_err: {}'.format(mean_err))
    mean_err_all = compute_mean_err(pred.reshape((-1, 1, 3)), gt.reshape((-1, 1,3)))
    print('mean_err_all: ', mean_err_all)


<<<<<<< HEAD
#for exp in ["itop_itop_itop", "itop_itop_panoptic", "itop_both_panoptic", "panoptic_panoptic_panoptic", "panoptic_panoptic_itop", "panoptic_both_itop", "both_both_itop", "both_both_panoptic"]:
#    plot_results_of_experiment(exp)

# %%
def get_accuracy_for_joints(experiment, needed_acc = 0.1):
    current_file_path = pathlib.Path(__file__).parent.absolute()
    gt_file = f'{current_file_path}/../res_files/{experiment}_gt.txt'
    pred_file = f'{current_file_path}/../res_files/{experiment}_predictions.txt'
=======
for exp in ["itop_itop_itop", "itop_itop_panoptic", "itop_both_panoptic", "panoptic_panoptic_panoptic", "panoptic_panoptic_itop", "panoptic_both_itop", "both_both_itop", "both_both_panoptic"]:
    plot_results_of_experiment(exp)

'''
# %%
def get_accuracy_for_joints(experiment, needed_acc = 0.1):
    current_file_path = pathlib.Path(__file__).parent.absolute()
    gt_file = f'{current_file_path}/res_files/{experiment}_gt.txt'
    pred_file = f'{current_file_path}/res_files/{experiment}_predictions.txt'
>>>>>>> brodkap/reverted_results

    gt = np.loadtxt(gt_file)
    gt = gt.reshape(gt.shape[0], -1, 3)

    pred = np.loadtxt(pred_file)
    pred = pred.reshape(pred.shape[0], -1, 3)

    dist, acc = compute_dist_acc_wrapper(pred, gt, max_dist=0.3, num=100)
    #print("dist, acc", dist, acc)
    #print("shape", dist.shape, acc.shape)
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
# %%
create_accuracy_df_for_experiments(0.1).to_latex()
create_accuracy_df_for_experiments(0.2).to_latex()

# %%
<<<<<<< HEAD
=======
'''
>>>>>>> brodkap/reverted_results
