import numpy as np
np.set_printoptions(suppress=True)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse

import pathlib
import os
import sys

from lib.solver import train_epoch, val_epoch, test_epoch
from network.v2v_model import V2VModel
from network.utils import V2VVoxelization

sys.path.append('..')
from datasets.DatasetFactory import DatasetFactory


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Hand Keypoints Estimation Training')
    parser.add_argument('--resume',  default=-1, type=int, help='resume after epoch')
    parser.add_argument('--directory_res' , default="./", type=str, help='directory where to save results')
    parser.add_argument('--startepoch', default=0, type=int, help='number of start epoch')
    parser.add_argument('--epochs', default=10, type=int, help='number of epochs in training')
    parser.add_argument('--train_dataset', default='ITOP', type=str, choices=['PANOPTIC', 'ITOP', 'BOTH'])
    parser.add_argument('--validation_dataset', default='NONE', type=str, choices=['PANOPTIC', 'ITOP', 'BOTH', 'NONE'])
    parser.add_argument('--test_dataset', default='NONE', type=str, choices=['PANOPTIC', 'ITOP', 'BOTH', 'NONE'])
    parser.add_argument('--experiment_code', default='NONE', type=str)

    return parser.parse_args()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device: ", device)

dtype = torch.float

args = parse_args()
resume_train = args.resume >= 0
resume_after_epoch = args.resume

save_checkpoint = True

current_file_path = pathlib.Path(__file__).parent.absolute()
checkpoint_dir = f'{current_file_path}/../../../external_volume/training_files/V2V_checkpoint/{args.train_dataset.lower()}'

start_epoch = args.startepoch
epochs_num = args.epochs

## Data, transform, dataset and loader
print('==> Preparing data ..')
keypoints_num = 15

voxelization_train = V2VVoxelization(augmentation=True)
voxelization_val = V2VVoxelization(augmentation=False)


def transform_train(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_train({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return (torch.from_numpy(input), torch.from_numpy(heatmap))

def transform_val(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return (torch.from_numpy(input), torch.from_numpy(heatmap))

def transform_test(sample):
    points, refpoint = sample['points'], sample['refpoint']
    input = voxelize_input(points, refpoint)
    return torch.from_numpy(input), torch.from_numpy(refpoint.reshape((1, -1)))


# TODO: Why so small?
#batch_size = 12
batch_size = 10


if args.validation_dataset == "NONE":
    dataset_factory = DatasetFactory(args.train_dataset, "ITOP", "ITOP", transform_train, transform_val, transform_test)
    training_phase = True
else:
    dataset_factory = DatasetFactory(args.train_dataset, args.validation_dataset, args.test_dataset, transform_train, transform_val, transform_test)
    training_phase = False

train_set = dataset_factory.get_train()
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6)

val_set = dataset_factory.get_validation()
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6)

test_set = dataset_factory.get_test()
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=6)


# Model, criterion and optimizer
print('==> Constructing model ..')
net = V2VModel(input_channels=1, output_channels=keypoints_num)

net = net.to(device, dtype)
if device == torch.device('cuda'):
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    print('cudnn.enabled: ', torch.backends.cudnn.enabled)

criterion = nn.MSELoss()

optimizer = optim.Adam(net.parameters())
# optimizer = optim.RMSprop(net.parameters(), lr=2.5e-4)

if training_phase:
    print('==> Training ..')
    for epoch in range(start_epoch, start_epoch + epochs_num):
        print('Epoch: {}'.format(epoch))
        train_loss = train_epoch(net, criterion, optimizer, train_loader, device=device, dtype=dtype)

        if save_checkpoint:
            if not os.path.exists(checkpoint_dir): 
                os.mkdir(checkpoint_dir)

            checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')
            checkpoint = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }
            torch.save(checkpoint, checkpoint_file)

if not training_phase:
    val_scores = []

    print('==> Validating ..')
    for epoch in range(start_epoch, start_epoch + epochs_num):
        print('Epoch: {}'.format(epoch))
        checkpoint_file = os.path.join(checkpoint_dir, 'epoch'+str(epoch)+'.pth')
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['model_state_dict'])

        val_loss = val_epoch(net, criterion, val_loader, device=device, dtype=dtype)
        val_scores.append(val_loss)

    epoch_to_test = np.argmin(val_scores)

    checkpoint = torch.load(os.path.join(checkpoint_dir, 'epoch'+str(epoch_to_test)+'.pth'))
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    ## Test
    print('==> Testing ..')
    voxelize_input = voxelization_train.voxelize
    evaluate_keypoints = voxelization_train.evaluate


    def transform_output(heatmaps, refpoints):
        keypoints = evaluate_keypoints(heatmaps, refpoints)
        return keypoints


    class BatchResultCollector():
        def __init__(self, samples_num, transform_output):
            self.samples_num = samples_num
            self.transform_output = transform_output
            self.keypoints = None
            self.idx = 0
        
        def __call__(self, data_batch):
            inputs_batch, outputs_batch, extra_batch = data_batch
            outputs_batch = outputs_batch.cpu().numpy()
            refpoints_batch = extra_batch.cpu().numpy()
        
            keypoints_batch = self.transform_output(outputs_batch, refpoints_batch)

            if self.keypoints is None:
                # Initialize keypoints until dimensions awailable now
                self.keypoints = np.zeros((self.samples_num, *keypoints_batch.shape[1:]))

            batch_size = keypoints_batch.shape[0] 
            self.keypoints[self.idx:self.idx+batch_size] = keypoints_batch
            self.idx += batch_size

        def get_result(self):
            return self.keypoints


    print('Test on test dataset ..')
    def save_keypoints(filename, keypoints):
        # Reshape one sample keypoints into one line
        keypoints = keypoints.reshape(keypoints.shape[0], -1)
        np.savetxt(filename, keypoints, fmt='%0.4f')


    test_res_collector = BatchResultCollector(len(test_set), transform_output)
    gt_path = f"{current_file_path}/../results/res_files/{args.experiment_code}_gt.txt"
    gt_joints = np.concatenate(test_set.get_joints(), axis=0).reshape(-1, 3 * keypoints_num)
    save_keypoints(gt_path, gt_joints)


    test_epoch(net, test_loader, test_res_collector, device, dtype)
    keypoints_test = test_res_collector.get_result()
    prediction_path = f"{current_file_path}/../results/res_files/{args.experiment_code}_predictions.txt"
    save_keypoints(prediction_path, keypoints_test)
    '''
    fit_on_train = False
    if fit_on_train:
        print('Fit on train dataset ..')
        fit_set = PoseDataset(points_train, joints_train, centers_train, "train", transform_test)
        fit_loader = torch.utils.data.DataLoader(fit_set, batch_size=batch_size, shuffle=False, num_workers=6)
        fit_res_collector = BatchResultCollector(len(fit_set), transform_output)

        test_epoch(net, fit_loader, fit_res_collector, device, dtype)
        keypoints_fit = fit_res_collector.get_result()
        save_keypoints(f"{args.directoryres}/train_res.txt", keypoints_fit)
    '''
