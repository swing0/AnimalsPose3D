# 03_run_animals.py
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
# Adapted for quadruped animals

import torch.optim as optim
import os
import sys
import errno
from time import time

from common.animals_dataset import AnimalsDataset
from common.arguments import parse_args
from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.utils import deterministic_random

# 导入动物数据集

args = parse_args()
print(args)

try:
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

print('Loading dataset...')
dataset_path = 'npz/real_npz/data_3d_' + args.dataset + '.npz'

if args.dataset == 'animals':
    dataset = AnimalsDataset(dataset_path)
else:
    raise KeyError('Invalid dataset')

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints_path = 'npz/real_npz/data_2d_' + args.dataset + '_' + args.keypoints + '.npz'
if not os.path.exists(keypoints_path):
    print(f"Warning: 2D keypoints file {keypoints_path} not found!")
    print("Please run 02_prepare_data_animals.py first to generate 2D projections.")
    sys.exit(1)

keypoints = np.load(keypoints_path, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

# 验证数据完整性
for subject in dataset.subjects():
    if subject not in keypoints:
        print(f'Warning: Subject {subject} is missing from the 2D detections dataset')
        continue

    for action in dataset[subject].keys():
        if action not in keypoints[subject]:
            print(f'Warning: Action {action} of subject {subject} is missing from 2D detections')
            continue

        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            if keypoints[subject][action][cam_idx].shape[0] < mocap_length:
                print(f'Warning: 2D sequence shorter than 3D for {subject}/{action}/cam{cam_idx}')
                # 截断3D数据以匹配2D
                dataset[subject][action]['positions_3d'][cam_idx] = \
                    dataset[subject][action]['positions_3d'][cam_idx][:keypoints[subject][action][cam_idx].shape[0]]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

# 归一化2D坐标
for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            if subject in dataset.cameras() and cam_idx < len(dataset.cameras()[subject]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

# 设置训练和测试主体
subjects_train = args.subjects_train.split(',') if hasattr(args, 'subjects_train') else ['Animal']
subjects_semi = [] if not hasattr(args,
                                  'subjects_unlabeled') or not args.subjects_unlabeled else args.subjects_unlabeled.split(
    ',')
if not args.render:
    subjects_test = args.subjects_test.split(',') if hasattr(args, 'subjects_test') else ['Animal']
else:
    subjects_test = [args.viz_subject]

semi_supervised = len(subjects_semi) > 0
if semi_supervised and not dataset.supports_semi_supervised():
    raise RuntimeError('Semi-supervised training is not implemented for this dataset')


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    """获取指定主体的数据"""
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []

    for subject in subjects:
        if subject not in keypoints:
            continue

        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                if len(cams) == len(poses_2d):
                    for cam in cams:
                        if 'intrinsic' in cam:
                            out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and subject in dataset._data and action in dataset[subject] and 'positions_3d' in \
                    dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                if len(poses_3d) == len(poses_2d):
                    for i in range(len(poses_3d)):
                        out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            if n_frames > 0:
                start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
                out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
                if out_poses_3d is not None and i < len(out_poses_3d):
                    out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None and i < len(out_poses_3d):
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


# 动作过滤
action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

# 获取验证数据
cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

# 构建模型
filter_widths = [int(x) for x in args.architecture.split(',')]
if not args.disable_optimizations and not args.dense and args.stride == 1:
    model_pos_train = TemporalModelOptimized1f(
        poses_valid_2d[0].shape[-2] if len(poses_valid_2d) > 0 else 17,
        poses_valid_2d[0].shape[-1] if len(poses_valid_2d) > 0 else 2,
        dataset.skeleton().num_joints(),
        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
        channels=args.channels
    )
else:
    model_pos_train = TemporalModel(
        poses_valid_2d[0].shape[-2] if len(poses_valid_2d) > 0 else 17,
        poses_valid_2d[0].shape[-1] if len(poses_valid_2d) > 0 else 2,
        dataset.skeleton().num_joints(),
        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
        channels=args.channels, dense=args.dense
    )

model_pos = TemporalModel(
    poses_valid_2d[0].shape[-2] if len(poses_valid_2d) > 0 else 17,
    poses_valid_2d[0].shape[-1] if len(poses_valid_2d) > 0 else 2,
    dataset.skeleton().num_joints(),
    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
    channels=args.channels, dense=args.dense
)

receptive_field = model_pos.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

# 加载检查点
if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

# 创建测试生成器
if len(poses_valid_2d) > 0:
    test_generator = UnchunkedGenerator(
        cameras_valid, poses_valid, poses_valid_2d,
        pad=pad, causal_shift=causal_shift, augment=False,
        kps_left=kps_left, kps_right=kps_right,
        joints_left=joints_left, joints_right=joints_right
    )
    print('INFO: Testing on {} frames'.format(test_generator.num_frames()))
else:
    test_generator = None
    print('WARNING: No validation data available')

# 训练循环
if not args.evaluate:
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate

    if semi_supervised:
        cameras_semi, _, poses_semi_2d = fetch(subjects_semi, action_filter, parse_3d_poses=False)

        if not args.disable_optimizations and not args.dense and args.stride == 1:
            model_traj_train = TemporalModelOptimized1f(
                poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels
            )
        else:
            model_traj_train = TemporalModel(
                poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                channels=args.channels, dense=args.dense
            )

        model_traj = TemporalModel(
            poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], 1,
            filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
            channels=args.channels, dense=args.dense
        )

        if torch.cuda.is_available():
            model_traj = model_traj.cuda()
            model_traj_train = model_traj_train.cuda()

        optimizer = optim.Adam(list(model_pos_train.parameters()) + list(model_traj_train.parameters()),
                               lr=lr, amsgrad=True)
    else:
        optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

    lr_decay = args.lr_decay

    # 训练状态记录
    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    # 创建数据生成器
    if len(poses_train_2d) > 0:
        train_generator = ChunkedGenerator(
            args.batch_size // args.stride, cameras_train, poses_train, poses_train_2d, args.stride,
            pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right
        )
        train_generator_eval = UnchunkedGenerator(
            cameras_train, poses_train, poses_train_2d,
            pad=pad, causal_shift=causal_shift, augment=False
        )
        print('INFO: Training on {} frames'.format(train_generator_eval.num_frames()))
    else:
        print('ERROR: No training data available!')
        sys.exit(1)

    # 训练循环主体
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        N = 0
        model_pos_train.train()

        # 简化的训练循环 - 仅监督学习
        for _, batch_3d, batch_2d in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

            inputs_3d[:, :, 0] = 0  # Remove root

            optimizer.zero_grad()

            # 预测3D姿态
            predicted_3d_pos = model_pos_train(inputs_2d)
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            loss_3d_pos.backward()
            optimizer.step()

        losses_3d_train.append(epoch_loss_3d_train / N if N > 0 else 0)

        # 评估
        with torch.no_grad():
            model_pos.load_state_dict(model_pos_train.state_dict())
            model_pos.eval()

            if test_generator and not args.no_eval:
                epoch_loss_3d_valid = 0
                N_valid = 0

                for cam, batch, batch_2d in test_generator.next_epoch():
                    inputs_3d = torch.from_numpy(batch.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

                    if torch.cuda.is_available():
                        inputs_3d = inputs_3d.cuda()
                        inputs_2d = inputs_2d.cuda()

                    inputs_3d[:, :, 0] = 0

                    predicted_3d_pos = model_pos(inputs_2d)
                    loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                    epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                    N_valid += inputs_3d.shape[0] * inputs_3d.shape[1]

                losses_3d_valid.append(epoch_loss_3d_valid / N_valid if N_valid > 0 else 0)

        elapsed = (time() - start_time) / 60

        if args.no_eval:
            print('[%d] time %.2f lr %f 3d_train %f' % (
                epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' % (
                epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000,
                losses_3d_valid[-1] * 1000 if losses_3d_valid else 0))

        # 学习率衰减
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        epoch += 1

        # 保存检查点
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_traj': model_traj_train.state_dict() if semi_supervised else None,
            }, chk_path)


# 评估函数
def evaluate(test_generator, action=None, return_predictions=False, use_trajectory_model=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0

    with torch.no_grad():
        model_pos.eval()
        N = 0

        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            predicted_3d_pos = model_pos(inputs_2d)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')

    e1 = (epoch_loss_3d_pos / N) * 1000 if N > 0 else 0
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('----------')

    return e1, 0, 0, 0  # 简化版本，只返回MPJPE


# 最终评估
if not args.render and test_generator:
    print('Evaluating...')

    # 简化的评估 - 只评估所有数据
    if test_generator:
        e1, e2, e3, ev = evaluate(test_generator)
        print('Final MPJPE:', e1, 'mm')

print('Done!')

# python 03_run_animals.py -d animals -k gt -str Animal -ste Animal --architecture 3,3,3 -e 200 -b 256 --channels 512 --dropout 0.2 --learning-rate 0.001 --downsample 4 --subset 0.5 --checkpoint checkpoint
# python 03_run_animals.py -d animals -k gt -str Animal -ste Animal --architecture 3,3,3,3 -e 100 -b 256 --channels 512 --dropout 0.2 --learning-rate 0.001 --downsample 4 --subset 0.5 --checkpoint checkpoint


# python 03_run_animals.py -d animals -k gt --evaluate epoch_60.bin