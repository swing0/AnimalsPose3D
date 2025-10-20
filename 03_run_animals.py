# 03_run_animals.py
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

args = parse_args()
print("=== Animal Pose Estimation Training ===")
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
            # 为每个2D视角创建一个对应的3D数据副本
            # 假设有4个视角（前、侧、斜、顶）
            for _ in range(4):  # 创建4个相同的3D数据副本，对应4个2D视角
                positions_3d.append(anim['positions'].copy())
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
keypoints_path = 'npz/real_npz/data_2d_' + args.dataset + '_' + args.keypoints + '.npz'
if not os.path.exists(keypoints_path):
    print(f"Error: 2D keypoints file {keypoints_path} not found!")
    print("Please run 02_prepare_data_animals.py first to generate 2D projections.")
    sys.exit(1)

keypoints = np.load(keypoints_path, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

# 验证数据完整性
print('Validating data...')
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

        # 修复：现在 positions_3d 只有一个元素（简化后的数据）
        positions_3d = dataset[subject][action]['positions_3d']
        keypoints_2d = keypoints[subject][action]

        # 检查3D数据和2D数据的帧数是否匹配
        for cam_idx in range(len(keypoints_2d)):
            if cam_idx < len(positions_3d):  # 添加边界检查
                mocap_length = positions_3d[cam_idx].shape[0]
                if keypoints_2d[cam_idx].shape[0] < mocap_length:
                    print(f'Truncating 3D data for {subject}/{action}/cam{cam_idx}')
                    positions_3d[cam_idx] = positions_3d[cam_idx][:keypoints_2d[cam_idx].shape[0]]
            else:
                print(f'Warning: Camera index {cam_idx} out of range for 3D data in {subject}/{action}')

        # 修复断言：现在 positions_3d 和 keypoints_2d 应该都有相同数量的视角
        assert len(keypoints_2d) == len(positions_3d), \
            f"Mismatch in number of views: 2D has {len(keypoints_2d)}, 3D has {len(positions_3d)}"

# 归一化2D坐标
# 简化归一化2D坐标部分
print('跳过2D坐标归一化（使用简单投影的归一化坐标）...')

# 在设置训练和测试主体部分之前添加自动主体检测
def get_all_available_subjects():
    """获取数据集中所有可用的动物主体"""
    all_subjects = list(dataset.subjects())
    print(f"📊 数据集包含 {len(all_subjects)} 个动物主体: {', '.join(all_subjects)}")
    return all_subjects

# 设置训练和测试主体
if args.subjects_train == 'Animal' or not args.subjects_train:
    # 自动使用所有动物
    all_subjects = get_all_available_subjects()
    subjects_train = all_subjects
    print("🎯 使用所有动物进行训练")
else:
    subjects_train = args.subjects_train.split(',')

if args.subjects_test == 'Animal' or not args.subjects_test:
    # 自动使用所有动物
    if 'all_subjects' not in locals():
        all_subjects = get_all_available_subjects()
    subjects_test = all_subjects
    print("🎯 使用所有动物进行评估")
else:
    subjects_test = args.subjects_test.split(',')

subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',')

semi_supervised = len(subjects_semi) > 0
if semi_supervised and not dataset.supports_semi_supervised():
    raise RuntimeError('Semi-supervised training is not implemented for this dataset')

print(f'Training subjects: {subjects_train}')
print(f'Test subjects: {subjects_test}')
if semi_supervised:
    print(f'Semi-supervised subjects: {subjects_semi}')


# 简化版 fetch 函数
def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    """获取指定主体的数据"""
    out_poses_3d = []
    out_poses_2d = []

    for subject in subjects:
        if subject not in dataset.subjects():
            continue

        for action in dataset[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            # 获取2D姿态
            if subject in keypoints and action in keypoints[subject]:
                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)):
                    out_poses_2d.append(poses_2d[i])

            # 获取3D姿态
            if parse_3d_poses and subject in dataset._data and action in dataset[subject] and 'positions_3d' in \
                    dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                if len(poses_3d) == len(poses_2d):
                    for i in range(len(poses_3d)):
                        out_poses_3d.append(poses_3d[i])

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

    # 返回None作为相机参数，因为我们不再使用相机
    return None, out_poses_3d, out_poses_2d


# 动作过滤
action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)

# 获取验证数据
print('Loading validation data...')
cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

# 构建模型
print('Building model...')
filter_widths = [int(x) for x in args.architecture.split(',')]

# 获取输入维度
if len(poses_valid_2d) > 0:
    num_joints = poses_valid_2d[0].shape[-2]
    input_2d_dim = poses_valid_2d[0].shape[-1]
else:
    num_joints = dataset.skeleton().num_joints()
    input_2d_dim = 2

if not args.disable_optimizations and not args.dense and args.stride == 1:
    model_pos_train = TemporalModelOptimized1f(
        num_joints, input_2d_dim, dataset.skeleton().num_joints(),
        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
        channels=args.channels
    )
else:
    model_pos_train = TemporalModel(
        num_joints, input_2d_dim, dataset.skeleton().num_joints(),
        filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
        channels=args.channels, dense=args.dense
    )

model_pos = TemporalModel(
    num_joints, input_2d_dim, dataset.skeleton().num_joints(),
    filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
    channels=args.channels, dense=args.dense
)

receptive_field = model_pos.receptive_field()
print(f'Receptive field: {receptive_field} frames')
pad = (receptive_field - 1) // 2
if args.causal:
    print('Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print(f'Trainable parameter count: {model_params}')

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()
    print('Using CUDA')

# 加载检查点
if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print(f'Loading checkpoint {chk_filename}')
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print(f'This model was trained for {checkpoint["epoch"]} epochs')
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
    print(f'Testing on {test_generator.num_frames()} frames')
else:
    test_generator = None
    print('WARNING: No validation data available')

# 训练循环
if not args.evaluate:
    print('Loading training data...')
    cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)

    lr = args.learning_rate

    if semi_supervised:
        cameras_semi, _, poses_semi_2d = fetch(subjects_semi, action_filter, parse_3d_poses=False)

        if not args.disable_optimizations and not args.dense and args.stride == 1:
            model_traj_train = TemporalModelOptimized1f(
                num_joints, input_2d_dim, 1,
                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels
            )
        else:
            model_traj_train = TemporalModel(
                num_joints, input_2d_dim, 1,
                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout,
                channels=args.channels, dense=args.dense
            )

        model_traj = TemporalModel(
            num_joints, input_2d_dim, 1,
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

    # 创建数据生成器
    if len(poses_train_2d) > 0:
        train_generator = ChunkedGenerator(
            args.batch_size, cameras_train, poses_train, poses_train_2d, args.stride,
            pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right
        )
        train_generator_eval = UnchunkedGenerator(
            cameras_train, poses_train, poses_train_2d,
            pad=pad, causal_shift=causal_shift, augment=False
        )
        print(f'Training on {train_generator_eval.num_frames()} frames')
    else:
        print('ERROR: No training data available!')
        sys.exit(1)

    print('Starting training...')
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        N = 0
        model_pos_train.train()

        # 训练循环
        for _, batch_3d, batch_2d in train_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

            # 移除根节点
            inputs_3d[:, :, 0] = 0

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

        # 打印训练状态
        if args.no_eval:
            print('[%03d/%03d] time %.2f lr %f 3d_train %f' % (
                epoch + 1, args.epochs, elapsed, lr, losses_3d_train[-1] * 1000))
        else:
            print('[%03d/%03d] time %.2f lr %f 3d_train %f 3d_valid %f' % (
                epoch + 1, args.epochs, elapsed, lr,
                losses_3d_train[-1] * 1000,
                losses_3d_valid[-1] * 1000 if losses_3d_valid else 0))

        # 学习率衰减
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        epoch += 1

        # 保存检查点
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epoch_{:03d}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'random_state': train_generator.random_state(),
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
                'model_traj': model_traj_train.state_dict() if semi_supervised else None,
                'loss_3d_train': losses_3d_train,
                'loss_3d_valid': losses_3d_valid,
            }, chk_path)

# 最终评估
if not args.render and test_generator:
    print('\n=== Final Evaluation ===')
    model_pos.eval()

    epoch_loss_3d_pos = 0
    N = 0

    with torch.no_grad():
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_3d = torch.from_numpy(batch.astype('float32'))
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()

            inputs_3d[:, :, 0] = 0
            predicted_3d_pos = model_pos(inputs_2d)
            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

    if N > 0:
        final_mpjpe = (epoch_loss_3d_pos / N) * 1000
        print(f'Final MPJPE: {final_mpjpe:.2f} mm')
    else:
        print('No test data available for evaluation')

print('Done!')


'''

# 默认训练所有动物
python 03_run_animals.py -d animals -k gt -e 100 -b 256 --checkpoint checkpoint_all_animals

# 默认评估所有动物
python 03_run_animals.py -d animals -k gt --evaluate epoch_010.bin --checkpoint checkpoint_all_animals

# 仍然可以手动指定
python 03_run_animals.py -d animals -k gt -str "Addax_Female,Addax_Juvenile" -ste "Addax_Male" -e 100 --checkpoint checkpoint_specific



'''