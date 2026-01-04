# common/arguments.py
import argparse
import numpy as np
import os


def get_all_subjects(dataset_path):
    """自动获取数据集中的所有动物主体"""
    try:
        if os.path.exists(dataset_path):
            dataset = np.load(dataset_path, allow_pickle=True)
            data = dataset['positions_3d'].item() if 'positions_3d' in dataset else dataset.item()
            subjects = list(data.keys())
            return subjects
    except:
        pass
    return ['Animal']  # 默认值


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for quadruped animal pose estimation')

    # 首先获取数据集路径以确定默认主体
    dataset_path = 'npz/real_npz/data_3d_animals.npz'
    all_subjects = get_all_subjects(dataset_path)
    all_subjects_str = ",".join(all_subjects)

    # General arguments
    parser.add_argument('-d', '--dataset', default='animals', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')

    # 修改默认值为所有动物
    parser.add_argument('-str', '--subjects-train', default=all_subjects_str, type=str, metavar='LIST',
                        help='training subjects separated by comma (default: all subjects)')
    parser.add_argument('-ste', '--subjects-test', default=all_subjects_str, type=str, metavar='LIST',
                        help='test subjects separated by comma (default: all subjects)')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-frequency', default=10, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')

    # Model arguments
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-drop', '--dropout', default=0.25, type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.95, type=float, metavar='LR',
                        help='learning rate decay per epoch')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='disable train-time flipping')
    parser.add_argument('-no-tta', '--no-test-time-augmentation', dest='test_time_augmentation', action='store_false',
                        help='disable test-time flipping')
    parser.add_argument('-arc', '--architecture', default='3,3,3', type=str, metavar='LAYERS',
                        help='filter widths separated by comma')
    parser.add_argument('--causal', action='store_true', help='use causal convolutions for real-time processing')
    parser.add_argument('-ch', '--channels', default=512, type=int, metavar='N',
                        help='number of channels in convolution layers')

    # Experimental
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR',
                        help='downsample frame rate by factor (semi-supervised)')
    parser.add_argument('--warmup', default=1, type=int, metavar='N', help='warm-up epochs for semi-supervision')
    parser.add_argument('--no-eval', action='store_true',
                        help='disable epoch evaluation while training (small speed-up)')
    parser.add_argument('--dense', action='store_true',
                        help='use dense convolutions instead of dilated convolutions')
    parser.add_argument('--disable-optimizations', action='store_true',
                        help='disable optimized model for single-frame predictions')
    parser.add_argument('--linear-projection', action='store_true',
                        help='use only linear coefficients for semi-supervised projection')
    parser.add_argument('--no-bone-length', action='store_false', dest='bone_length_term',
                        help='disable bone length term in semi-supervised settings')
    parser.add_argument('--no-proj', action='store_true', help='disable projection for semi-supervised setting')

    # Visualization
    parser.add_argument('--viz-subject', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
    parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')

    # Animal-specific parameters
    parser.add_argument('--animal-joints', default=17, type=int, metavar='N',
                        help='number of joints in animal skeleton')
    parser.add_argument('--fps', default=50, type=int, metavar='N',
                        help='frame rate for animal motion data')

    parser.set_defaults(bone_length_term=True)
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)

    args = parser.parse_args()

    # 打印检测到的主体信息
    if args.subjects_train == all_subjects_str:
        print(f"📊 自动检测到 {len(all_subjects)} 个动物主体: {', '.join(all_subjects)}")

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()

    return args