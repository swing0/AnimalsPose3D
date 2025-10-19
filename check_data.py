# check_data_advanced.py
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_dataset_size(data_3d_path, data_2d_path):
    """分析数据集规模并提供参数建议"""

    print("=" * 60)
    print("数据集规模分析报告")
    print("=" * 60)

    # 检查3D数据
    if os.path.exists(data_3d_path):
        data_3d = np.load(data_3d_path, allow_pickle=True)
        print(f"3D数据文件: {data_3d_path}")

        if 'positions_3d' in data_3d.files:
            positions_3d = data_3d['positions_3d'].item()
            total_frames_3d = 0
            subject_info = {}

            for subject in positions_3d.keys():
                subject_frames = 0
                subject_actions = []

                for action in positions_3d[subject].keys():
                    shape = positions_3d[subject][action].shape
                    frames = shape[0]
                    subject_frames += frames
                    total_frames_3d += frames
                    subject_actions.append((action, frames, shape))

                subject_info[subject] = {
                    'total_frames': subject_frames,
                    'actions': subject_actions
                }

                print(f"\n主体 [{subject}]:")
                print(f"  总帧数: {subject_frames}")
                print(f"  动作数量: {len(subject_actions)}")
                for action, frames, shape in subject_actions:
                    print(f"    - {action}: {frames}帧, 形状: {shape}")

            print(f"\n3D数据总计: {total_frames_3d} 帧")

            # 分析数据分布
            analyze_data_distribution(subject_info, total_frames_3d)

            # 提供参数建议
            provide_training_recommendations(total_frames_3d, subject_info)

    # 检查2D数据
    if os.path.exists(data_2d_path):
        data_2d = np.load(data_2d_path, allow_pickle=True)
        print(f"\n2D数据文件: {data_2d_path}")

        if 'positions_2d' in data_2d.files:
            positions_2d = data_2d['positions_2d'].item()
            total_views = 0

            for subject in positions_2d.keys():
                for action in positions_2d[subject].keys():
                    views = len(positions_2d[subject][action])
                    total_views += views
                    if views > 0:
                        sample_shape = positions_2d[subject][action][0].shape
                        print(f"  {subject}/{action}: {views}个视角, 示例形状: {sample_shape}")

            print(f"2D数据总计: {total_views} 个视角序列")


def analyze_data_distribution(subject_info, total_frames):
    """分析数据分布"""
    print("\n" + "=" * 40)
    print("数据分布分析")
    print("=" * 40)

    if total_frames == 0:
        return

    # 按主体分析
    print("按主体分布:")
    for subject, info in subject_info.items():
        percentage = (info['total_frames'] / total_frames) * 100
        print(f"  {subject}: {info['total_frames']}帧 ({percentage:.1f}%)")

    # 数据规模分类
    print(f"\n数据规模分类:")
    if total_frames < 5000:
        print("  📊 小规模数据集 (< 5K 帧)")
    elif total_frames < 50000:
        print("  📊 中等规模数据集 (5K - 50K 帧)")
    elif total_frames < 200000:
        print("  📊 大规模数据集 (50K - 200K 帧)")
    else:
        print("  📊 超大规模数据集 (> 200K 帧)")


def provide_training_recommendations(total_frames, subject_info):
    """根据数据规模提供训练参数建议"""
    print("\n" + "=" * 40)
    print("训练参数建议")
    print("=" * 40)

    # 基础参数建议
    if total_frames < 5000:
        print("🔍 小数据集建议:")
        print("  - 架构: 3,3,3 (浅层网络)")
        print("  - 批次大小: 128-256")
        print("  - 训练轮数: 100-200")
        print("  - 通道数: 256-512")
        print("  - Dropout: 0.3-0.5")
        print("  - 学习率: 0.001")
        print("  ⚠️  注意: 数据量较少，容易过拟合")

    elif total_frames < 50000:
        print("🔍 中等数据集建议:")
        print("  - 架构: 3,3,3,3 或 5,5,5")
        print("  - 批次大小: 512-1024")
        print("  - 训练轮数: 200-300")
        print("  - 通道数: 512-1024")
        print("  - Dropout: 0.2-0.3")
        print("  - 学习率: 0.001")

    elif total_frames < 200000:
        print("🔍 大数据集建议:")
        print("  - 架构: 3,3,3,3,3 或 5,5,5,5")
        print("  - 批次大小: 1024-2048")
        print("  - 训练轮数: 300-500")
        print("  - 通道数: 1024")
        print("  - Dropout: 0.1-0.2")
        print("  - 学习率: 0.001-0.0005")

    else:
        print("🔍 超大数据集建议:")
        print("  - 架构: 5,5,5,5,5,5 (深层网络)")
        print("  - 批次大小: 2048-4096")
        print("  - 训练轮数: 500+")
        print("  - 通道数: 1024-2048")
        print("  - Dropout: 0.05-0.1")
        print("  - 学习率: 0.0005")

    # 数据划分建议
    print(f"\n📋 数据划分建议:")
    subjects = list(subject_info.keys())
    if len(subjects) > 1:
        print("  - 使用不同主体进行训练/测试划分")
        train_subjects = ','.join(subjects[:-1])
        test_subjects = subjects[-1]
        print(f"    训练: {train_subjects}")
        print(f"    测试: {test_subjects}")
    else:
        print("  - 单一主体，建议按时间或动作划分")
        print("  - 例如: --actions 'action_0*' 用于训练，'action_1*' 用于测试")


def check_data_quality(data_3d_path):
    """检查数据质量"""
    print("\n" + "=" * 40)
    print("数据质量检查")
    print("=" * 40)

    if not os.path.exists(data_3d_path):
        print("数据文件不存在")
        return

    data_3d = np.load(data_3d_path, allow_pickle=True)
    if 'positions_3d' not in data_3d.files:
        print("未找到3D位置数据")
        return

    positions_3d = data_3d['positions_3d'].item()

    for subject in positions_3d.keys():
        print(f"\n主体 [{subject}] 数据质量:")
        for action in positions_3d[subject].keys():
            data = positions_3d[subject][action]

            # 检查NaN值
            nan_count = np.isnan(data).sum()
            # 检查无限值
            inf_count = np.isinf(data).sum()
            # 检查数值范围
            data_range = np.ptp(data)  # 峰峰值范围
            data_mean = np.mean(data)
            data_std = np.std(data)

            print(f"  {action}:")
            print(f"    - NaN值: {nan_count}")
            print(f"    - 无限值: {inf_count}")
            print(f"    - 数值范围: {data_range:.3f}")
            print(f"    - 均值: {data_mean:.3f} ± {data_std:.3f}")

            if nan_count > 0 or inf_count > 0:
                print("    ⚠️  数据质量问题!")


def generate_training_commands(total_frames):
    """生成训练命令"""
    print("\n" + "=" * 40)
    print("推荐训练命令")
    print("=" * 40)

    if total_frames < 5000:
        print("小数据集命令:")
        print('python run_animals.py -d animals -k gt -str Animal -ste Animal  --architecture 3,3,3 -e 150 -b 256 '
              '--channels 512   --dropout 0.4 --checkpoint checkpoint')

    elif total_frames < 50000:
        print("中等数据集命令:")
        print('python run_animals.py -d animals -k gt -str Animal -ste Animal  --architecture 3,3,3,3 -e 250 -b 512 '
              '--channels 1024  --dropout 0.25 --checkpoint checkpoint')

    else:
        print("大数据集命令:")
        print('python run_animals.py -d animals -k gt -str Animal -ste Animal  --architecture 5,5,5,5 -e 400 -b 1024 '
              '--channels 1024   --dropout 0.15 --checkpoint checkpoint')


if __name__ == '__main__':
    # 数据文件路径
    data_3d_path = 'npz/real_npz/data_3d_animals.npz'
    data_2d_path = 'npz/real_npz/data_2d_animals_gt.npz'

    # 运行分析
    analyze_dataset_size(data_3d_path, data_2d_path)
    check_data_quality(data_3d_path)

    # 获取总帧数以生成命令
    if os.path.exists(data_3d_path):
        data_3d = np.load(data_3d_path, allow_pickle=True)
        if 'positions_3d' in data_3d.files:
            positions_3d = data_3d['positions_3d'].item()
            total_frames = sum(
                positions_3d[subject][action].shape[0]
                for subject in positions_3d.keys()
                for action in positions_3d[subject].keys()
            )
            generate_training_commands(total_frames)