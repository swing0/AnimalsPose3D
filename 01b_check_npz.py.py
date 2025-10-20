# check_npz.py
import numpy as np
import os
from pathlib import Path


def check_npz_structure(npz_file_path):
    """
    检查NPZ文件的结构和内容
    """
    print(f"检查NPZ文件: {npz_file_path}")
    print("=" * 60)

    if not os.path.exists(npz_file_path):
        print(f"错误: 文件不存在 - {npz_file_path}")
        return

    try:
        # 加载NPZ文件
        data = np.load(npz_file_path, allow_pickle=True)

        # 显示文件中的键
        print("NPZ文件中的键:")
        for key in data.files:
            print(f"  - {key}")

        print("\n" + "=" * 60)

        # 检查主要数据结构
        if 'positions_3d' in data:
            positions_3d = data['positions_3d'].item()
            print("positions_3d 结构:")
            print_structure(positions_3d)

            # 统计信息
            print_statistics(positions_3d)

            # 数据质量检查
            check_data_quality(positions_3d)

        else:
            print("警告: 未找到 'positions_3d' 键")
            # 显示所有数组的形状
            for key in data.files:
                if isinstance(data[key], np.ndarray):
                    print(f"{key}: shape={data[key].shape}, dtype={data[key].dtype}")
                else:
                    print(f"{key}: type={type(data[key])}")

    except Exception as e:
        print(f"读取NPZ文件时出错: {e}")


def print_structure(positions_3d, indent=0):
    """
    递归打印数据结构
    """
    indent_str = "  " * indent

    for key, value in positions_3d.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_structure(value, indent + 1)
        elif isinstance(value, np.ndarray):
            print(f"{indent_str}{key}: ndarray {value.shape}, dtype={value.dtype}")
        else:
            print(f"{indent_str}{key}: {type(value)}")


def print_statistics(positions_3d):
    """
    打印数据统计信息
    """
    print("\n" + "=" * 60)
    print("数据统计:")
    print("=" * 60)

    total_animals = 0
    total_animations = 0
    total_frames = 0
    animal_stats = []

    for animal_name, animations in positions_3d.items():
        animal_frames = 0
        animal_animations = 0

        for anim_name, anim_data in animations.items():
            if isinstance(anim_data, np.ndarray):
                frames = anim_data.shape[0]
                animal_frames += frames
                animal_animations += 1
                total_frames += frames
                total_animations += 1

        total_animals += 1
        animal_stats.append((animal_name, animal_animations, animal_frames))

    print(f"动物种类总数: {total_animals}")
    print(f"动画序列总数: {total_animations}")
    print(f"总帧数: {total_frames}")

    print(f"\n各动物详细统计:")
    for animal_name, anim_count, frame_count in sorted(animal_stats, key=lambda x: x[2], reverse=True):
        print(f"  {animal_name}: {anim_count}个动画, {frame_count}帧")

    # 帧数分布统计
    if total_animations > 0:
        frame_counts = []
        for animal_name, animations in positions_3d.items():
            for anim_name, anim_data in animations.items():
                if isinstance(anim_data, np.ndarray):
                    frame_counts.append(anim_data.shape[0])

        if frame_counts:
            print(f"\n动画帧数分布:")
            print(f"  最短动画: {min(frame_counts)} 帧")
            print(f"  最长动画: {max(frame_counts)} 帧")
            print(f"  平均帧数: {sum(frame_counts) / len(frame_counts):.1f} 帧")
            print(f"  总动画数: {len(frame_counts)} 个")


def check_data_quality(positions_3d):
    """
    检查数据质量
    """
    print("\n" + "=" * 60)
    print("数据质量检查:")
    print("=" * 60)

    zero_frames_count = 0
    short_animations = 0
    data_issues = []

    for animal_name, animations in positions_3d.items():
        for anim_name, anim_data in animations.items():
            if isinstance(anim_data, np.ndarray):
                frames, keypoints, dims = anim_data.shape

                # 检查形状
                if keypoints != 17 or dims != 3:
                    data_issues.append(f"{animal_name}.{anim_name}: 形状异常 {anim_data.shape}, 期望 (frames, 17, 3)")

                # 检查零帧动画
                if frames == 0:
                    zero_frames_count += 1
                    data_issues.append(f"{animal_name}.{anim_name}: 零帧动画")

                # 检查短动画
                if frames < 10:
                    short_animations += 1

                # 检查NaN值
                if np.isnan(anim_data).any():
                    nan_count = np.isnan(anim_data).sum()
                    data_issues.append(f"{animal_name}.{anim_name}: 包含 {nan_count} 个NaN值")

                # 检查无限值
                if np.isinf(anim_data).any():
                    inf_count = np.isinf(anim_data).sum()
                    data_issues.append(f"{animal_name}.{anim_name}: 包含 {inf_count} 个无限值")

                # 检查零值比例
                zero_ratio = np.sum(anim_data == 0) / anim_data.size
                if zero_ratio > 0.5:
                    data_issues.append(f"{animal_name}.{anim_name}: 零值比例过高 ({zero_ratio:.1%})")

    # 输出质量问题
    if data_issues:
        print("发现的数据质量问题:")
        for issue in data_issues:
            print(f"  ⚠️  {issue}")
    else:
        print("✅ 未发现明显数据质量问题")

    if zero_frames_count > 0:
        print(f"⚠️  发现 {zero_frames_count} 个零帧动画")

    if short_animations > 0:
        print(f"⚠️  发现 {short_animations} 个短动画 (<10帧)")


def check_sample_data(positions_3d, num_samples=3):
    """
    检查样本数据
    """
    print("\n" + "=" * 60)
    print("样本数据检查:")
    print("=" * 60)

    sample_count = 0
    for animal_name, animations in positions_3d.items():
        if sample_count >= num_samples:
            break

        for anim_name, anim_data in animations.items():
            if isinstance(anim_data, np.ndarray) and anim_data.shape[0] > 0:
                print(f"\n样本: {animal_name}.{anim_name}")
                print(f"  形状: {anim_data.shape}")
                print(f"  数据类型: {anim_data.dtype}")
                print(f"  数值范围: [{anim_data.min():.3f}, {anim_data.max():.3f}]")
                print(f"  第一帧第一个关键点: {anim_data[0, 0, :]}")
                print(f"  最后一帧第一个关键点: {anim_data[-1, 0, :]}")

                sample_count += 1
                if sample_count >= num_samples:
                    break
        if sample_count >= num_samples:
            break


def main():
    # 设置NPZ文件路径
    npz_file_path = r"npz\real_npz\data_3d_animals.npz"
    check_npz_structure(npz_file_path)


if __name__ == "__main__":
    main()