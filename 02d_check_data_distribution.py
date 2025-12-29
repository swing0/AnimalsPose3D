import numpy as np
import os


def analyze_npz(file_path, name="Dataset"):
    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)

    # 获取主要数据键名 (positions_2d 或 positions_3d)
    main_key = 'positions_2d' if 'positions_2d' in data else 'positions_3d'
    content = data[main_key].item()

    all_coords = []

    print(f"\n=== 分析报告: {name} ===")
    print(f"文件路径: {file_path}")

    # 递归提取所有帧的坐标
    for sub in content:
        for act in content[sub]:
            # 2D 数据通常是列表格式（多视角），3D 可能是单数组
            frames = content[sub][act]
            if isinstance(frames, list):
                for view in frames:
                    all_coords.append(view.reshape(-1, view.shape[-1]))
            else:
                all_coords.append(frames.reshape(-1, frames.shape[-1]))

    all_coords = np.concatenate(all_coords, axis=0)

    # 计算统计指标
    min_val = np.min(all_coords, axis=0)
    max_val = np.max(all_coords, axis=0)
    mean_val = np.mean(all_coords, axis=0)
    std_val = np.std(all_coords, axis=0)

    print(f"数据总点数: {len(all_coords)}")
    print(f"维度分布 (X, Y, Z/None):")
    print(f"  最小值 (Min): {min_val}")
    print(f"  最大值 (Max): {max_val}")
    print(f"  平均值 (Mean): {mean_val}")
    print(f"  标准差 (Std):  {std_val}")

    # 特殊检查：Root 节点是否中心化
    # 我们检查每一帧的第一行（Root）是否接近 [0, 0, 0]
    roots = []
    for sub in content:
        for act in content[sub]:
            frames = content[sub][act]
            if isinstance(frames, list):
                for view in frames: roots.append(view[:, 0, :])
            else:
                roots.append(frames[:, 0, :])

    root_mean = np.mean(np.concatenate(roots), axis=0)
    print(f"Root 节点平均偏移 (应接近0): {root_mean}")


if __name__ == "__main__":
    # 分析 2D 数据
    analyze_npz(r'npz\real_npz\data_2d_animals_gt.npz', "2D Detection Data")
    # 分析 3D 数据
    analyze_npz(r'npz\real_npz\data_3d_animals.npz', "3D Ground Truth Data")