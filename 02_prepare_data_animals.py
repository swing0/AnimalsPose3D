# 02_prepare_data_animals.py
import argparse
import os
import numpy as np
import sys

# 添加common模块路径
sys.path.append('../')

from common.animals_dataset import AnimalsDataset


def simple_orthographic_project(positions_3d, view_angle='front'):
    """
    简单的正交投影到2D
    view_angle: 'front', 'side', 'top', 'oblique'
    """
    if view_angle == 'front':
        # 前视图: 移除Z轴，只保留X,Y
        return positions_3d[..., :2].copy()
    elif view_angle == 'side':
        # 侧视图: 移除X轴，Z,Y (注意调整坐标顺序)
        return positions_3d[..., [2, 1]].copy()
    elif view_angle == 'top':
        # 顶视图: 移除Y轴，X,Z
        return positions_3d[..., [0, 2]].copy()
    elif view_angle == 'oblique':
        # 斜前视图: 简单的轴测投影
        x = positions_3d[..., 0]
        y = positions_3d[..., 1]
        z = positions_3d[..., 2]
        # 简单的45度斜投影
        u = x - z * 0.35
        v = y - z * 0.35
        return np.stack([u, v], axis=-1)
    else:
        raise ValueError(f"Unknown view angle: {view_angle}")


def normalize_2d_positions(positions_2d):
    """
    归一化2D坐标到 [-1, 1] 范围
    """
    # 找到所有帧的边界
    all_positions = positions_2d.reshape(-1, 2)
    min_val = all_positions.min(axis=0)
    max_val = all_positions.max(axis=0)

    # 计算中心点和范围
    center = (min_val + max_val) / 2
    scale = np.max(max_val - min_val)

    if scale == 0:
        scale = 1.0

    # 归一化到 [-1, 1]
    normalized = (positions_2d - center) / (scale / 2)
    return normalized


def main():
    # 输入输出文件
    input_npz = r'npz\real_npz\data_3d_animals.npz'
    output_2d = r'npz\real_npz\data_2d_animals_gt.npz'

    # 检查输入文件是否存在
    if not os.path.exists(input_npz):
        print(f"错误: 输入文件 {input_npz} 不存在")
        return

    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_2d)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")

    print(f"输入文件: {input_npz}")
    print(f"输出文件: {output_2d}")

    # 创建2D姿态文件
    print('使用简单投影生成2D姿态...')

    try:
        # 加载动物数据集
        dataset = AnimalsDataset(input_npz)
        output_2d_poses = {}

        # 定义四个简单的视角
        view_angles = ['front', 'side', 'oblique', 'top']
        view_names = ['前视图', '侧视图', '斜视图', '顶视图']

        for subject in dataset.subjects():
            output_2d_poses[subject] = {}
            print(f"处理动物: {subject}")

            for action in dataset[subject].keys():
                anim = dataset[subject][action]
                positions_3d = anim['positions']

                print(f"  动作 {action} 的3D数据形状: {positions_3d.shape}")
                print(f"  3D数据范围 - X: [{positions_3d[..., 0].min():.2f}, {positions_3d[..., 0].max():.2f}]")
                print(f"            Y: [{positions_3d[..., 1].min():.2f}, {positions_3d[..., 1].max():.2f}]")
                print(f"            Z: [{positions_3d[..., 2].min():.2f}, {positions_3d[..., 2].max():.2f}]")

                positions_2d_all_views = []

                for i, (view_angle, view_name) in enumerate(zip(view_angles, view_names)):
                    print(f"    生成 {view_name} 投影...")

                    # 简单的正交投影
                    positions_2d = simple_orthographic_project(positions_3d, view_angle)

                    print(f"      原始2D范围 - u: [{positions_2d[..., 0].min():.2f}, {positions_2d[..., 0].max():.2f}]")
                    print(f"                 v: [{positions_2d[..., 1].min():.2f}, {positions_2d[..., 1].max():.2f}]")

                    # 归一化到 [-1, 1] 范围
                    positions_2d_normalized = normalize_2d_positions(positions_2d)

                    print(
                        f"      归一化2D范围 - u: [{positions_2d_normalized[..., 0].min():.2f}, {positions_2d_normalized[..., 0].max():.2f}]")
                    print(
                        f"                   v: [{positions_2d_normalized[..., 1].min():.2f}, {positions_2d_normalized[..., 1].max():.2f}]")

                    positions_2d_all_views.append(positions_2d_normalized.astype('float32'))

                output_2d_poses[subject][action] = positions_2d_all_views
                print(f"  动作 {action}: 生成 {len(positions_2d_all_views)} 个视角")

        # 保存元数据
        metadata = {
            'num_joints': dataset.skeleton().num_joints(),
            'keypoints_symmetry': [
                dataset.skeleton().joints_left(),
                dataset.skeleton().joints_right()
            ],
            'skeleton_parents': dataset.skeleton().parents(),
            'keypoints_name': [
                "Root of Tail", "Left Eye", "Right Eye", "Nose", "Neck",
                "Left Shoulder", "Left Elbow", "Left Front Paw",
                "Right Shoulder", "Right Elbow", "Right Front Paw",
                "Left Hip", "Left Knee", "Left Back Paw",
                "Right Hip", "Right Knee", "Right Back Paw"
            ],
            'view_angles': view_angles,
            'view_names': view_names,
            'projection_type': 'simple_orthographic'
        }

        # 保存2D姿态数据
        print('保存...')
        np.savez_compressed(output_2d, positions_2d=output_2d_poses, metadata=metadata)

        # 验证保存的数据
        print('验证保存的数据...')
        saved_data = np.load(output_2d, allow_pickle=True)
        positions_2d_loaded = saved_data['positions_2d'].item()
        metadata_loaded = saved_data['metadata'].item()

        print(f"保存的动物数量: {len(positions_2d_loaded)}")
        for subject, actions in positions_2d_loaded.items():
            print(f"  {subject}: {len(actions)} 个动作")
            for action, views in actions.items():
                print(f"    {action}: {len(views)} 个视角, 形状: {views[0].shape if len(views) > 0 else 'N/A'}")

        print(f"关节数量: {metadata_loaded['num_joints']}")
        print(f"投影类型: {metadata_loaded['projection_type']}")
        print(f"视角: {metadata_loaded['view_angles']}")

        print('完成!')

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()