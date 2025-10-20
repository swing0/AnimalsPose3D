import argparse
import os
import numpy as np
import sys


# 添加common模块路径
sys.path.append('../')

from common.animals_dataset import AnimalsDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates
from common.utils import wrap


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
    print('计算地面真实的2D姿态...')

    try:
        # 加载动物数据集
        dataset = AnimalsDataset(input_npz)
        output_2d_poses = {}

        for subject in dataset.subjects():
            output_2d_poses[subject] = {}
            print(f"处理动物: {subject}")

            for action in dataset[subject].keys():
                anim = dataset[subject][action]

                positions_2d = []
                for cam in anim['cameras']:
                    # 3D世界坐标 -> 相机坐标
                    pos_3d = world_to_camera(
                        anim['positions'],
                        R=cam['orientation'],
                        t=cam['translation']
                    )
                    # 相机坐标 -> 2D投影
                    pos_2d = wrap(project_to_2d, pos_3d, cam['intrinsic'], unsqueeze=True)
                    # 2D归一化坐标 -> 像素坐标
                    pos_2d_pixel_space = image_coordinates(
                        pos_2d, w=cam['res_w'], h=cam['res_h']
                    )
                    positions_2d.append(pos_2d_pixel_space.astype('float32'))

                output_2d_poses[subject][action] = positions_2d
                print(f"  动作 {action}: {anim['positions'].shape} -> {len(positions_2d)} 个相机视角")

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
            ]
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
            for action, cams in actions.items():
                print(f"    {action}: {len(cams)} 个相机视角, 形状: {cams[0].shape if len(cams) > 0 else 'N/A'}")

        print(f"关节数量: {metadata_loaded['num_joints']}")
        print(f"左关节: {metadata_loaded['keypoints_symmetry'][0]}")
        print(f"右关节: {metadata_loaded['keypoints_symmetry'][1]}")

        print('完成!')

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()