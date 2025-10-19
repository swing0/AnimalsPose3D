# 02_prepare_data_animals.py
import os
import numpy as np
import torch

from common.animals_dataset import QuadrupedAnimalDataset
from common.camera import world_to_camera, project_to_2d, image_coordinates


def generate_2d_poses_from_3d():
    """模仿H36M的流程，从3D关键点生成2D投影"""

    # 输入输出文件
    input_npz = r'npz\real_npz\animals_keypoint.npz'
    output_3d = r'npz\real_npz\data_3d_animals.npz'
    output_2d = r'npz\real_npz\data_2d_animals_gt.npz'

    if not os.path.exists(input_npz):
        print(f"Error: Input file {input_npz} not found!")
        print(f"Current directory: {os.getcwd()}")
        return

    print('Loading and reorganizing 3D animal data...')

    try:
        dataset = QuadrupedAnimalDataset(input_npz)
        print(f"Loaded dataset with subjects: {dataset.subjects()}")

        # 保存3D数据 - 修复这里
        positions_3d = {}
        for subject in dataset.subjects():
            positions_3d[subject] = {}
            for action in dataset[subject].keys():
                # 确保获取的是numpy数组而不是其他对象
                positions_data = dataset[subject][action]['positions']
                if isinstance(positions_data, np.ndarray):
                    positions_3d[subject][action] = positions_data
                else:
                    # 如果是其他类型，尝试转换
                    positions_3d[subject][action] = np.array(positions_data)
                print(f"  {subject}/{action}: {positions_3d[subject][action].shape}")

        # 使用allow_pickle=True保存，因为数据结构是嵌套字典
        np.savez_compressed(output_3d, positions_3d=positions_3d)
        print(f'Saved 3D data to {output_3d}')

        # 生成2D投影
        print('Computing ground-truth 2D poses...')
        output_2d_poses = {}

        for subject in dataset.subjects():
            output_2d_poses[subject] = {}

            for action in dataset[subject].keys():
                anim = dataset[subject][action]
                positions_3d_world = anim['positions']
                cameras = anim['cameras']

                positions_2d = []
                for cam_idx, cam in enumerate(cameras):
                    print(f"  Processing {subject}/{action} camera {cam_idx}...")

                    # 确保数据格式正确
                    R = cam['orientation']
                    t = cam['translation']
                    intrinsic = cam['intrinsic']

                    # 转换为torch tensor用于project_to_2d
                    intrinsic_tensor = torch.from_numpy(intrinsic).unsqueeze(0)

                    # 世界坐标到相机坐标
                    pos_3d_camera = world_to_camera(
                        positions_3d_world,
                        R=R,
                        t=t
                    )

                    # 投影到2D - 需要转换为torch tensor
                    pos_3d_camera_tensor = torch.from_numpy(pos_3d_camera).float()

                    # 扩展intrinsic参数以匹配批次大小
                    batch_size = pos_3d_camera_tensor.shape[0]
                    intrinsic_batch = intrinsic_tensor.repeat(batch_size, 1)

                    pos_2d = project_to_2d(pos_3d_camera_tensor, intrinsic_batch)
                    pos_2d_np = pos_2d.numpy()

                    # 转换到像素坐标
                    pos_2d_pixel_space = image_coordinates(
                        pos_2d_np,
                        w=cam['res_w'],
                        h=cam['res_h']
                    )
                    positions_2d.append(pos_2d_pixel_space.astype('float32'))

                output_2d_poses[subject][action] = positions_2d

        # 保存2D数据
        metadata = {
            'num_joints': dataset.skeleton().num_joints(),
            'keypoints_symmetry': [
                dataset.skeleton().joints_left(),
                dataset.skeleton().joints_right()
            ]
        }

        np.savez_compressed(output_2d, positions_2d=output_2d_poses, metadata=metadata)
        print(f'Saved 2D data to {output_2d}')
        print('Done!')

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    generate_2d_poses_from_3d()