# 06_pose3d_to_bvh.py
import os
import os.path as osp
import sys
import numpy as np

from common.animal_skeleton import AnimalSkeleton
from common.camera import camera_to_world
from common.config import get_config

# 添加必要的路径
sys.path.append(osp.join(osp.abspath(osp.dirname(__file__))))


def generate_bvh_simple(npz_path):
    """
    简化版本：直接从npz文件生成bvh文件
    """
    # 加载配置
    cfg = get_config(osp.join(r'common/pose_config_with_bvh.yaml'))

    # 加载npz文件
    print(f'>>> Loading 3D keypoints from {npz_path}')
    data = np.load(npz_path, allow_pickle=True)

    # 对于data_3d_animals.npz，数据结构是嵌套字典
    if 'positions_3d' in data.files:
        positions_3d_dict = data['positions_3d'].item()  # 使用.item()获取字典

        # 选择第一个subject和action的数据
        first_subject = list(positions_3d_dict.keys())[0]
        first_action = list(positions_3d_dict[first_subject].keys())[0]

        camera_keypoints_3d = positions_3d_dict[first_subject][first_action]
        print(f'>>> Using data from {first_subject}/{first_action}, shape: {camera_keypoints_3d.shape}')
        print(f'>>> Total frames: {len(camera_keypoints_3d)}')

    else:
        # 如果不是标准格式，尝试直接加载关键点
        camera_keypoints_3d = None
        for key in data.files:
            array = data[key]
            if hasattr(array, 'shape') and array.ndim == 3 and array.shape[-1] == 3:
                camera_keypoints_3d = array
                print(f'>>> Using 3D keypoints with key: "{key}", shape: {camera_keypoints_3d.shape}')
                break

        if camera_keypoints_3d is None:
            raise ValueError(f"在 {npz_path} 中找不到合适的3D关键点数据。可用的键: {list(data.files)}")

    print(f'>>> Loaded 3D keypoints with shape: {camera_keypoints_3d.shape}')
    print(f'>>> Data type: {camera_keypoints_3d.dtype}')

    # 确保数据类型正确
    if camera_keypoints_3d.dtype == np.object_:
        print("警告: 数据是object类型，尝试转换...")
        camera_keypoints_3d = np.array(camera_keypoints_3d, dtype=np.float32)

    # 坐标系转换
    bvh_cfg = cfg.bvh
    world_keypoints_3d = camera_to_world(camera_keypoints_3d,
                                         R=np.array(bvh_cfg.camera.extrinsics.R, dtype=np.float32),
                                         t=0)

    # 转换为h36m格式
    world_keypoints_3d_h36m = np.zeros_like(world_keypoints_3d)
    X = world_keypoints_3d[..., 0] * bvh_cfg.scale_factor
    Y = world_keypoints_3d[..., 1] * bvh_cfg.scale_factor
    Z = world_keypoints_3d[..., 2] * bvh_cfg.scale_factor

    world_keypoints_3d_h36m[..., 0] = -X
    world_keypoints_3d_h36m[..., 1] = Z
    world_keypoints_3d_h36m[..., 2] = Y

    # 生成bvh文件
    output_dir = 'bvh'
    os.makedirs(output_dir, exist_ok=True)
    npz_basename = osp.basename(npz_path).split('.')[0]
    save_path = osp.join(output_dir, npz_basename + '.bvh')

    skeleton = AnimalSkeleton()
    skeleton.poses2bvh(world_keypoints_3d_h36m, output_file=save_path)

    print(f'>>> Successfully generated BVH file: {save_path}')
    print(f'>>> Total frames in BVH: {len(world_keypoints_3d_h36m)}')
    return save_path


if __name__ == '__main__':
    # 直接指定npz文件路径
    # npz_file = 'npz/real_npz/animals_keypoint.npz'
    npz_file = 'npz/real_npz/data_3d_animals.npz'
    # npz_file = 'npz/estimate_npz/data_3d_animals.npz'

    try:
        bvh_path = generate_bvh_simple(npz_file)
        print(f"BVH文件已生成: {bvh_path}")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()