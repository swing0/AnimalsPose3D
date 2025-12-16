# 05_pose3d_to_bvh.py
import os
import os.path as osp
import sys
import numpy as np

from common.animal_skeleton import AnimalSkeleton
from common.camera import camera_to_world
from common.config import get_config

# 添加必要的路径
sys.path.append(osp.join(osp.abspath(osp.dirname(__file__))))

# 选择相应的subject(动物)
subject_num = 0
# action_num = 0


def generate_bvh_for_all_actions(npz_path, subject_num=0):
    """
    为指定subject的所有action生成bvh文件
    """
    # 加载配置
    cfg = get_config(osp.join(r'common/pose_config_with_bvh.yaml'))

    # 加载npz文件
    print(f'>>> Loading 3D keypoints from {npz_path}')
    data = np.load(npz_path, allow_pickle=True)

    # 检查数据结构
    if 'positions_3d' not in data.files:
        raise ValueError(f"在 {npz_path} 中找不到 'positions_3d' 键。可用的键: {list(data.files)}")

    positions_3d_dict = data['positions_3d'].item()

    # 获取所有subject
    subjects = list(positions_3d_dict.keys())
    if subject_num >= len(subjects):
        raise ValueError(f"subject_num {subject_num} 超出范围。共有 {len(subjects)} 个subject")

    selected_subject = subjects[subject_num]
    print(f'>>> 选择的subject: {selected_subject}')

    # 获取该subject的所有actions
    actions_dict = positions_3d_dict[selected_subject]
    actions = list(actions_dict.keys())
    print(f'>>> 找到 {len(actions)} 个actions: {actions}')

    # 为每个action生成BVH文件
    generated_files = []

    for action_name in actions:
        print(f'\n>>> 处理action: {action_name}')

        # 获取该action的关键点数据
        camera_keypoints_3d = actions_dict[action_name]
        print(f'>>> 数据形状: {camera_keypoints_3d.shape}')
        print(f'>>> 总帧数: {len(camera_keypoints_3d)}')

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

        # 使用subject和action名称作为文件名
        safe_subject = selected_subject.replace(' ', '_').replace('/', '_')
        safe_action = action_name.replace(' ', '_').replace('/', '_')
        bvh_filename = f"{safe_subject}_{safe_action}.bvh"
        save_path = osp.join(output_dir, bvh_filename)

        # 生成BVH
        skeleton = AnimalSkeleton()
        skeleton.poses2bvh(world_keypoints_3d_h36m, output_file=save_path)

        print(f'>>> 成功生成BVH文件: {save_path}')
        print(f'>>> BVH总帧数: {len(world_keypoints_3d_h36m)}')
        generated_files.append(save_path)

    return generated_files


def check_npz_structure(npz_path):
    """
    检查NPZ文件结构，帮助选择subject
    """
    print(f"检查NPZ文件结构: {npz_path}")
    print("=" * 60)

    data = np.load(npz_path, allow_pickle=True)

    if 'positions_3d' in data.files:
        positions_3d_dict = data['positions_3d'].item()

        print("NPZ文件结构:")
        print("=" * 60)

        for i, (subject, actions) in enumerate(positions_3d_dict.items()):
            print(f"Subject {i}: {subject}")
            for action_name, action_data in actions.items():
                if isinstance(action_data, np.ndarray):
                    print(f"  └── {action_name}: {action_data.shape} (frames: {len(action_data)})")
                else:
                    print(f"  └── {action_name}: {type(action_data)}")

        print("=" * 60)
        return positions_3d_dict
    else:
        print("错误: 未找到 'positions_3d' 键")
        return None


if __name__ == '__main__':
    # 直接指定npz文件路径
    npz_file = 'npz/real_npz/data_3d_animals.npz'
    # npz_file = 'npz/estimate_npz/data_3d_estimated.npz'

    try:
        # 首先检查文件结构
        structure = check_npz_structure(npz_file)

        if structure:
            # 显示可用的subject
            subjects = list(structure.keys())
            print(f"\n可用的subject (0-{len(subjects) - 1}):")
            for i, subject in enumerate(subjects):
                actions = list(structure[subject].keys())
                print(f"  {i}: {subject} (包含 {len(actions)} 个actions)")

            # 使用预设的subject_num
            print(f"\n使用subject_num = {subject_num}: {subjects[subject_num]}")

            # 生成所有actions的BVH文件
            bvh_files = generate_bvh_for_all_actions(npz_file, subject_num)

            print(f"\n=== 生成完成 ===")
            print(f"共生成 {len(bvh_files)} 个BVH文件:")
            for bvh_file in bvh_files:
                print(f"  ✓ {bvh_file}")

    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()