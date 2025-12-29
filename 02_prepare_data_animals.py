import os
import numpy as np
import sys
from tqdm import tqdm

# 假设你的目录结构，确保能导入 AnimalsDataset
sys.path.append('../')
# 注意：这里保留你的 AnimalsDataset 导入，但在处理逻辑中加入了坐标修正
from common.animals_dataset import AnimalsDataset


def get_random_camera(dist_range=(4.0, 10.0), elev_range=(0, 45), azim_range=(0, 360)):
    """在球面上随机采样一个相机位置并计算 LookAt 矩阵"""
    dist = np.random.uniform(*dist_range)
    elev = np.deg2rad(np.random.uniform(*elev_range))
    azim = np.deg2rad(np.random.uniform(*azim_range))

    # 相机在世界坐标系的位置 (Y-up)
    cam_x = dist * np.cos(elev) * np.sin(azim)
    cam_y = dist * np.sin(elev)
    cam_z = dist * np.cos(elev) * np.cos(azim)
    cam_pos = np.array([cam_x, cam_y, cam_z])

    target = np.array([0, 0, 0])  # 始终注视原点
    up = np.array([0, 1, 0])  # Y方向向上

    # 计算 LookAt 旋转矩阵
    z_axis = cam_pos - target
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3, 3) 外参旋转
    return R, cam_pos


def project_perspective(pos_3d, R, cam_pos, f=1.0):
    """透视投影：将世界坐标转换为2D归一化平面"""
    # 1. 转换到相机坐标系: P_cam = R * (P_world - cam_pos)
    rel_pos = pos_3d - cam_pos
    pos_cam = np.dot(rel_pos, R.T)

    # 2. 透视公式: u = f * x / z; v = f * y / z
    # 注意：在相机坐标系中，向前是 -z 轴，所以深度为 -pos_cam[..., 2]
    depth = -pos_cam[..., 2]
    depth[depth < 0.1] = 0.1  # 防止除以0

    u = (f * pos_cam[..., 0]) / depth
    v = (f * pos_cam[..., 1]) / depth

    return np.stack([u, v], axis=-1)


def main():
    input_npz = r'npz\real_npz\data_3d_animals.npz'
    output_2d = r'npz\real_npz\data_2d_animals_gt.npz'
    NUM_VIEWS_PER_ANIM = 4  # 每个动画序列生成多少个随机视角

    if not os.path.exists(input_npz):
        print(f"Error: {input_npz} not found")
        return

    print('加载 3D 数据并重新处理坐标系...')
    dataset = AnimalsDataset(input_npz)
    output_2d_poses = {}
    processed_3d_data = {}  # 存储修正后的 3D 坐标

    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        processed_3d_data[subject] = {}

        for action in dataset[subject].keys():
            # 原始数据 (Frames, Joints, 3)
            pos_3d_raw = dataset[subject][action]['positions']

            # --- [1] 坐标处理 ---
            # 修正：step 01 已经将坐标系转换为 视觉 Y-up (X, Y, Z)
            # 其中 Y 是高度，Z 是深度 (-Z 向前)
            # 所以这里不需要再次进行轴交换，直接使用即可
            pos_3d_yup = pos_3d_raw.copy()

            # --- [2] Root-Relative 中心化 ---
            # 假设第0个关键点是 "Root of Tail"
            # 注意：在 01 中其实已经做过一次 Root-Relative，但为了保险起见（或针对多视图增强），
            # 这里基于当前序列的第一帧或每一帧重新计算中心化是合理的。
            # 通常我们每一帧都减去根节点位置，或者减去第一帧根节点。
            # 这里保持每一帧都减去根节点，使得根节点永远在原点 (0,0,0)
            root_pos = pos_3d_yup[:, 0:1, :]
            pos_3d_rel = pos_3d_yup - root_pos

            processed_3d_data[subject][action] = pos_3d_rel

            # --- [3] 生成多个随机透视视角 ---
            views_2d = []
            for _ in range(NUM_VIEWS_PER_ANIM):
                R, cam_pos = get_random_camera()
                pos_2d = project_perspective(pos_3d_rel, R, cam_pos)

                # 逐帧归一化到 [-1, 1] (研究常用做法)
                for f in range(pos_2d.shape[0]):
                    frame = pos_2d[f]
                    max_range = np.max(np.abs(frame))
                    if max_range > 0:
                        pos_2d[f] = frame / max_range

                views_2d.append(pos_2d.astype('float32'))

            output_2d_poses[subject][action] = views_2d

    # 保存元数据与更新后的 3D 数据
    metadata = {
        'num_joints': 17,
        'keypoints_name': ["Root of Tail", "Left Eye", "Right Eye", "Nose", "Neck", "L-Shld", "L-Elbw", "L-Paw",
                           "R-Shld", "R-Elbw", "R-Paw", "L-Hip", "L-Knee", "L-BackPaw", "R-Hip", "R-Knee", "R-BackPaw"],
        'projection': 'perspective_random_spherical'
    }

    # 注意：我们同时保存修正后的 3D 坐标，这才是训练 3D 提升模型的目标值
    np.savez_compressed(output_2d, positions_2d=output_2d_poses, metadata=metadata)
    np.savez_compressed(input_npz, positions_3d=processed_3d_data)  # 覆写或另存修正后的 3D
    print(f'成功生成数据。每个动作包含 {NUM_VIEWS_PER_ANIM} 个随机透视视角。')


if __name__ == '__main__':
    main()