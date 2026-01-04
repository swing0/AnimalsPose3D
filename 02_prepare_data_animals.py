import os
import numpy as np
import sys

# 假设你的目录结构
sys.path.append('../')
from common.animals_dataset import AnimalsDataset

def get_camera_fixed_views(animal_heading_rad, camera_azim_deg, elev_deg=20, dist=8.0):
    """
    针对 Z-up 坐标系，X为正前方设计的相机系统
    
    Args:
        animal_heading_rad: 动物朝向弧度（从Root指向Neck在XY平面的角度）
        camera_azim_deg: 0=前, 90=右, 180=后, 270=左
        elev_deg: 仰角
        dist: 距离
    """
    # 转换角度：
    # 0° 前 -> 相机在 animal_heading + 0°
    # 90° 右 -> 相机在 animal_heading - 90° (顺时针)
    # 180° 后 -> 相机在 animal_heading + 180°
    # 270° 左 -> 相机在 animal_heading + 90°
    
    rel_azim_rad = -np.deg2rad(camera_azim_deg) # 顺时针为正
    total_azim_rad = animal_heading_rad + rel_azim_rad
    elev_rad = np.deg2rad(elev_deg)

    # 计算相机在世界坐标系中的位置 (Z是高度)
    # X = Forward, Y = Side, Z = Up
    cam_x = dist * np.cos(elev_rad) * np.cos(total_azim_rad)
    cam_y = dist * np.cos(elev_rad) * np.sin(total_azim_rad)
    cam_z = dist * np.sin(elev_rad)
    
    cam_pos = np.array([cam_x, cam_y, cam_z])
    target = np.array([0, 0, 0])
    world_up = np.array([0, 0, 1]) # 明确 Z 为向上

    # 构建 LookAt 矩阵 (World to Camera)
    # z_axis: 从相机指向目标 (作为相机坐标系的深度的负方向)
    # 这里我们定义相机坐标系：z为看向物体的方向
    z_axis = target - cam_pos
    z_axis /= np.linalg.norm(z_axis)
    
    # x_axis: 相机坐标系的右方向 (WorldUp cross z_axis)
    x_axis = np.cross(z_axis, world_up)
    x_axis /= np.linalg.norm(x_axis)
    
    # y_axis: 相机坐标系的上方向
    y_axis = np.cross(x_axis, z_axis)
    
    # 旋转矩阵 R
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    
    return R, cam_pos

def project_perspective(pos_3d, R, cam_pos, f=1.0):
    """
    将 3D 点投影到 2D 平面
    pos_3d: (Frames, Joints, 3)
    """
    # 1. 转换到相机坐标系
    rel_pos = pos_3d - cam_pos
    # 批量矩阵相乘: pos_cam = (pos_3d - cam_pos) @ R.T
    pos_cam = np.einsum('nfj,ij->nfi', rel_pos, R)

    # 2. 透视投影
    # 在我们的 LookAt 中，相机看向 +z 方向，所以深度就是 pos_cam[..., 2]
    depth = pos_cam[..., 2]
    depth[np.abs(depth) < 1e-5] = 1e-5 # 避免除零

    u = f * pos_cam[..., 0] / depth
    v = f * pos_cam[..., 1] / depth

    return np.stack([u, v], axis=-1)

def main():
    input_npz = r'npz\real_npz\data_3d_animals.npz'
    output_2d = r'npz\real_npz\data_2d_animals_gt.npz'

    print('正在处理四足动物 3D 投影...')
    dataset = AnimalsDataset(input_npz)
    output_2d_poses = {}
    processed_3d_data = {}

    for subject in dataset.subjects():
        output_2d_poses[subject] = {}
        processed_3d_data[subject] = {}

        for action in dataset[subject].keys():
            pos_3d_raw = dataset[subject][action]['positions']
            
            # Root-Relative (Root 为第0个点)
            root_pos = pos_3d_raw[:, 0:1, :]
            pos_3d_rel = pos_3d_raw - root_pos
            processed_3d_data[subject][action] = pos_3d_rel

            # 计算动物初始朝向 (基于第一帧)
            # Neck(4) - Root(0). 在你的坐标系中，X是前，Y是侧
            v_forward = pos_3d_rel[0, 4, :] - pos_3d_rel[0, 0, :]
            heading_rad = np.arctan2(v_forward[1], v_forward[0]) # y/x

            views_2d = []
            # 定义四个视角
            camera_azimuths = [0, 90, 180, 270] 
            
            for azim in camera_azimuths:
                # 获取该视角的相机参数
                R, cam_pos = get_camera_fixed_views(heading_rad, azim, elev_deg=20, dist=10.0)
                # 投影
                pos_2d = project_perspective(pos_3d_rel, R, cam_pos, f=1.5)

                # 逐帧归一化 (保持在 -1 到 1 之间)
                for f in range(pos_2d.shape[0]):
                    norm_val = np.max(np.abs(pos_2d[f]))
                    if norm_val > 0:
                        pos_2d[f] /= norm_val

                views_2d.append(pos_2d.astype('float32'))

            output_2d_poses[subject][action] = views_2d

    # 保存
    metadata = {
        'num_joints': 17,
        'keypoints_name': ["Root", "LEye", "REye", "Nose", "Neck", "LShl", "LElb", "LPaw",
                           "RShl", "RElb", "RPaw", "LHip", "LKne", "LBPa", "RHip", "RKne", "RBPa"],
        'projection': 'custom_quadruped_v1'
    }
    np.savez_compressed(output_2d, positions_2d=output_2d_poses, metadata=metadata)
    np.savez_compressed(input_npz, positions_3d=processed_3d_data)
    print("完成！请运行可视化脚本检查四个视角是否分别为：正面、右侧、背面、左侧。")

if __name__ == '__main__':
    main()