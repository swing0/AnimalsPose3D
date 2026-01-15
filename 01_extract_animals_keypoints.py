# 01_extract_animals_keypoints.py
import os
import json
import zipfile
import glob
import numpy as np
from pathlib import Path

KEYPOINT_MAPPING = {
    "Root of Tail": ["def_c_tail1_joint"],
    "Left Eye": ["def_eye_joint.L"],
    "Right Eye": ["def_eye_joint.R"],
    "Nose": ["def_c_nose_joint"],
    "Neck": ["def_c_neck1_joint", "def_c_neck2_joint"],
    "Left Shoulder": ["def_frontLegLwrHalfTwist_joint.L"],
    "Left Elbow": ["def_frontHorselinkHalfTwist_joint.L"],
    "Left Front Paw": ["def_frontFoot_joint.L"],
    "Right Shoulder": ["def_frontLegLwrHalfTwist_joint.R"],
    "Right Elbow": ["def_frontHorselinkHalfTwist_joint.R"],
    "Right Front Paw": ["def_frontFoot_joint.R"],
    "Left Hip": ["def_rearLegLwrHalfTwist_joint.L"],
    "Left Knee": ["def_rearHorselink_joint.L"],
    "Left Back Paw": ["def_rearFoot_joint.L"],
    "Right Hip": ["def_rearLegLwrHalfTwist_joint.R"],
    "Right Knee": ["def_rearHorselink_joint.R"],
    "Right Back Paw": ["def_rearFoot_joint.R"]
}



KEYPOINT_MAPPING2 = {
    "Root of Tail": ["def_c_tail1_joint"],
    "Left Eye": ["def_eye_joint.L"],
    "Right Eye": ["def_eye_joint.R"],
    "Nose": ["def_c_nose_joint"],
    "Neck": ["def_c_neck1_joint", "def_c_neck2_joint"],
    "Left Shoulder": ["def_frontLegUprHalfTwist_joint.L"],
    "Left Elbow": ["def_frontLegLwrHalfTwist_joint.L"],
    "Left Front Paw": ["def_frontFoot_joint.L"],
    "Right Shoulder": ["def_frontLegUprHalfTwist_joint.R"],
    "Right Elbow": ["def_frontLegLwrHalfTwist_joint.R"],
    "Right Front Paw": ["def_frontFoot_joint.R"],
    "Left Hip": ["def_rearLegUprHalfTwist_joint.L"],
    "Left Knee": ["def_rearLegLwrHalfTwist_joint.L"],
    "Left Back Paw": ["def_rearFoot_joint.L"],
    "Right Hip": ["def_rearLegUprHalfTwist_joint.R"],
    "Right Knee": ["def_rearLegLwrHalfTwist_joint.R"],
    "Right Back Paw": ["def_rearFoot_joint.R"]
}


KEYPOINT_ORDER = list(KEYPOINT_MAPPING.keys())


# 过滤阈值
MIN_FRAMES = 60  # 最少帧数要求


def extract_keypoints_from_frame(frame_data):
    """
    从单个帧数据中提取关键点坐标
    如果KEYPOINT_MAPPING没有匹配上，使用KEYPOINT_MAPPING2作为备用方案
    """
    keypoints = {}

    for keypoint_name, joint_names in KEYPOINT_MAPPING.items():
        coordinates = None

        # 首先尝试KEYPOINT_MAPPING
        for joint_name in joint_names:
            if joint_name in frame_data:
                coordinates = frame_data[joint_name]
                break

        # 如果KEYPOINT_MAPPING没有匹配上，尝试KEYPOINT_MAPPING2
        if coordinates is None and keypoint_name in KEYPOINT_MAPPING2:
            joint_names2 = KEYPOINT_MAPPING2[keypoint_name]
            for joint_name in joint_names2:
                if joint_name in frame_data:
                    coordinates = frame_data[joint_name]
                    break

        if coordinates and len(coordinates) == 3:
            keypoints[keypoint_name] = {
                'x': coordinates[0],
                'y': coordinates[1],
                'z': coordinates[2]
            }
        else:
            keypoints[keypoint_name] = {'x': None, 'y': None, 'z': None}

    return keypoints


def extract_animal_name(zip_name):
    """
    从zip文件名中提取动物名称
    例如: Addax_Female.ovl.zip -> Addax_Female
    """
    if '.ovl.zip' in zip_name:
        return zip_name.replace('.ovl.zip', '')
    elif '.zip' in zip_name:
        return zip_name.replace('.zip', '')
    elif '.ovl' in zip_name:
        return zip_name.replace('.ovl', '')
    else:
        return zip_name


def extract_animation_name(json_file_name):
    """
    从JSON文件名中提取动画名称
    例如: addax_female__animationmotionextractedfighting.manisetadb82aa_addax_female_fightchaseoff_keypoints.json -> fightchaseoff
    """
    # 获取文件名（不含路径和扩展名）
    file_stem = Path(json_file_name).stem

    # 分割文件名，通常格式为: prefix_animalname_animationname_keypoints
    parts = file_stem.split('_')

    # 查找包含动物名的部分，动画名称通常在动物名之后
    if len(parts) >= 3:
        # 从后往前找，动画名称通常在倒数第二个位置
        animation_name = parts[-2] if parts[-1] == 'keypoints' else parts[-1]
        return animation_name
    else:
        # 如果分割失败，返回整个文件名（去掉_keypoints后缀）
        return file_stem.replace('_keypoints', '')


def process_json_file(json_data, zip_name, json_file_name):
    """
    处理单个JSON文件中的所有帧，返回动画名称和对应的关键点数据
    """
    frames_np_data = []

    for frame_key, frame_data in json_data.items():
        if frame_key.isdigit():
            keypoints = extract_keypoints_from_frame(frame_data)

            frame_coords = []
            valid_frame = True

            for kp_name in KEYPOINT_ORDER:
                coords = keypoints.get(kp_name, {'x': None, 'y': None, 'z': None})

                # 检查坐标是否有效
                if coords['x'] is not None and coords['y'] is not None and coords['z'] is not None:
                    frame_coords.extend([coords['x'], coords['y'], coords['z']])
                else:
                    valid_frame = False
                    frame_coords.extend([0.0, 0.0, 0.0])

            if valid_frame:
                frames_np_data.append(frame_coords)

    if frames_np_data:
        # 1. 形状转换为 (Frames, Joints, 3)
        np_data = np.array(frames_np_data, dtype=np.float32).reshape(-1, len(KEYPOINT_ORDER), 3)

        # 2. 坐标系转换: 修正动物朝向 (X, Y, Z) -> (X, Y, Z)
        # 正确的映射逻辑：原坐标系是Y-up，保持不变
        standard_3d = np_data.copy()  # 直接使用原始数据，不进行坐标轴交换

        # 3. Root-Relative 中心化
        # 假设 KEYPOINT_ORDER[0] 是 "Root of Tail" (根节点)
        root_pos = standard_3d[:, 0:1, :]
        standard_3d = standard_3d - root_pos

        animation_name = extract_animation_name(json_file_name)
        return animation_name, standard_3d
    else:
        return None, np.array([], dtype=np.float32)


def process_zip_file(zip_path):
    """
    处理单个ZIP文件中的所有JSON文件，返回动物名称和动画数据字典
    """
    zip_name = Path(zip_path).stem
    animal_name = extract_animal_name(zip_name)
    animations_data = {}

    print(f"处理ZIP文件: {zip_name} -> 动物: {animal_name}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            json_files = [f for f in zip_ref.namelist() if f.lower().endswith('.json')]

            filtered_count = 0
            for json_file in json_files:
                try:
                    with zip_ref.open(json_file) as f:
                        json_data = json.load(f)

                    animation_name, np_data = process_json_file(json_data, zip_name, json_file)

                    if animation_name and len(np_data) >= MIN_FRAMES:
                        if animation_name in animations_data:
                            print(f"  跳过动画 {animation_name}: 已存在，不重复保留")
                        else:
                            animations_data[animation_name] = np_data
                            print(f"  保留动画 {animation_name}: {len(np_data)} 有效帧")
                    elif animation_name:
                        filtered_count += 1

                except Exception as e:
                    print(f"  处理文件 {json_file} 时出错: {e}")
                    continue

            if filtered_count > 0:
                print(f"  过滤掉 {filtered_count} 个短动画 (< {MIN_FRAMES} 帧)")

        return animal_name, animations_data

    except Exception as e:
        print(f"处理ZIP文件 {zip_path} 时出错: {e}")
        return animal_name, {}


def main():
    # 设置路径
    input_dir = r"D:\workplace\export_json_20"
    output_dir = r"npz\real_npz"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有ZIP文件
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))

    if not zip_files:
        print(f"在 {input_dir} 中没有找到ZIP文件")
        return

    print(f"找到 {len(zip_files)} 个ZIP文件")
    print(f"过滤阈值: 只保留 {MIN_FRAMES} 帧以上的动画")

    # 处理所有ZIP文件并收集数据
    positions_3d = {}
    successful_zips = 0
    total_animations = 0
    total_frames = 0
    filtered_animations = 0
    filtered_zips = 0

    for zip_file in zip_files:
        animal_name, animations_data = process_zip_file(zip_file)

        if animations_data:
            positions_3d[animal_name] = animations_data
            successful_zips += 1
            total_animations += len(animations_data)
            for anim_data in animations_data.values():
                total_frames += len(anim_data)

            print(f"  动物 {animal_name}: {len(animations_data)} 个动画")
        else:
            filtered_zips += 1
            print(f"  动物 {animal_name}: 没有满足条件的动画")

    # 统计被过滤的动画数量
    for zip_file in zip_files:
        animal_name = extract_animal_name(Path(zip_file).stem)
        if animal_name not in positions_3d:
            filtered_animations += 1

    # 保存为NPZ文件
    if positions_3d:
        output_npz = os.path.join(output_dir, "data_3d_animals.npz")

        # 创建最终的字典结构
        npz_data = {
            'positions_3d': positions_3d
        }

        # 保存NPZ文件
        np.savez(output_npz, **npz_data)

        print(f"\n已保存结构化NPZ: {output_npz}")
        print(f"数据结构:")
        print(f"  包含 {len(positions_3d)} 种动物")
        print(f"  总共 {total_animations} 个动画")
        print(f"  总共 {total_frames} 帧数据")

        # 打印详细结构
        print(f"\n详细结构:")
        for animal, anims in positions_3d.items():
            print(f"  {animal}: {len(anims)} 个动画")
            for anim_name, anim_data in anims.items():
                print(f"    - {anim_name}: {anim_data.shape}")

        # 过滤统计
        print(f"\n过滤统计:")
        print(f"  成功处理的ZIP文件: {successful_zips}/{len(zip_files)}")
        print(f"  被过滤的ZIP文件: {filtered_zips}")
        print(f"  保留的动画序列: {total_animations}")
        print(f"  过滤掉的短动画: {filtered_animations}")

    else:
        print(f"\n警告: 没有找到任何满足条件的数据（所有动画都少于 {MIN_FRAMES} 帧）")

    print(f"\n处理完成!")
    print(f"输出文件保存在: {output_dir}")


if __name__ == "__main__":
    main()