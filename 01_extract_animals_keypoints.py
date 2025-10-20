# 01_extract_animals_keypoints.py
import os
import json
import zipfile
import glob
import numpy as np
from pathlib import Path

# 关键点映射关系
KEYPOINT_MAPPING = {
    "Root of Tail": ["def_c_tail1_joint"],
    "Left Eye": ["def_eye_joint.L"],
    "Right Eye": ["def_eye_joint.R"],
    "Nose": ["def_c_nose_joint"],
    "Neck": ["def_c_neck1_joint", "def_c_neck2_joint"],
    "Left Shoulder": ["def_clavicle_joint.L"],
    "Left Elbow": ["def_frontLegLwr_joint.L"],
    "Left Front Paw": ["def_frontFoot_joint.L"],
    "Right Shoulder": ["def_clavicle_joint.R"],
    "Right Elbow": ["def_frontLegLwr_joint.R"],
    "Right Front Paw": ["def_frontFoot_joint.R"],
    "Left Hip": ["def_rearLegUpr_joint.L"],
    "Left Knee": ["def_rearLegLwr_joint.L"],
    "Left Back Paw": ["def_rearFoot_joint.L"],
    "Right Hip": ["def_rearLegUpr_joint.R"],
    "Right Knee": ["def_rearLegLwr_joint.R"],
    "Right Back Paw": ["def_rearFoot_joint.R"]
}

KEYPOINT_ORDER = list(KEYPOINT_MAPPING.keys())

# 配置开关 - 高效模式：只生成整合的NPZ，适合机器学习
GENERATE_CSV = False
COMBINE_NPZ = True


def extract_keypoints_from_frame(frame_data):
    """
    从单个帧数据中提取关键点坐标
    """
    keypoints = {}

    for keypoint_name, joint_names in KEYPOINT_MAPPING.items():
        coordinates = None

        for joint_name in joint_names:
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


def process_json_file(json_data, zip_name, json_file_name):
    """
    处理单个JSON文件中的所有帧
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
        np_data = np.array(frames_np_data, dtype=np.float32)
        np_data = np_data.reshape(len(frames_np_data), len(KEYPOINT_ORDER), 3)
    else:
        np_data = np.array([], dtype=np.float32)

    return np_data


def process_zip_file(zip_path, output_dir):
    """
    处理单个ZIP文件中的所有JSON文件
    """
    zip_name = Path(zip_path).stem
    all_np_data = []

    print(f"处理ZIP文件: {zip_name}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            json_files = [f for f in zip_ref.namelist() if f.lower().endswith('.json')]

            for json_file in json_files:
                try:
                    with zip_ref.open(json_file) as f:
                        json_data = json.load(f)

                    np_data = process_json_file(json_data, zip_name, json_file)

                    if len(np_data) > 0:
                        if len(all_np_data) == 0:
                            all_np_data = np_data
                        else:
                            all_np_data = np.concatenate([all_np_data, np_data], axis=0)

                    print(f"  处理 {Path(json_file).stem}: NPZ数据: {len(np_data)} 有效帧")

                except Exception as e:
                    print(f"  处理文件 {json_file} 时出错: {e}")
                    continue

        return all_np_data, zip_name, len(all_np_data)

    except Exception as e:
        print(f"处理ZIP文件 {zip_path} 时出错: {e}")
        return None, zip_name, 0


def main():
    # 设置路径
    input_dir = r"D:\workplace\export_json_test"
    output_dir = r"npz\real_npz"

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有ZIP文件
    zip_files = glob.glob(os.path.join(input_dir, "*.zip"))

    if not zip_files:
        print(f"在 {input_dir} 中没有找到ZIP文件")
        return

    print(f"找到 {len(zip_files)} 个ZIP文件")
    print("模式: 高效模式 (只生成整合的NPZ文件)")

    # 处理所有ZIP文件并收集数据用于整合
    total_valid_frames = 0
    successful_zips = 0
    all_np_data_collection = []
    zip_names_collection = []

    for zip_file in zip_files:
        np_data, zip_name, valid_frames = process_zip_file(zip_file, output_dir)
        if valid_frames > 0:
            successful_zips += 1
            if np_data is not None:
                all_np_data_collection.append(np_data)
                zip_names_collection.append(zip_name)
                total_valid_frames += len(np_data)

    # 保存整合的NPZ文件
    if all_np_data_collection:
        combined_keypoints = np.concatenate(all_np_data_collection, axis=0)

        # 创建文件信息列表
        file_info = []
        for i, np_data in enumerate(all_np_data_collection):
            file_info.extend([zip_names_collection[i]] * len(np_data))

        output_npz = os.path.join(output_dir, "animals_keypoint.npz")
        np.savez(
            output_npz,
            keypoints=combined_keypoints,
            keypoint_names=KEYPOINT_ORDER,
            file_info=file_info,
            total_frames=len(combined_keypoints),
            frame_indices=np.arange(len(combined_keypoints)),
            source_zips=zip_names_collection
        )
        print(f"\n已保存整合NPZ: {output_npz}")
        print(f"整合数据形状: {combined_keypoints.shape}")

    print(f"\n处理完成!")
    print(f"成功处理 {successful_zips}/{len(zip_files)} 个ZIP文件")
    print(f"总共 {total_valid_frames} 帧包含完整关键点数据")
    print(f"输出文件保存在: {output_dir}")


if __name__ == "__main__":
    main()