# 01_extract_animals_keypoints.py
import os
import json
import zipfile
import glob
import pandas as pd
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

# 配置开关
GENERATE_CSV = False  # 设置为False则不生成CSV文件
COMBINE_NPZ = True  # 设置为False则不整合NPZ文件


# GENERATE_CSV = True, COMBINE_NPZ = True：完整模式，生成CSV和整合的NPZ
# GENERATE_CSV = False, COMBINE_NPZ = True：高效模式，只生成整合的NPZ，适合机器学习
# GENERATE_CSV = True, COMBINE_NPZ = False：分析模式，生成CSV和单独的NPZ，便于数据检查
# GENERATE_CSV = False, COMBINE_NPZ = False：最小模式，只生成单独的NPZ文件

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
    frames_data = []
    frames_np_data = []

    for frame_key, frame_data in json_data.items():
        if frame_key.isdigit():
            keypoints = extract_keypoints_from_frame(frame_data)

            # 只有在需要生成CSV时才构建frame_record
            frame_record = {}
            if GENERATE_CSV:
                frame_record = {
                    'zip_file': zip_name,
                    'json_file': Path(json_file_name).stem,
                    'frame_id': int(frame_key),
                    'source_file': json_file_name
                }

            frame_coords = []
            valid_frame = True

            for kp_name in KEYPOINT_ORDER:
                coords = keypoints.get(kp_name, {'x': None, 'y': None, 'z': None})

                # 只有在需要生成CSV时才添加到frame_record
                if GENERATE_CSV:
                    frame_record[f'{kp_name}_x'] = coords['x']
                    frame_record[f'{kp_name}_y'] = coords['y']
                    frame_record[f'{kp_name}_z'] = coords['z']

                # 检查坐标是否有效
                if coords['x'] is not None and coords['y'] is not None and coords['z'] is not None:
                    frame_coords.extend([coords['x'], coords['y'], coords['z']])
                else:
                    valid_frame = False
                    frame_coords.extend([0.0, 0.0, 0.0])

            if GENERATE_CSV:
                frames_data.append(frame_record)

            if valid_frame:
                frames_np_data.append(frame_coords)

    if frames_np_data:
        np_data = np.array(frames_np_data, dtype=np.float32)
        np_data = np_data.reshape(len(frames_np_data), len(KEYPOINT_ORDER), 3)
    else:
        np_data = np.array([], dtype=np.float32)

    return frames_data, np_data


def process_zip_file(zip_path, output_dir):
    """
    处理单个ZIP文件中的所有JSON文件
    """
    zip_name = Path(zip_path).stem
    all_frames_data = []
    all_np_data = []

    print(f"处理ZIP文件: {zip_name}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            json_files = [f for f in zip_ref.namelist() if f.lower().endswith('.json')]

            for json_file in json_files:
                try:
                    with zip_ref.open(json_file) as f:
                        json_data = json.load(f)

                    frames_data, np_data = process_json_file(json_data, zip_name, json_file)

                    if GENERATE_CSV:
                        all_frames_data.extend(frames_data)

                    if len(np_data) > 0:
                        if len(all_np_data) == 0:
                            all_np_data = np_data
                        else:
                            all_np_data = np.concatenate([all_np_data, np_data], axis=0)

                    print(
                        f"  处理 {Path(json_file).stem}: {len(frames_data) if GENERATE_CSV else 'N/A'} 帧, NPZ数据: {len(np_data)} 有效帧")

                except Exception as e:
                    print(f"  处理文件 {json_file} 时出错: {e}")
                    continue

        # 保存为CSV文件（如果开启）
        if GENERATE_CSV and all_frames_data:
            df = pd.DataFrame(all_frames_data)
            output_csv = os.path.join(output_dir, f"{zip_name}_keypoints.csv")
            df.to_csv(output_csv, index=False, encoding='utf-8-sig')
            print(f"  已保存CSV: {output_csv} (包含 {len(all_frames_data)} 帧数据)")

        # 保存为单独的NPZ文件（如果不整合模式）
        if len(all_np_data) > 0 and not COMBINE_NPZ:
            output_npz = os.path.join(output_dir, f"{zip_name}_keypoints.npz")
            np.savez(
                output_npz,
                keypoints=all_np_data,
                keypoint_names=KEYPOINT_ORDER,
                zip_file=zip_name,
                total_frames=len(all_np_data),
                frame_indices=np.arange(len(all_np_data))
            )
            print(f"  已保存单独NPZ: {output_npz} (包含 {len(all_np_data)} 有效帧数据)")
            print(f"    数据形状: {all_np_data.shape}")

        return all_np_data, zip_name, len(all_frames_data) if GENERATE_CSV else len(all_np_data)

    except Exception as e:
        print(f"处理ZIP文件 {zip_path} 时出错: {e}")
        return None, zip_name, 0


def combine_npz_files(output_dir):
    """
    将目录下所有的NPZ文件整合成一个
    """
    npz_files = glob.glob(os.path.join(output_dir, "*_keypoints.npz"))

    if not npz_files:
        print("没有找到NPZ文件进行整合")
        return

    print(f"\n开始整合 {len(npz_files)} 个NPZ文件...")

    all_keypoints = []
    file_info = []
    total_frames = 0

    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
            keypoints = data['positions_3d']
            zip_file = data['zip_file']

            all_keypoints.append(keypoints)
            file_info.extend([zip_file] * len(keypoints))
            total_frames += len(keypoints)

            print(f"  已加载 {npz_file}: {len(keypoints)} 帧")

        except Exception as e:
            print(f"  加载文件 {npz_file} 时出错: {e}")
            continue

    if all_keypoints:
        # 合并所有关键点数据
        combined_keypoints = np.concatenate(all_keypoints, axis=0)

        # 保存整合后的文件
        combined_npz = os.path.join(output_dir, "combined_keypoints.npz")
        np.savez(
            combined_npz,
            keypoints=combined_keypoints,
            keypoint_names=KEYPOINT_ORDER,
            file_info=file_info,
            total_frames=total_frames,
            frame_indices=np.arange(total_frames),
            source_files=npz_files
        )

        print(f"\n整合完成!")
        print(f"合并了 {len(npz_files)} 个文件，总共 {total_frames} 帧")
        print(f"数据形状: {combined_keypoints.shape}")
        print(f"已保存: {combined_npz}")

        return combined_npz
    else:
        print("没有有效数据可以整合")
        return None


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
    print(f"CSV生成模式: {'开启' if GENERATE_CSV else '关闭'}")
    print(f"NPZ整合模式: {'开启' if COMBINE_NPZ else '关闭'}")

    # 处理所有ZIP文件
    total_frames = 0
    total_valid_frames = 0
    successful_zips = 0

    # 如果选择整合模式，先收集所有数据
    if COMBINE_NPZ:
        all_np_data_collection = []
        zip_names_collection = []

        for zip_file in zip_files:
            np_data, zip_name, frames_processed = process_zip_file(zip_file, output_dir)
            if frames_processed > 0:
                successful_zips += 1
                total_frames += frames_processed
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

    else:
        # 不整合模式，每个ZIP文件保存单独的NPZ
        for zip_file in zip_files:
            np_data, zip_name, frames_processed = process_zip_file(zip_file, output_dir)
            if frames_processed > 0:
                successful_zips += 1
                total_frames += frames_processed
                if np_data is not None:
                    total_valid_frames += len(np_data)

    print(f"\n处理完成!")
    print(f"成功处理 {successful_zips}/{len(zip_files)} 个ZIP文件")

    if GENERATE_CSV:
        print(f"总共提取 {total_frames} 帧的关键点数据（CSV）")

    print(f"其中 {total_valid_frames} 帧包含完整关键点数据（NPZ）")
    print(f"输出文件保存在: {output_dir}")

    # 如果不整合模式，但后来想整合，可以取消下面这行的注释
    # combine_npz_files(output_dir)


if __name__ == "__main__":
    main()
