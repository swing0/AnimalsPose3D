# 04_video_to_3d_keypoints.py
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import json
from common.ap10k_detector import AP10KAnimalPoseDetector
from common.keypoint_mapper import KeypointMapper
from common.model import TemporalModel


class VideoTo3DKeypoints:
    def __init__(self, model_checkpoint, onnx_model_path, output_dir="npz/estimate_npz",
                 architecture="3,3,3", channels=512, causal=False, dropout=0.25):
        """
        初始化视频到3D关键点转换器 - 适配正交投影版本
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 保存模型参数
        self.architecture = architecture
        self.channels = channels
        self.causal = causal
        self.dropout = dropout

        # 初始化组件
        self.detector = AP10KAnimalPoseDetector(onnx_model_path)
        self.mapper = KeypointMapper()

        # 加载3D模型
        self.model_3d = self.load_3d_model(model_checkpoint)

        print("✅ 视频到3D关键点转换器初始化完成（正交投影适配版）")

    def load_3d_model(self, checkpoint_path):
        """加载训练好的3D姿态估计模型"""
        print(f"📥 加载3D模型: {checkpoint_path}")

        # 创建模型架构
        filter_widths = [int(x) for x in self.architecture.split(',')]
        print(f"模型架构: {filter_widths}, 通道数: {self.channels}")

        model = TemporalModel(
            17, 2, 17,  # 输入: 17个关节, 2D坐标; 输出: 17个关节, 3D坐标
            filter_widths=filter_widths,
            causal=self.causal,
            dropout=self.dropout,
            channels=self.channels
        )

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        if 'model_pos' in checkpoint:
            model_weights = checkpoint['model_pos']
            model.load_state_dict(model_weights)
            print("✅ 从检查点加载模型权重成功")
        else:
            print("❌ 检查点中没有找到 'model_pos' 键")
            return None

        if torch.cuda.is_available():
            model = model.cuda()
            print("✅ 模型已移动到GPU")
        else:
            print("ℹ️ 使用CPU进行推理")

        model.eval()

        # 计算感受野
        self.receptive_field = model.receptive_field()
        self.pad = (self.receptive_field - 1) // 2
        print(f"模型感受野: {self.receptive_field}帧, 填充: {self.pad}帧")

        # 计算最小输入长度
        self.min_input_length = self.receptive_field
        print(f"最小输入序列长度: {self.min_input_length}帧")

        return model

    def normalize_keypoints_simple(self, keypoints_2d):
        """
        简单归一化2D关键点到 [-1, 1] 范围
        与训练数据的归一化方式保持一致
        """
        if len(keypoints_2d) == 0:
            return np.array([])

        # 重塑为 (N, 17, 2)
        keypoints_reshaped = keypoints_2d.reshape(-1, 17, 2)

        normalized_keypoints = []

        for frame_kps in keypoints_reshaped:
            # 找到当前帧的边界
            min_val = frame_kps.min(axis=0)
            max_val = frame_kps.max(axis=0)

            # 计算中心点和范围
            center = (min_val + max_val) / 2
            scale = np.max(max_val - min_val)

            if scale == 0:
                scale = 1.0

            # 归一化到 [-1, 1]
            normalized_frame = (frame_kps - center) / (scale / 2)
            normalized_keypoints.append(normalized_frame)

        return np.array(normalized_keypoints)

    def extract_2d_keypoints_from_video(self, video_path, confidence_threshold=0.3, max_frames=None):
        """
        从视频中提取2D关键点 - 适配正交投影
        """
        print(f"🎥 开始处理视频: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames is not None:
            total_frames = min(total_frames, max_frames)

        print(f"  视频信息: {total_frames}帧, {fps:.1f}FPS, 分辨率: {frame_width}x{frame_height}")

        keypoints_2d_sequence = []
        valid_frames = 0
        frame_info = []

        # 进度条
        pbar = tqdm(total=total_frames, desc="提取2D关键点")

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # 保存临时图像文件用于检测
            temp_img_path = f"temp_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(temp_img_path, frame)

            try:
                # 使用AP10K检测器获取2D关键点
                result = self.detector.predict(temp_img_path)
                keypoints_ap10k = result['keypoints']

                # 过滤低置信度关键点
                valid_keypoints = np.sum(keypoints_ap10k[:, 2] > confidence_threshold)

                if valid_keypoints >= 8:  # 至少8个有效关键点
                    # 映射到训练模型格式
                    keypoints_training = self.mapper.map_ap10k_to_training(keypoints_ap10k)

                    # 只保留坐标，去掉置信度
                    keypoints_2d = keypoints_training[:, :2]

                    # 添加到序列
                    keypoints_2d_sequence.append(keypoints_2d)
                    frame_info.append({
                        'frame_idx': frame_idx,
                        'valid_keypoints': valid_keypoints,
                        'timestamp': frame_idx / fps
                    })
                    valid_frames += 1

                # 清理临时文件
                os.remove(temp_img_path)

            except Exception as e:
                if frame_idx % 100 == 0:
                    print(f"⚠️ 帧 {frame_idx} 处理失败: {e}")
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

            pbar.update(1)

        pbar.close()
        cap.release()

        print(f"✅ 2D关键点提取完成: {valid_frames}/{total_frames} 有效帧")

        if valid_frames < self.min_input_length:
            print(f"❌ 有效帧数 ({valid_frames}) 小于模型要求的最小帧数 ({self.min_input_length})")
            print("💡 建议使用更长的视频或降低置信度阈值")
            return None

        # 转换为numpy数组
        keypoints_2d_array = np.array(keypoints_2d_sequence)

        # 归一化关键点坐标
        keypoints_2d_normalized = self.normalize_keypoints_simple(keypoints_2d_array)

        return {
            'keypoints_2d': keypoints_2d_normalized,
            'frame_info': frame_info,
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'valid_frames': valid_frames,
                'video_path': video_path,
                'resolution': (frame_width, frame_height)
            }
        }

    def process_sequence_in_chunks(self, keypoints_2d_sequence, chunk_size=243, overlap=81):
        """
        分块处理长序列 - 适配正交投影
        """
        seq_length = len(keypoints_2d_sequence)
        all_3d_keypoints = []

        print(f"分块处理序列: 总长度{seq_length}, 块大小{chunk_size}, 重叠{overlap}")

        start_idx = 0
        while start_idx < seq_length:
            end_idx = min(start_idx + chunk_size, seq_length)

            # 确保最后一个块有足够长度
            if end_idx - start_idx < self.min_input_length:
                break

            chunk = keypoints_2d_sequence[start_idx:end_idx]

            # 处理当前块
            chunk_3d = self.convert_chunk_to_3d(chunk)
            if len(chunk_3d) > 0:
                # 如果是重叠部分，取后半段
                if start_idx > 0:
                    overlap_start = overlap
                    chunk_3d = chunk_3d[overlap_start:]

                all_3d_keypoints.append(chunk_3d)

            start_idx += (chunk_size - overlap)

        if all_3d_keypoints:
            return np.concatenate(all_3d_keypoints, axis=0)
        else:
            return np.array([])

    def convert_chunk_to_3d(self, keypoints_2d_chunk):
        """
        将2D关键点块转换为3D关键点 - 适配正交投影
        """
        if len(keypoints_2d_chunk) == 0:
            return np.array([])

        with torch.no_grad():
            # 准备输入数据
            inputs_2d = torch.from_numpy(keypoints_2d_chunk.astype('float32')).unsqueeze(0)
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # 模型推理
            predicted_3d = self.model_3d(inputs_2d)
            keypoints_3d = predicted_3d.squeeze(0).cpu().numpy()

        return keypoints_3d

    def convert_2d_to_3d(self, keypoints_2d_sequence):
        """
        将2D关键点序列转换为3D关键点
        """
        print("🔄 开始2D到3D转换...")

        if len(keypoints_2d_sequence) == 0:
            raise ValueError("没有有效的2D关键点数据")

        seq_length = len(keypoints_2d_sequence)
        print(f"输入序列长度: {seq_length}帧")

        if seq_length < self.min_input_length:
            print(f"❌ 序列长度不足，无法处理")
            return np.array([])

        # 如果序列太长，分块处理
        if seq_length > 500:
            print("序列较长，使用分块处理...")
            keypoints_3d = self.process_sequence_in_chunks(keypoints_2d_sequence)
        else:
            # 直接处理整个序列
            keypoints_3d = self.convert_chunk_to_3d(keypoints_2d_sequence)

        print(f"✅ 3D转换完成: {keypoints_3d.shape if len(keypoints_3d) > 0 else '空'}")
        return keypoints_3d

    def save_to_npz(self, keypoints_3d, video_info, output_filename=None):
        """
        保存为与训练数据相同格式的NPZ文件 - 适配正交投影
        """
        if len(keypoints_3d) == 0:
            print("❌ 没有3D关键点数据可保存")
            return None

        # 生成输出文件名
        if output_filename is None:
            video_name = os.path.splitext(os.path.basename(video_info['video_path']))[0]
            output_filename = f"data_3d_{video_name}.npz"

        # 创建与训练数据相同的结构
        positions_3d = {
            'Animal': {
                'video_action': keypoints_3d
            }
        }

        # 保存为npz
        output_path = os.path.join(self.output_dir, output_filename)
        np.savez_compressed(output_path, positions_3d=positions_3d)

        # 保存元数据
        metadata = {
            'video_info': video_info,
            'keypoint_format': 'AP10K mapped to training format',
            'num_frames': len(keypoints_3d),
            'num_joints': 17,
            'processing_date': str(np.datetime64('now')),
            'model_architecture': self.architecture,
            'model_channels': self.channels,
            'receptive_field': self.receptive_field,
            'projection_type': 'orthographic',  # 明确标注使用正交投影
            'normalization': 'simple_centering'  # 标注归一化方法
        }

        metadata_path = os.path.join(self.output_dir, f"{os.path.splitext(output_filename)[0]}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 3D关键点已保存: {output_path}")
        print(f"   - 帧数: {len(keypoints_3d)}")
        print(f"   - 形状: {keypoints_3d.shape}")

        return output_path

    def process_video(self, video_path, confidence_threshold=0.3, max_frames=None, output_filename=None):
        """
        完整处理流程：视频 -> 2D关键点 -> 3D关键点 -> NPZ文件

        Args:
            video_path: 输入视频路径
            confidence_threshold: 关键点置信度阈值
            max_frames: 最大处理帧数
            output_filename: 输出文件名
        """
        print("🚀 开始完整处理流程...")
        print(f"模型要求: 至少 {self.min_input_length} 帧输入")

        try:
            # 1. 提取2D关键点
            extraction_result = self.extract_2d_keypoints_from_video(
                video_path, confidence_threshold, max_frames
            )

            if extraction_result is None:
                return None

            # 2. 转换为3D
            keypoints_3d = self.convert_2d_to_3d(extraction_result['keypoints_2d'])

            if len(keypoints_3d) == 0:
                print("❌ 3D转换失败")
                return None

            # 3. 保存为NPZ
            output_path = self.save_to_npz(
                keypoints_3d,
                extraction_result['video_info'],
                output_filename
            )

            print("🎉 处理完成!")
            return output_path

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数 - 使用示例"""
    # 配置路径
    MODEL_CHECKPOINT = "checkpoint_all_animals/epoch_010.bin"
    ONNX_MODEL_PATH = "model/ap10k/end2end.onnx"
    VIDEO_PATH = "video/test_video.mp4"

    # 检查文件是否存在
    if not os.path.exists(MODEL_CHECKPOINT):
        print(f"❌ 模型检查点不存在: {MODEL_CHECKPOINT}")
        return

    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ ONNX模型文件不存在: {ONNX_MODEL_PATH}")
        return

    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 视频文件不存在: {VIDEO_PATH}")
        return

    # 创建处理器
    processor = VideoTo3DKeypoints(
        model_checkpoint=MODEL_CHECKPOINT,
        onnx_model_path=ONNX_MODEL_PATH,
        architecture="3,3,3",
        channels=512,
        causal=False,
        dropout=0.25
    )

    # 处理视频
    output_npz = processor.process_video(
        VIDEO_PATH,
        confidence_threshold=0.3,
        max_frames=1000,  # 可选：限制处理帧数
        output_filename="data_3d_estimated.npz"
    )

    if output_npz:
        print(f"✅ 处理完成！输出文件: {output_npz}")

        # 验证输出文件
        try:
            data = np.load(output_npz, allow_pickle=True)
            positions_3d = data['positions_3d'].item()
            print(f"📊 输出文件验证:")
            for subject, actions in positions_3d.items():
                for action, keypoints in actions.items():
                    print(f"   {subject}/{action}: {keypoints.shape}")

            # 检查数据范围
            all_keypoints = np.concatenate([keypoints for actions in positions_3d.values()
                                            for keypoints in actions.values()], axis=0)
            print(f"   3D数据范围 - X: [{all_keypoints[..., 0].min():.3f}, {all_keypoints[..., 0].max():.3f}]")
            print(f"               Y: [{all_keypoints[..., 1].min():.3f}, {all_keypoints[..., 1].max():.3f}]")
            print(f"               Z: [{all_keypoints[..., 2].min():.3f}, {all_keypoints[..., 2].max():.3f}]")

        except Exception as e:
            print(f"⚠️ 输出文件验证失败: {e}")
    else:
        print("❌ 处理失败")


if __name__ == "__main__":
    main()