# video_to_3d_keypoints.py
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import json
from common.ap10k_detector import AP10KAnimalPoseDetector
from common.keypoint_mapper import KeypointMapper
from common.model import TemporalModel
from common.camera import normalize_screen_coordinates


class VideoTo3DKeypoints:
    def __init__(self, model_checkpoint, onnx_model_path, output_dir="npz/estimate_npz",
                 architecture="3,3,3,3", channels=512, causal=False, dropout=0.2):
        """
        初始化视频到3D关键点转换器
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

        print("✅ 视频到3D关键点转换器初始化完成")

    def load_3d_model(self, checkpoint_path):
        """加载训练好的3D姿态估计模型"""
        print(f"📥 加载3D模型: {checkpoint_path}")

        # 创建模型架构
        filter_widths = [int(x) for x in self.architecture.split(',')]
        print(f"模型架构: {filter_widths}, 通道数: {self.channels}")

        model = TemporalModel(
            17, 2, 17,
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

    def extract_2d_keypoints_from_video(self, video_path, confidence_threshold=0.3):
        """
        从视频中提取2D关键点

        Args:
            video_path: 视频文件路径
            confidence_threshold: 关键点置信度阈值
        """
        print(f"🎥 开始处理视频: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"  视频信息: {total_frames}帧, {fps:.1f}FPS")

        keypoints_2d_sequence = []
        valid_frames = 0
        frame_info = []

        # 进度条 - 使用实际总帧数
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

                if valid_keypoints >= 8:
                    # 映射到训练模型格式
                    keypoints_training = self.mapper.map_ap10k_to_training(keypoints_ap10k)

                    # 添加到序列
                    keypoints_2d_sequence.append(keypoints_training)
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
            print("💡 建议使用更长的视频")
            return None

        return {
            'keypoints_2d': np.array(keypoints_2d_sequence),
            'frame_info': frame_info,
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'valid_frames': valid_frames,
                'video_path': video_path
            }
        }

    def convert_2d_to_3d(self, keypoints_2d_sequence):
        """
        将2D关键点序列转换为3D关键点
        """
        print("🔄 开始2D到3D转换...")

        if len(keypoints_2d_sequence) == 0:
            raise ValueError("没有有效的2D关键点数据")

        # 检查序列长度
        seq_length = len(keypoints_2d_sequence)
        print(f"输入序列长度: {seq_length}帧")

        if seq_length < self.min_input_length:
            print(f"❌ 序列长度不足，无法处理")
            return np.array([])

        # 归一化2D坐标
        keypoints_2d_normalized = []
        for kp_2d in keypoints_2d_sequence:
            kp_normalized = normalize_screen_coordinates(kp_2d, w=1000, h=1000)
            keypoints_2d_normalized.append(kp_normalized)

        keypoints_2d_normalized = np.array(keypoints_2d_normalized)
        print(f"归一化后2D关键点形状: {keypoints_2d_normalized.shape}")

        # 直接处理整个序列，不进行滑动窗口
        all_3d_keypoints = []

        with torch.no_grad():
            # 直接处理整个序列
            inputs_2d = torch.from_numpy(keypoints_2d_normalized.astype('float32')).unsqueeze(0)
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            predicted_3d = self.model_3d(inputs_2d)
            keypoints_3d = predicted_3d.squeeze(0).cpu().numpy()
            all_3d_keypoints.append(keypoints_3d)

        # 合并所有结果
        if all_3d_keypoints:
            keypoints_3d = np.concatenate(all_3d_keypoints, axis=0)
        else:
            keypoints_3d = np.array([])

        print(f"✅ 3D转换完成: {keypoints_3d.shape}")
        return keypoints_3d

    def save_to_npz(self, keypoints_3d, video_info, output_filename="data_3d_animals.npz"):
        """
        保存为与训练数据相同格式的NPZ文件
        """
        if len(keypoints_3d) == 0:
            print("❌ 没有3D关键点数据可保存")
            return None

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
            'receptive_field': self.receptive_field
        }

        metadata_path = os.path.join(self.output_dir, "processing_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 3D关键点已保存: {output_path}")
        print(f"   - 帧数: {len(keypoints_3d)}")
        print(f"   - 形状: {keypoints_3d.shape}")

        return output_path

    def process_video(self, video_path, confidence_threshold=0.3):
        """
        完整处理流程：视频 -> 2D关键点 -> 3D关键点 -> NPZ文件

        Args:
            video_path: 输入视频路径
            confidence_threshold: 关键点置信度阈值
        """
        print("🚀 开始完整处理流程...")
        print(f"模型要求: 至少 {self.min_input_length} 帧输入")

        try:
            # 1. 提取2D关键点（处理整个视频）
            extraction_result = self.extract_2d_keypoints_from_video(
                video_path, confidence_threshold
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
                extraction_result['video_info']
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
    MODEL_CHECKPOINT = "checkpoint/epoch_100.bin"
    ONNX_MODEL_PATH = "model/ap10k/end2end.onnx"
    VIDEO_PATH = "video/test_video.mp4"

    # 创建处理器
    processor = VideoTo3DKeypoints(
        model_checkpoint=MODEL_CHECKPOINT,
        onnx_model_path=ONNX_MODEL_PATH,
        architecture="3,3,3,3",
        channels=512,
        causal=False,
        dropout=0.2
    )

    # 处理整个视频，不限制帧数
    output_npz = processor.process_video(
        VIDEO_PATH,
        confidence_threshold=0.2
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
        except Exception as e:
            print(f"⚠️ 输出文件验证失败: {e}")
    else:
        print("❌ 处理失败")


if __name__ == "__main__":
    main()