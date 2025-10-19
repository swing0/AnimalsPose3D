# ap10k_detector.py
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Dict, Any


class AP10KAnimalPoseDetector:
    def __init__(self, onnx_path: str):
        """
        初始化AP10K动物姿态检测器

        Args:
            onnx_path: ONNX模型路径
        """
        # 加载模型并获取输入尺寸
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape

        # 从模型获取输入尺寸 (batch, channel, height, width)
        # 根据错误信息，模型期望的是256x256
        self.input_size = (input_shape[3], input_shape[2])  # (width, height) = (256, 256)
        print(f"模型输入尺寸: {self.input_size}")

        self.output_names = [output.name for output in self.session.get_outputs()]

        # AP10K数据集信息
        self.keypoint_info = {
            0: dict(name='L_Eye', id=0, color=[0, 255, 0], type='upper', swap='R_Eye'),
            1: dict(name='R_Eye', id=1, color=[255, 128, 0], type='upper', swap='L_Eye'),
            2: dict(name='Nose', id=2, color=[51, 153, 255], type='upper', swap=''),
            3: dict(name='Neck', id=3, color=[51, 153, 255], type='upper', swap=''),
            4: dict(name='Root of tail', id=4, color=[51, 153, 255], type='lower', swap=''),
            5: dict(name='L_Shoulder', id=5, color=[51, 153, 255], type='upper', swap='R_Shoulder'),
            6: dict(name='L_Elbow', id=6, color=[51, 153, 255], type='upper', swap='R_Elbow'),
            7: dict(name='L_F_Paw', id=7, color=[0, 255, 0], type='upper', swap='R_F_Paw'),
            8: dict(name='R_Shoulder', id=8, color=[0, 255, 0], type='upper', swap='L_Shoulder'),
            9: dict(name='R_Elbow', id=9, color=[255, 128, 0], type='upper', swap='L_Elbow'),
            10: dict(name='R_F_Paw', id=10, color=[0, 255, 0], type='lower', swap='L_F_Paw'),
            11: dict(name='L_Hip', id=11, color=[255, 128, 0], type='lower', swap='R_Hip'),
            12: dict(name='L_Knee', id=12, color=[255, 128, 0], type='lower', swap='R_Knee'),
            13: dict(name='L_B_Paw', id=13, color=[0, 255, 0], type='lower', swap='R_B_Paw'),
            14: dict(name='R_Hip', id=14, color=[0, 255, 0], type='lower', swap='L_Hip'),
            15: dict(name='R_Knee', id=15, color=[0, 255, 0], type='lower', swap='L_Knee'),
            16: dict(name='R_B_Paw', id=16, color=[0, 255, 0], type='lower', swap='L_B_Paw'),
        }

        self.skeleton_info = {
            0: dict(link=('L_Eye', 'R_Eye'), id=0, color=[0, 0, 255]),
            1: dict(link=('L_Eye', 'Nose'), id=1, color=[0, 0, 255]),
            2: dict(link=('R_Eye', 'Nose'), id=2, color=[0, 0, 255]),
            3: dict(link=('Nose', 'Neck'), id=3, color=[0, 255, 0]),
            4: dict(link=('Neck', 'Root of tail'), id=4, color=[0, 255, 0]),
            5: dict(link=('Neck', 'L_Shoulder'), id=5, color=[0, 255, 255]),
            6: dict(link=('L_Shoulder', 'L_Elbow'), id=6, color=[0, 255, 255]),
            7: dict(link=('L_Elbow', 'L_F_Paw'), id=7, color=[0, 255, 255]),
            8: dict(link=('Neck', 'R_Shoulder'), id=8, color=[6, 156, 250]),
            9: dict(link=('R_Shoulder', 'R_Elbow'), id=9, color=[6, 156, 250]),
            10: dict(link=('R_Elbow', 'R_F_Paw'), id=10, color=[6, 156, 250]),
            11: dict(link=('Root of tail', 'L_Hip'), id=11, color=[0, 255, 255]),
            12: dict(link=('L_Hip', 'L_Knee'), id=12, color=[0, 255, 255]),
            13: dict(link=('L_Knee', 'L_B_Paw'), id=13, color=[0, 255, 255]),
            14: dict(link=('Root of tail', 'R_Hip'), id=14, color=[6, 156, 250]),
            15: dict(link=('R_Hip', 'R_Knee'), id=15, color=[6, 156, 250]),
            16: dict(link=('R_Knee', 'R_B_Paw'), id=16, color=[6, 156, 250]),
        }

        # 创建关键点名称到ID的映射
        self.name_to_id = {info['name']: kid for kid, info in self.keypoint_info.items()}

    def preprocess(self, image: np.ndarray, center: Tuple[float, float], scale: float) -> np.ndarray:
        """
        预处理图像

        Args:
            image: 输入图像 (H, W, C)
            center: 边界框中心点
            scale: 缩放比例

        Returns:
            预处理后的图像张量
        """
        # 计算仿射变换矩阵 - 使用模型期望的256x256尺寸
        trans = self.get_affine_transform(center, scale, 0, self.input_size)

        # 应用仿射变换
        input_image = cv2.warpAffine(
            image, trans,
            (int(self.input_size[0]), int(self.input_size[1])),
            flags=cv2.INTER_LINEAR
        )

        # BGR to RGB
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # 归一化 - 使用配置文件中的参数
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        input_image = (input_image - mean) / std

        # 调整维度顺序: HWC -> CHW
        input_image = np.transpose(input_image, (2, 0, 1)).astype(np.float32)

        # 添加batch维度
        input_image = np.expand_dims(input_image, axis=0)

        return input_image

    def get_affine_transform(self, center: Tuple[float, float], scale: float, rot: float,
                             output_size: Tuple[int, int]) -> np.ndarray:
        """
        获取仿射变换矩阵
        """
        src_w = scale
        dst_w = output_size[0]
        dst_h = output_size[1]

        src_dir = self.get_dir([0, src_w * -0.5], rot)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(src, dst)
        return trans

    def get_dir(self, src_point: np.ndarray, rot_rad: float) -> np.ndarray:
        """获取旋转后的方向"""
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs
        return np.array(src_result, dtype=np.float32)

    def get_3rd_point(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """获取第三个点"""
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def simcc_decode(self, simcc_x: np.ndarray, simcc_y: np.ndarray) -> np.ndarray:
        """
        SimCC解码，将模型输出转换为关键点坐标

        Args:
            simcc_x: x方向预测
            simcc_y: y方向预测

        Returns:
            关键点坐标 (17, 3) [x, y, score]
        """
        keypoints = []

        # 获取输出形状
        batch_size, num_kpts, simcc_dim_x = simcc_x.shape
        _, _, simcc_dim_y = simcc_y.shape

        # print(f"SimCC输出形状 - X: {simcc_x.shape}, Y: {simcc_y.shape}")

        for i in range(num_kpts):  # 17个关键点
            # 找到最大响应的位置
            max_idx_x = np.argmax(simcc_x[0, i])
            max_idx_y = np.argmax(simcc_y[0, i])

            # 获取置信度分数
            score_x = simcc_x[0, i, max_idx_x]
            score_y = simcc_y[0, i, max_idx_y]
            score = (score_x + score_y) / 2

            # 转换为坐标 (0-1范围)
            x = max_idx_x / (simcc_dim_x - 1)
            y = max_idx_y / (simcc_dim_y - 1)

            keypoints.append([x, y, score])

        return np.array(keypoints)

    def predict(self, image_path: str, bbox: List[float] = None) -> Dict[str, Any]:
        """
        预测图像中的关键点

        Args:
            image_path: 图像路径
            bbox: 边界框 [x1, y1, x2, y2]，如果为None则使用整张图像

        Returns:
            包含关键点和元数据的字典
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        original_h, original_w = image.shape[:2]

        # 如果没有提供边界框，使用整张图像
        if bbox is None:
            bbox = [0, 0, original_w, original_h]

        x1, y1, x2, y2 = bbox
        bbox_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        bbox_size = np.array([x2 - x1, y2 - y1])

        # 计算缩放比例 (padding=1.25)
        scale = max(bbox_size) * 1.25

        # 预处理
        input_tensor = self.preprocess(image, bbox_center, scale)
        # print(f"输入张量形状: {input_tensor.shape}")

        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        simcc_x, simcc_y = outputs

        # 解码关键点
        keypoints_normalized = self.simcc_decode(simcc_x, simcc_y)

        # 将关键点坐标转换回原图坐标
        keypoints_original = self.transform_keypoints_to_original(
            keypoints_normalized, bbox_center, scale, (original_w, original_h)
        )

        return {
            'keypoints': keypoints_original,
            'bbox': bbox,
            'image_shape': (original_h, original_w),
            'keypoints_normalized': keypoints_normalized
        }

    def transform_keypoints_to_original(self, keypoints: np.ndarray, center: np.ndarray,
                                        scale: float, image_size: Tuple[int, int]) -> np.ndarray:
        """
        将归一化关键点坐标转换回原图坐标
        """
        output_size = self.input_size
        trans = self.get_affine_transform(center, scale, 0, output_size)
        trans_inv = cv2.invertAffineTransform(trans)

        keypoints_original = []
        for kp in keypoints:
            x_norm, y_norm, score = kp
            # 转换到输入尺寸坐标
            x_input = x_norm * (output_size[0] - 1)
            y_input = y_norm * (output_size[1] - 1)

            # 应用逆仿射变换
            point = np.array([x_input, y_input, 1.0])
            x_orig, y_orig = np.dot(trans_inv, point)[:2]

            # 确保坐标在图像范围内
            x_orig = np.clip(x_orig, 0, image_size[0] - 1)
            y_orig = np.clip(y_orig, 0, image_size[1] - 1)

            keypoints_original.append([x_orig, y_orig, score])

        return np.array(keypoints_original)

    def visualize(self, image_path: str, result: Dict[str, Any],
                  confidence_threshold: float = 0.3, save_path: str = None):
        """
        可视化检测结果

        Args:
            image_path: 原图像路径
            result: predict方法的返回结果
            confidence_threshold: 置信度阈值
            save_path: 保存路径，如果为None则显示图像
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(image_rgb)

        keypoints = result['keypoints']

        # 绘制骨骼连接
        for skeleton_id, skeleton in self.skeleton_info.items():
            start_name, end_name = skeleton['link']
            color = skeleton['color']

            start_id = self.name_to_id[start_name]
            end_id = self.name_to_id[end_name]

            start_kp = keypoints[start_id]
            end_kp = keypoints[end_id]

            # 只有当两个关键点的置信度都超过阈值时才绘制
            if (start_kp[2] > confidence_threshold and
                    end_kp[2] > confidence_threshold):
                start_point = (int(start_kp[0]), int(start_kp[1]))
                end_point = (int(end_kp[0]), int(end_kp[1]))

                ax.plot([start_point[0], end_point[0]],
                        [start_point[1], end_point[1]],
                        color=np.array(color) / 255.0, linewidth=3, alpha=0.8)

        # 绘制关键点
        for kp_id, keypoint in enumerate(keypoints):
            x, y, score = keypoint
            if score > confidence_threshold:
                color = self.keypoint_info[kp_id]['color']
                ax.scatter(x, y, s=80, color=np.array(color) / 255.0,
                           marker='o', edgecolors='white', linewidth=2)
                ax.text(x, y - 10, self.keypoint_info[kp_id]['name'],
                        fontsize=8, color='white', weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=np.array(color) / 255.0, alpha=0.8))

        ax.set_title('AP10K Animal Pose Estimation', fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"结果已保存至: {save_path}")
        else:
            plt.show()

    def print_keypoint_scores(self, result: Dict[str, Any]):
        """打印关键点置信度分数"""
        print("关键点置信度分数:")
        print("-" * 40)
        for i, kp in enumerate(result['keypoints']):
            info = self.keypoint_info[i]
            print(f"{info['name']:12s}: {kp[2]:.3f}")
        print("-" * 40)


# 使用示例
def main():
    # 初始化检测器
    detector = AP10KAnimalPoseDetector("../model/ap10k/end2end.onnx")

    # 检测图像
    image_path = "addax.png"  # 替换为您的图像路径

    try:
        # 进行预测（如果不提供bbox，则使用整张图像）
        result = detector.predict(image_path)

        # 打印关键点分数
        # detector.print_keypoint_scores(result)

        # 可视化结果
        detector.visualize(image_path, result, confidence_threshold=0.3,
                           save_path="result_visualization.jpg")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()