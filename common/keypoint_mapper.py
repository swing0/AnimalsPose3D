# keypoint_mapper.py
import numpy as np
from typing import Dict, List


class KeypointMapper:
    """处理AP10K关键点与训练模型关键点之间的映射"""

    def __init__(self):
        # AP10K关键点名称到ID的映射
        self.ap10k_mapping = {
            'L_Eye': 0, 'R_Eye': 1, 'Nose': 2, 'Neck': 3, 'Root of tail': 4,
            'L_Shoulder': 5, 'L_Elbow': 6, 'L_F_Paw': 7,
            'R_Shoulder': 8, 'R_Elbow': 9, 'R_F_Paw': 10,
            'L_Hip': 11, 'L_Knee': 12, 'L_B_Paw': 13,
            'R_Hip': 14, 'R_Knee': 15, 'R_B_Paw': 16
        }

        # 训练模型的关键点顺序（根据你的animals_keypoint.npz）
        self.training_keypoints_order = [
            "Root of Tail", "Left Eye", "Right Eye", "Nose", "Neck",
            "Left Shoulder", "Left Elbow", "Left Front Paw",
            "Right Shoulder", "Right Elbow", "Right Front Paw",
            "Left Hip", "Left Knee", "Left Back Paw",
            "Right Hip", "Right Knee", "Right Back Paw"
        ]

        # AP10K到训练模型的映射关系
        self.ap10k_to_training = {
            'Root of tail': 'Root of Tail',
            'L_Eye': 'Left Eye',
            'R_Eye': 'Right Eye',
            'Nose': 'Nose',
            'Neck': 'Neck',
            'L_Shoulder': 'Left Shoulder',
            'L_Elbow': 'Left Elbow',
            'L_F_Paw': 'Left Front Paw',
            'R_Shoulder': 'Right Shoulder',
            'R_Elbow': 'Right Elbow',
            'R_F_Paw': 'Right Front Paw',
            'L_Hip': 'Left Hip',
            'L_Knee': 'Left Knee',
            'L_B_Paw': 'Left Back Paw',
            'R_Hip': 'Right Hip',
            'R_Knee': 'Right Knee',
            'R_B_Paw': 'Right Back Paw'
        }

        # 训练模型到AP10K的映射关系
        self.training_to_ap10k = {v: k for k, v in self.ap10k_to_training.items()}

        # 创建索引映射
        self.index_mapping = self._create_index_mapping()
        self.reverse_mapping = {v: k for k, v in self.index_mapping.items()}

    def _create_index_mapping(self) -> Dict[int, int]:
        """创建从AP10K索引到训练模型索引的映射"""
        mapping = {}

        for ap10k_name, training_name in self.ap10k_to_training.items():
            ap10k_idx = self.ap10k_mapping[ap10k_name]
            training_idx = self.training_keypoints_order.index(training_name)
            mapping[ap10k_idx] = training_idx

        return mapping

    def map_ap10k_to_training(self, ap10k_keypoints: np.ndarray) -> np.ndarray:
        """
        将AP10K关键点映射到训练模型的格式

        Args:
            ap10k_keypoints: AP10K格式的关键点 (17, 3) [x, y, confidence]

        Returns:
            训练模型格式的关键点 (17, 2) [x, y]
        """
        # 创建空的训练格式关键点数组
        training_keypoints = np.zeros((17, 2), dtype=np.float32)

        # 应用映射
        for ap10k_idx, training_idx in self.index_mapping.items():
            if ap10k_idx < len(ap10k_keypoints):
                training_keypoints[training_idx] = ap10k_keypoints[ap10k_idx][:2]  # 只取x,y

        return training_keypoints

    def map_training_to_ap10k(self, training_3d_keypoints: np.ndarray) -> np.ndarray:
        """
        将训练模型的3D关键点映射回AP10K格式

        Args:
            training_3d_keypoints: 训练模型输出的3D关键点 (17, 3)

        Returns:
            AP10K格式的3D关键点 (17, 3)
        """
        ap10k_3d_keypoints = np.zeros((17, 3), dtype=np.float32)

        for ap10k_idx, training_idx in self.index_mapping.items():
            if training_idx < len(training_3d_keypoints):
                # 确保维度匹配
                if training_3d_keypoints.shape[1] == 3:  # 已经是3D
                    ap10k_3d_keypoints[ap10k_idx] = training_3d_keypoints[training_idx]
                else:  # 如果是2D，添加零作为z坐标
                    ap10k_3d_keypoints[ap10k_idx, :2] = training_3d_keypoints[training_idx]
                    ap10k_3d_keypoints[ap10k_idx, 2] = 0.0  # 默认z=0

        return ap10k_3d_keypoints

    def get_ap10k_keypoint_name(self, index: int) -> str:
        """根据索引获取AP10K关键点名称"""
        for name, idx in self.ap10k_mapping.items():
            if idx == index:
                return name
        return f"Unknown_{index}"

    def get_training_keypoint_name(self, index: int) -> str:
        """根据索引获取训练模型关键点名称"""
        if index < len(self.training_keypoints_order):
            return self.training_keypoints_order[index]
        return f"Unknown_{index}"

    def print_mapping_info(self):
        """打印映射信息"""
        print("🔗 关键点映射信息:")
        print("-" * 50)
        print(f"{'AP10K名称':<15} {'AP10K索引':<10} {'训练名称':<15} {'训练索引':<10}")
        print("-" * 50)

        for ap10k_name, training_name in self.ap10k_to_training.items():
            ap10k_idx = self.ap10k_mapping[ap10k_name]
            training_idx = self.training_keypoints_order.index(training_name)
            print(f"{ap10k_name:<15} {ap10k_idx:<10} {training_name:<15} {training_idx:<10}")
        print("-" * 50)