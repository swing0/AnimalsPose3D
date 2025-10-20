# 02b_visualize_2d_poses.py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from common.keypoint_mapper import KeypointMapper


class PoseVisualizer:
    def __init__(self):
        self.mapper = KeypointMapper()

        # 使用AP10K检测器中的颜色和骨骼定义
        self.keypoint_colors = {
            'L_Eye': [0, 255, 0],  # 绿色
            'R_Eye': [255, 128, 0],  # 橙色
            'Nose': [51, 153, 255],  # 蓝色
            'Neck': [51, 153, 255],  # 蓝色
            'Root of tail': [51, 153, 255],  # 蓝色
            'L_Shoulder': [51, 153, 255],  # 蓝色
            'L_Elbow': [51, 153, 255],  # 蓝色
            'L_F_Paw': [0, 255, 0],  # 绿色
            'R_Shoulder': [0, 255, 0],  # 绿色
            'R_Elbow': [255, 128, 0],  # 橙色
            'R_F_Paw': [0, 255, 0],  # 绿色
            'L_Hip': [255, 128, 0],  # 橙色
            'L_Knee': [255, 128, 0],  # 橙色
            'L_B_Paw': [0, 255, 0],  # 绿色
            'R_Hip': [0, 255, 0],  # 绿色
            'R_Knee': [0, 255, 0],  # 绿色
            'R_B_Paw': [0, 255, 0]  # 绿色
        }

        # 骨骼连接定义（基于AP10K检测器）
        self.skeleton_connections = [
            ('L_Eye', 'R_Eye'),  # 0: 眼睛连接
            ('L_Eye', 'Nose'),  # 1: 左眼到鼻子
            ('R_Eye', 'Nose'),  # 2: 右眼到鼻子
            ('Nose', 'Neck'),  # 3: 鼻子到颈部
            ('Neck', 'Root of tail'),  # 4: 颈部到尾巴根
            ('Neck', 'L_Shoulder'),  # 5: 颈部到左肩
            ('L_Shoulder', 'L_Elbow'),  # 6: 左肩到左肘
            ('L_Elbow', 'L_F_Paw'),  # 7: 左肘到左前爪
            ('Neck', 'R_Shoulder'),  # 8: 颈部到右肩
            ('R_Shoulder', 'R_Elbow'),  # 9: 右肩到右肘
            ('R_Elbow', 'R_F_Paw'),  # 10: 右肘到右前爪
            ('Root of tail', 'L_Hip'),  # 11: 尾巴根到左髋
            ('L_Hip', 'L_Knee'),  # 12: 左髋到左膝
            ('L_Knee', 'L_B_Paw'),  # 13: 左膝到左后爪
            ('Root of tail', 'R_Hip'),  # 14: 尾巴根到右髋
            ('R_Hip', 'R_Knee'),  # 15: 右髋到右膝
            ('R_Knee', 'R_B_Paw')  # 16: 右膝到右后爪
        ]

        # 骨骼颜色（基于AP10K检测器）
        self.skeleton_colors = [
            [0, 0, 255],  # 0: 蓝色 - 眼睛连接
            [0, 0, 255],  # 1: 蓝色 - 左眼到鼻子
            [0, 0, 255],  # 2: 蓝色 - 右眼到鼻子
            [0, 255, 0],  # 3: 绿色 - 鼻子到颈部
            [0, 255, 0],  # 4: 绿色 - 颈部到尾巴根
            [0, 255, 255],  # 5: 青色 - 颈部到左肩
            [0, 255, 255],  # 6: 青色 - 左肩到左肘
            [0, 255, 255],  # 7: 青色 - 左肘到左前爪
            [6, 156, 250],  # 8: 亮蓝色 - 颈部到右肩
            [6, 156, 250],  # 9: 亮蓝色 - 右肩到右肘
            [6, 156, 250],  # 10: 亮蓝色 - 右肘到右前爪
            [0, 255, 255],  # 11: 青色 - 尾巴根到左髋
            [0, 255, 255],  # 12: 青色 - 左髋到左膝
            [0, 255, 255],  # 13: 青色 - 左膝到左后爪
            [6, 156, 250],  # 14: 亮蓝色 - 尾巴根到右髋
            [6, 156, 250],  # 15: 亮蓝色 - 右髋到右膝
            [6, 156, 250]  # 16: 亮蓝色 - 右膝到右后爪
        ]

    def create_canvas(self, width=1000, height=1000, bg_color=(255, 255, 255)):
        """创建画布"""
        return np.ones((height, width, 3), dtype=np.uint8) * bg_color

    def draw_pose(self, canvas, keypoints, confidence_threshold=0.0):
        """
        在画布上绘制完整的姿态

        Args:
            canvas: 画布图像
            keypoints: 关键点数组 (17, 2) 或 (17, 3)
            confidence_threshold: 置信度阈值
        """
        # 首先绘制骨骼连接
        for i, (start_name, end_name) in enumerate(self.skeleton_connections):
            # 获取关键点索引
            start_idx = self.mapper.training_keypoints_order.index(
                self.mapper.ap10k_to_training[start_name]
            )
            end_idx = self.mapper.training_keypoints_order.index(
                self.mapper.ap10k_to_training[end_name]
            )

            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]

                # 检查置信度（如果有关键点分数）
                start_conf = start_kp[2] if len(start_kp) == 3 else 1.0
                end_conf = end_kp[2] if len(end_kp) == 3 else 1.0

                if start_conf > confidence_threshold and end_conf > confidence_threshold:
                    start_point = (int(start_kp[0]), int(start_kp[1]))
                    end_point = (int(end_kp[0]), int(end_kp[1]))

                    # 绘制骨骼
                    color = tuple(self.skeleton_colors[i])
                    cv2.line(canvas, start_point, end_point, color, 3)

        # 然后绘制关键点
        for i, kp in enumerate(keypoints):
            if len(kp) >= 2:
                x, y = int(kp[0]), int(kp[1])
                confidence = kp[2] if len(kp) == 3 else 1.0

                if confidence > confidence_threshold:
                    # 获取关键点名称和颜色
                    kp_name = self.mapper.training_keypoints_order[i]
                    ap10k_name = self.mapper.training_to_ap10k[kp_name]
                    color = tuple(self.keypoint_colors[ap10k_name])

                    # 绘制关键点
                    cv2.circle(canvas, (x, y), 8, color, -1)
                    cv2.circle(canvas, (x, y), 8, (255, 255, 255), 2)

                    # 添加关键点标签
                    label = f"{i}:{ap10k_name}"
                    cv2.putText(canvas, label, (x + 10, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def visualize_single_pose(self, keypoints, save_path=None, show=True):
        """可视化单个姿态"""
        canvas = self.create_canvas(1000, 1000)
        self.draw_pose(canvas, keypoints)

        # 添加标题
        cv2.putText(canvas, "2D Animal Pose Visualization", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if save_path:
            cv2.imwrite(save_path, canvas)
            print(f"✅ 姿态图像已保存: {save_path}")

        if show:
            # 转换为RGB显示
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 10))
            plt.imshow(canvas_rgb)
            plt.axis('off')
            plt.title("2D Animal Pose")
            plt.tight_layout()
            plt.show()

        return canvas


def visualize_2d_poses_accurate(npz_path, output_dir="visualization/2d_poses_accurate"):
    """
    准确可视化2D姿态数据，使用正确的关键点映射

    Args:
        npz_path: 2D姿态NPZ文件路径
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"📁 加载2D姿态数据: {npz_path}")

    # 加载数据
    data = np.load(npz_path, allow_pickle=True)

    if 'positions_2d' not in data.files:
        print("❌ 文件中没有找到 'positions_2d' 数据")
        return

    positions_2d = data['positions_2d'].item()
    metadata = data['metadata'].item() if 'metadata' in data.files else {}

    # 初始化可视化器
    visualizer = PoseVisualizer()

    print(f"📊 数据统计:")
    print(f"  主体数量: {len(positions_2d)}")

    # 为每个主体和动作创建可视化
    for subject, actions in positions_2d.items():
        print(f"\n🎨 可视化主体: {subject}")

        for action, camera_views in actions.items():
            print(f"  动作: {action}")

            for cam_idx, poses in enumerate(camera_views):
                print(f"    相机 {cam_idx}: {poses.shape} 帧")

                # 创建输出子目录
                action_dir = os.path.join(output_dir, f"{subject}_{action}_cam{cam_idx}")
                os.makedirs(action_dir, exist_ok=True)

                total_frames = len(poses)

                # 选择关键帧进行可视化
                sample_frames = min(10, total_frames)  # 最多可视化10帧
                step = max(1, total_frames // sample_frames)

                print(f"      采样 {sample_frames} 帧进行可视化...")

                for frame_idx in range(0, total_frames, step):
                    if frame_idx >= total_frames:
                        break

                    # 获取当前帧的关键点
                    keypoints = poses[frame_idx]

                    # 可视化单个姿态
                    output_path = os.path.join(action_dir, f"frame_{frame_idx:06d}.png")
                    canvas = visualizer.visualize_single_pose(
                        keypoints,
                        save_path=output_path,
                        show=False  # 不显示，只保存
                    )

                    # 在图像上添加帧信息
                    cv2.putText(canvas, f"Frame: {frame_idx}/{total_frames}",
                                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    cv2.putText(canvas, f"Camera: {cam_idx}",
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    # 重新保存带信息的图像
                    cv2.imwrite(output_path, canvas)

                print(f"    ✅ 相机 {cam_idx} 可视化完成: {action_dir}")

                # 创建第一帧的预览
                if len(poses) > 0:
                    preview_path = os.path.join(output_dir, f"preview_{subject}_{action}_cam{cam_idx}.png")
                    visualizer.visualize_single_pose(
                        poses[0],
                        save_path=preview_path,
                        show=False
                    )

    print(f"\n🎉 所有可视化完成! 输出目录: {output_dir}")




if __name__ == "__main__":
    NPZ_PATH = "npz/real_npz/data_2d_animals_gt.npz"

    if os.path.exists(NPZ_PATH):
        print("🚀 开始准确的2D姿态可视化...")

        # 显示关键点映射信息
        mapper = KeypointMapper()
        mapper.print_mapping_info()

        visualize_2d_poses_accurate(NPZ_PATH)


        print("\n🎊 可视化任务完成!")
    else:
        print(f"❌ 文件不存在: {NPZ_PATH}")
        print("💡 请先运行 02_prepare_data_animals.py 生成2D数据")