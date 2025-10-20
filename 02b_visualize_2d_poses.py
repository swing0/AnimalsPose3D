# 02b_visualize_2d_poses.py
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from common.keypoint_mapper import KeypointMapper


class PoseVisualizer:
    def __init__(self):
        self.mapper = KeypointMapper()

        # 使用更鲜艳的颜色
        self.keypoint_colors = [
            [255, 0, 0],  # 0: 红色 - 尾巴根
            [0, 255, 0],  # 1: 绿色 - 左眼
            [0, 0, 255],  # 2: 蓝色 - 右眼
            [255, 255, 0],  # 3: 黄色 - 鼻子
            [255, 0, 255],  # 4: 紫色 - 颈部
            [0, 255, 255],  # 5: 青色 - 左肩
            [128, 0, 128],  # 6: 深紫 - 左肘
            [255, 165, 0],  # 7: 橙色 - 左前爪
            [128, 128, 0],  # 8: 橄榄色 - 右肩
            [0, 128, 128],  # 9: 深青 - 右肘
            [128, 0, 0],  # 10: 深红 - 右前爪
            [0, 128, 0],  # 11: 深绿 - 左髋
            [0, 0, 128],  # 12: 深蓝 - 左膝
            [128, 128, 128],  # 13: 灰色 - 左后爪
            [255, 192, 203],  # 14: 粉色 - 右髋
            [165, 42, 42],  # 15: 棕色 - 右膝
            [210, 180, 140]  # 16: 棕褐色 - 右后爪
        ]

        # 训练数据的关键点顺序（根据你的数据集）
        self.training_keypoint_names = [
            "Root of Tail", "Left Eye", "Right Eye", "Nose", "Neck",
            "Left Shoulder", "Left Elbow", "Left Front Paw",
            "Right Shoulder", "Right Elbow", "Right Front Paw",
            "Left Hip", "Left Knee", "Left Back Paw",
            "Right Hip", "Right Knee", "Right Back Paw"
        ]

        # 简化的骨骼连接（基于四足动物解剖结构）
        self.skeleton_connections = [
            # 头部和躯干
            (0, 4),  # 尾巴根 -> 颈部
            (4, 3),  # 颈部 -> 鼻子
            (3, 1),  # 鼻子 -> 左眼
            (3, 2),  # 鼻子 -> 右眼

            # 前腿 - 左侧
            (4, 5),  # 颈部 -> 左肩
            (5, 6),  # 左肩 -> 左肘
            (6, 7),  # 左肘 -> 左前爪

            # 前腿 - 右侧
            (4, 8),  # 颈部 -> 右肩
            (8, 9),  # 右肩 -> 右肘
            (9, 10),  # 右肘 -> 右前爪

            # 后腿 - 左侧
            (0, 11),  # 尾巴根 -> 左髋
            (11, 12),  # 左髋 -> 左膝
            (12, 13),  # 左膝 -> 左后爪

            # 后腿 - 右侧
            (0, 14),  # 尾巴根 -> 右髋
            (14, 15),  # 右髋 -> 右膝
            (15, 16)  # 右膝 -> 右后爪
        ]

        # 骨骼颜色（按连接分组）- 修复：确保有15个颜色
        self.skeleton_colors = [
            [255, 0, 0],  # 0: 红色 - 躯干
            [255, 128, 0],  # 1: 橙色 - 头部
            [255, 128, 0],  # 2: 橙色 - 头部
            [255, 128, 0],  # 3: 橙色 - 头部
            [0, 255, 0],  # 4: 绿色 - 左前腿
            [0, 255, 0],  # 5: 绿色 - 左前腿
            [0, 255, 0],  # 6: 绿色 - 左前腿
            [0, 0, 255],  # 7: 蓝色 - 右前腿
            [0, 0, 255],  # 8: 蓝色 - 右前腿
            [0, 0, 255],  # 9: 蓝色 - 右前腿
            [255, 0, 255],  # 10: 紫色 - 左后腿
            [255, 0, 255],  # 11: 紫色 - 左后腿
            [255, 0, 255],  # 12: 紫色 - 左后腿
            [128, 0, 128],  # 13: 深紫 - 右后腿
            [128, 0, 128],  # 14: 深紫 - 右后腿
            [128, 0, 128]  # 15: 深紫 - 右后腿
        ]

    def create_canvas(self, width=800, height=800, bg_color=(240, 240, 240)):
        """创建画布"""
        canvas = np.ones((height, width, 3), dtype=np.uint8) * bg_color

        # 添加网格线
        grid_size = 50
        for x in range(0, width, grid_size):
            cv2.line(canvas, (x, 0), (x, height), (220, 220, 220), 1)
        for y in range(0, height, grid_size):
            cv2.line(canvas, (0, y), (width, y), (220, 220, 220), 1)

        # 添加中心十字
        cv2.line(canvas, (width // 2, 0), (width // 2, height), (200, 200, 200), 2)
        cv2.line(canvas, (0, height // 2), (width, height // 2), (200, 200, 200), 2)

        return canvas

    def denormalize_coordinates(self, normalized_coords, canvas_width=800, canvas_height=800):
        """将归一化坐标 [-1,1] 转换为画布坐标"""
        # 归一化坐标到像素坐标的转换
        x = (normalized_coords[0] + 1) * (canvas_width / 2)
        y = (normalized_coords[1] + 1) * (canvas_height / 2)
        # 注意：在图像坐标系中Y轴是向下的，所以需要翻转Y轴
        y = canvas_height - y  # 翻转Y轴
        return int(x), int(y)

    def draw_pose(self, canvas, keypoints):
        """
        在画布上绘制完整的姿态

        Args:
            canvas: 画布图像
            keypoints: 关键点数组 (17, 2) 归一化坐标
        """
        canvas_height, canvas_width = canvas.shape[:2]

        # 首先绘制骨骼连接
        for i, (start_idx, end_idx) in enumerate(self.skeleton_connections):
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]

                # 转换坐标
                start_point = self.denormalize_coordinates(start_kp, canvas_width, canvas_height)
                end_point = self.denormalize_coordinates(end_kp, canvas_width, canvas_height)

                # 绘制骨骼
                color = tuple(self.skeleton_colors[i])
                cv2.line(canvas, start_point, end_point, color, 3)

                # 在骨骼中间添加连接编号
                mid_x = (start_point[0] + end_point[0]) // 2
                mid_y = (start_point[1] + end_point[1]) // 2
                cv2.putText(canvas, str(i), (mid_x, mid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # 然后绘制关键点
        for i, kp in enumerate(keypoints):
            if len(kp) >= 2:
                x, y = self.denormalize_coordinates(kp, canvas_width, canvas_height)
                color = tuple(self.keypoint_colors[i])

                # 绘制关键点
                cv2.circle(canvas, (x, y), 8, color, -1)
                cv2.circle(canvas, (x, y), 8, (255, 255, 255), 2)

                # 添加关键点标签
                label = f"{i}:{self.training_keypoint_names[i]}"
                cv2.putText(canvas, label, (x + 12, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def visualize_single_pose(self, keypoints, save_path=None, show=True, title="2D Animal Pose"):
        """可视化单个姿态"""
        canvas = self.create_canvas(800, 800)
        self.draw_pose(canvas, keypoints)

        # 添加标题和信息
        cv2.putText(canvas, title, (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # 添加图例说明
        info_lines = [
            "Color Legend:",
            "Red: Torso, Orange: Head",
            "Green: Left Front, Blue: Right Front",
            "Purple: Left Back, Dark Purple: Right Back"
        ]

        for i, line in enumerate(info_lines):
            cv2.putText(canvas, line, (20, 60 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        if save_path:
            cv2.imwrite(save_path, canvas)
            print(f"✅ 姿态图像已保存: {save_path}")

        if show:
            # 转换为RGB显示
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 10))
            plt.imshow(canvas_rgb)
            plt.axis('off')
            plt.title(title)
            plt.tight_layout()
            plt.show()

        return canvas

    def print_keypoint_info(self):
        """打印关键点信息"""
        print("\n🔍 关键点信息:")
        for i, name in enumerate(self.training_keypoint_names):
            color = self.keypoint_colors[i]
            print(f"  {i:2d}: {name:15s} - Color: {color}")

    def print_skeleton_info(self):
        """打印骨骼连接信息"""
        print("\n🔗 骨骼连接信息:")
        for i, (start_idx, end_idx) in enumerate(self.skeleton_connections):
            start_name = self.training_keypoint_names[start_idx]
            end_name = self.training_keypoint_names[end_idx]
            color = self.skeleton_colors[i]
            print(f"  {i:2d}: {start_name:15s} -> {end_name:15s} - Color: {color}")


def visualize_2d_poses_simple(npz_path, output_dir="visualization/2d_poses_simple"):
    """
    简化可视化2D姿态数据

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

    # 打印关键点和骨骼信息
    visualizer.print_keypoint_info()
    visualizer.print_skeleton_info()

    print(f"\n📊 数据统计:")
    print(f"  主体数量: {len(positions_2d)}")

    # 获取视角信息
    view_angles = metadata.get('view_angles', ['view_0', 'view_1', 'view_2', 'view_3'])
    view_names = metadata.get('view_names', ['View 0', 'View 1', 'View 2', 'View 3'])

    # 为每个主体和动作创建可视化
    for subject, actions in positions_2d.items():
        print(f"\n🎨 可视化主体: {subject}")

        for action, camera_views in actions.items():
            print(f"  动作: {action}")
            print(f"  视角数量: {len(camera_views)}")

            # 为每个视角创建可视化
            for view_idx, poses in enumerate(camera_views):
                view_name = view_names[view_idx] if view_idx < len(view_names) else f"View {view_idx}"
                print(f"    视角 {view_idx} ({view_name}): {poses.shape} 帧")

                # 创建输出子目录
                action_dir = os.path.join(output_dir, f"{subject}_{action}_view{view_idx}")
                os.makedirs(action_dir, exist_ok=True)

                total_frames = len(poses)

                # 选择关键帧进行可视化
                sample_frames = min(5, total_frames)  # 最多可视化5帧
                step = max(1, total_frames // sample_frames)

                print(f"      采样 {sample_frames} 帧进行可视化...")

                for frame_idx in range(0, total_frames, step):
                    if frame_idx >= total_frames:
                        break

                    # 获取当前帧的关键点
                    keypoints = poses[frame_idx]

                    # 可视化单个姿态
                    output_path = os.path.join(action_dir, f"frame_{frame_idx:06d}.png")
                    title = f"{subject} - {action} - {view_name} - Frame {frame_idx}"

                    canvas = visualizer.visualize_single_pose(
                        keypoints,
                        save_path=output_path,
                        show=False,
                        title=title
                    )

                print(f"    ✅ 视角 {view_idx} 可视化完成: {action_dir}")

                # 创建第一帧的预览
                if len(poses) > 0:
                    preview_path = os.path.join(output_dir, f"preview_{subject}_{action}_view{view_idx}.png")
                    title = f"{subject} - {action} - {view_name} (Preview)"
                    visualizer.visualize_single_pose(
                        poses[0],
                        save_path=preview_path,
                        show=False,
                        title=title
                    )

    print(f"\n🎉 所有可视化完成! 输出目录: {output_dir}")


def create_comparison_visualization(npz_path, output_dir="visualization/comparison"):
    """
    创建多视角对比可视化
    """
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    positions_2d = data['positions_2d'].item()
    metadata = data['metadata'].item()

    visualizer = PoseVisualizer()
    view_names = metadata.get('view_names', ['Front', 'Side', 'Oblique', 'Top'])

    # 选择一个样本
    subject = list(positions_2d.keys())[0]
    action = list(positions_2d[subject].keys())[0]
    views = positions_2d[subject][action]

    print(f"创建多视角对比: {subject} - {action}")

    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.flatten()

    for view_idx, (view_poses, view_name) in enumerate(zip(views, view_names)):
        if view_idx >= 4:  # 最多显示4个视角
            break

        # 获取第一帧
        keypoints = view_poses[0]
        canvas = visualizer.create_canvas(400, 400)
        visualizer.draw_pose(canvas, keypoints)

        # 转换为RGB
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # 显示
        axes[view_idx].imshow(canvas_rgb)
        axes[view_idx].set_title(f"{view_name} View", fontsize=12, fontweight='bold')
        axes[view_idx].axis('off')

    plt.suptitle(f"Multi-view Comparison: {subject} - {action}", fontsize=16, fontweight='bold')
    plt.tight_layout()

    comparison_path = os.path.join(output_dir, f"comparison_{subject}_{action}.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✅ 对比可视化已保存: {comparison_path}")


if __name__ == "__main__":
    NPZ_PATH = "npz/real_npz/data_2d_animals_gt.npz"

    if os.path.exists(NPZ_PATH):
        print("🚀 开始2D姿态可视化...")

        # 简单可视化
        visualize_2d_poses_simple(NPZ_PATH)

        # 创建对比可视化
        create_comparison_visualization(NPZ_PATH)

        print("\n🎊 所有可视化任务完成!")
    else:
        print(f"❌ 文件不存在: {NPZ_PATH}")
        print("💡 请先运行 02_prepare_data_animals.py 生成2D数据")