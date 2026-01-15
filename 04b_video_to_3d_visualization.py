# 04b_video_to_3d_visualization.py
# 视频到3D关键点的可视化工具 - 同时显示2D视频和3D动画

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from tqdm import tqdm
from common.ap10k_detector import AP10KAnimalPoseDetector
from common.keypoint_mapper import KeypointMapper
from common.transformer_model import UltraLightAnimalPoseTransformer

# 骨架分组定义（与训练数据一致）
SKELETON_GROUPS = {
    'trunk': {
        'edges': [(0, 4), (4, 3), (3, 1), (3, 2)],
        'color': 'black', 'label': 'Head & Neck'
    },
    'front_left': {
        'edges': [(4, 5), (5, 6), (6, 7)],
        'color': 'red', 'label': 'Front Left'
    },
    'front_right': {
        'edges': [(4, 8), (8, 9), (9, 10)],
        'color': 'orange', 'label': 'Front Right'
    },
    'back_left': {
        'edges': [(0, 11), (11, 12), (12, 13)],
        'color': 'blue', 'label': 'Back Left'
    },
    'back_right': {
        'edges': [(0, 14), (14, 15), (15, 16)],
        'color': 'cyan', 'label': 'Back Right'
    }
}

# 2D可视化颜色定义
SKELETON_COLORS_2D = {
    'trunk': (0, 0, 0),        # 黑色
    'front_left': (0, 0, 255),  # 红色
    'front_right': (0, 165, 255),  # 橙色
    'back_left': (255, 0, 0),   # 蓝色
    'back_right': (255, 255, 0) # 青色
}


class VideoTo3DVisualizer:
    def __init__(self, model_checkpoint, onnx_model_path):
        """初始化可视化器"""
        print("🎯 初始化视频到3D可视化器...")
        
        # 初始化组件
        self.detector = AP10KAnimalPoseDetector(onnx_model_path)
        self.mapper = KeypointMapper()
        
        # 加载3D模型
        self.model_3d = self.load_3d_model(model_checkpoint)
        
        # 视频相关变量
        self.video_cap = None
        self.video_frames = []
        self.video_info = {}
        
        # 关键点数据
        self.keypoints_2d_sequence = []
        self.keypoints_3d_sequence = []
        
        print("✅ 可视化器初始化完成")
    
    def load_3d_model(self, checkpoint_path):
        """加载训练好的3D姿态估计模型"""
        print(f"📥 加载3D模型: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ 模型文件不存在: {checkpoint_path}")
            return None
        
        # 创建模型实例（使用与训练时相同的参数）
        model = UltraLightAnimalPoseTransformer(
            num_joints=17, 
            in_dim=2, 
            embed_dim=96,
            depth=2, 
            num_heads=4, 
            seq_len=16, 
            dropout=0.1
        )
        
        # 加载权重
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 处理不同的检查点格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # 直接加载整个检查点
                model.load_state_dict(checkpoint)
            
            print("✅ 模型权重加载成功")
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📊 模型参数量: {total_params:,}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return None
        
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("✅ 模型已移动到GPU")
        else:
            print("ℹ️ 使用CPU进行推理")
        
        return model
    
    def extract_video_frames(self, video_path, max_frames=100):
        """提取视频帧并检测2D关键点"""
        print(f"🎥 处理视频: {video_path}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 限制处理帧数
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
        
        print(f"📊 视频信息: {total_frames}帧, {fps:.1f}FPS, 分辨率: {frame_width}x{frame_height}")
        
        self.video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'resolution': (frame_width, frame_height),
            'video_path': video_path
        }
        
        # 提取帧和关键点
        self.video_frames = []
        self.keypoints_2d_sequence = []
        valid_frames = 0
        
        pbar = tqdm(total=total_frames, desc="提取视频帧和关键点")
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存原始帧
            self.video_frames.append(frame.copy())
            
            # 保存临时图像用于检测
            temp_img_path = f"temp_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(temp_img_path, frame)
            
            try:
                # 检测2D关键点
                result = self.detector.predict(temp_img_path)
                keypoints_ap10k = result['keypoints']
                
                # 过滤低置信度关键点
                valid_keypoints = np.sum(keypoints_ap10k[:, 2] > 0.3)
                
                if valid_keypoints >= 8:  # 至少8个有效关键点
                    # 映射到训练格式
                    keypoints_training = self.mapper.map_ap10k_to_training(keypoints_ap10k)
                    keypoints_2d = keypoints_training[:, :2]  # 只保留坐标
                    
                    self.keypoints_2d_sequence.append(keypoints_2d)
                    valid_frames += 1
                else:
                    # 如果没有检测到足够的关键点，添加空数据
                    self.keypoints_2d_sequence.append(np.full((17, 2), np.nan))
                
                # 清理临时文件
                os.remove(temp_img_path)
                
            except Exception as e:
                if frame_idx % 50 == 0:
                    print(f"⚠️ 帧 {frame_idx} 处理失败: {e}")
                # 添加空数据
                self.keypoints_2d_sequence.append(np.full((17, 2), np.nan))
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        print(f"✅ 视频处理完成: {valid_frames}/{total_frames} 有效帧")
        
        return valid_frames
    
    def convert_2d_to_3d(self):
        """将2D关键点序列转换为3D关键点"""
        print("🔄 开始2D到3D转换...")
        
        if len(self.keypoints_2d_sequence) == 0:
            print("❌ 没有2D关键点数据")
            return False
        
        # 过滤无效帧
        valid_keypoints = []
        valid_indices = []
        
        for i, kps in enumerate(self.keypoints_2d_sequence):
            if not np.any(np.isnan(kps)):
                valid_keypoints.append(kps)
                valid_indices.append(i)
        
        if len(valid_keypoints) < 16:
            print(f"❌ 有效帧数 ({len(valid_keypoints)}) 不足，需要至少16帧")
            return False
        
        print(f"📊 使用 {len(valid_keypoints)} 个有效帧进行3D转换")
        
        # 转换为numpy数组
        keypoints_2d_array = np.array(valid_keypoints)
        
        # 归一化关键点（与训练时一致）
        keypoints_2d_normalized = self.normalize_keypoints(keypoints_2d_array)
        
        # 分块处理（适应模型输入长度）
        self.keypoints_3d_sequence = []
        seq_len = 16  # 模型输入序列长度
        
        for i in range(0, len(keypoints_2d_normalized) - seq_len + 1, seq_len):
            chunk = keypoints_2d_normalized[i:i+seq_len]
            
            with torch.no_grad():
                # 准备输入
                inputs_2d = torch.from_numpy(chunk.astype('float32')).unsqueeze(0)
                
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                
                # 模型推理
                predicted_3d = self.model_3d(inputs_2d)
                chunk_3d = predicted_3d.squeeze(0).cpu().numpy()
                
                # 反归一化到原始尺度
                chunk_3d_denorm = self.denormalize_3d(chunk_3d, keypoints_2d_array[i:i+seq_len])
                
                # 映射回原始帧索引
                for j in range(len(chunk_3d_denorm)):
                    frame_3d = np.full((17, 3), np.nan)
                    frame_3d[:] = chunk_3d_denorm[j]
                    self.keypoints_3d_sequence.append(frame_3d)
        
        print(f"✅ 3D转换完成: {len(self.keypoints_3d_sequence)} 帧3D关键点")
        return True
    
    def normalize_keypoints(self, keypoints_2d):
        """归一化2D关键点到 [-1, 1] 范围"""
        normalized = []
        
        for frame_kps in keypoints_2d:
            # 找到边界
            min_val = frame_kps.min(axis=0)
            max_val = frame_kps.max(axis=0)
            
            # 计算中心和尺度
            center = (min_val + max_val) / 2
            scale = np.max(max_val - min_val)
            
            if scale == 0:
                scale = 1.0
            
            # 归一化
            normalized_frame = (frame_kps - center) / (scale / 2)
            normalized.append(normalized_frame)
        
        return np.array(normalized)
    
    def denormalize_3d(self, keypoints_3d, original_2d):
        """将3D关键点反归一化到合理尺度"""
        denormalized = []
        
        for i, frame_3d in enumerate(keypoints_3d):
            # 使用原始2D数据的尺度信息
            frame_2d = original_2d[i]
            
            # 计算2D数据的尺度
            min_2d = frame_2d.min(axis=0)
            max_2d = frame_2d.max(axis=0)
            scale_2d = np.max(max_2d - min_2d)
            
            if scale_2d == 0:
                scale_2d = 100.0  # 默认尺度
            
            # 将3D数据缩放到合理范围
            scale_3d = scale_2d * 0.5  # 3D尺度约为2D的一半
            frame_3d_scaled = frame_3d * scale_3d
            

            # frame_3d_scaled[:, 0] = -frame_3d_scaled[:, 0]  # 反转X轴
            # frame_3d_scaled[:, 1] = -frame_3d_scaled[:, 1]
            frame_3d_scaled[:, 2] = -frame_3d_scaled[:, 2]  # 反转Z轴
            
            denormalized.append(frame_3d_scaled)
        
        return np.array(denormalized)
    
    def draw_2d_skeleton(self, frame, keypoints_2d, confidence_threshold=0.3):
        """在视频帧上绘制2D骨架"""
        if np.any(np.isnan(keypoints_2d)):
            return frame
        
        frame_with_skeleton = frame.copy()
        
        # 绘制关节点
        for i, (x, y) in enumerate(keypoints_2d):
            if not np.isnan(x) and not np.isnan(y):
                center = (int(x), int(y))
                cv2.circle(frame_with_skeleton, center, 4, (0, 255, 255), -1)  # 黄色点
                cv2.putText(frame_with_skeleton, str(i), (center[0] + 5, center[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 绘制骨架连线
        for group_name, group_info in SKELETON_GROUPS.items():
            color = SKELETON_COLORS_2D[group_name]
            
            for edge in group_info['edges']:
                start_joint, end_joint = edge
                
                if (start_joint < len(keypoints_2d) and end_joint < len(keypoints_2d) and
                    not np.any(np.isnan(keypoints_2d[start_joint])) and 
                    not np.any(np.isnan(keypoints_2d[end_joint]))):
                    
                    start_point = (int(keypoints_2d[start_joint][0]), int(keypoints_2d[start_joint][1]))
                    end_point = (int(keypoints_2d[end_joint][0]), int(keypoints_2d[end_joint][1]))
                    
                    cv2.line(frame_with_skeleton, start_point, end_point, color, 2)
        
        return frame_with_skeleton
    
    def create_3d_visualization(self, sequence_3d, title="3D Animal Pose"):
        """创建3D关键点动画可视化"""
        print("🎨 创建3D可视化...")
        
        # 使用更稳定的后端
        import matplotlib
        matplotlib.use('TkAgg')
        
        fig = plt.figure(figsize=(16, 8))
        
        # 创建3D子图
        ax_3d = fig.add_subplot(121, projection='3d')
        
        # 计算合适的坐标轴范围
        valid_positions = []
        for frame in sequence_3d:
            if not np.any(np.isnan(frame)):
                valid_positions.extend(frame)
        
        if len(valid_positions) == 0:
            print("❌ 没有有效的3D数据点")
            return
        
        valid_positions = np.array(valid_positions)
        
        max_range = np.array([
            valid_positions[:, 0].max() - valid_positions[:, 0].min(),
            valid_positions[:, 1].max() - valid_positions[:, 1].min(),
            valid_positions[:, 2].max() - valid_positions[:, 2].min()
        ]).max() / 2.0
        
        if max_range == 0:
            max_range = 1.0
        
        mid_x = (valid_positions[:, 0].max() + valid_positions[:, 0].min()) * 0.5
        mid_y = (valid_positions[:, 1].max() + valid_positions[:, 1].min()) * 0.5
        mid_z = (valid_positions[:, 2].max() + valid_positions[:, 2].min()) * 0.5
        
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 设置坐标轴标签
        ax_3d.set_xlabel('X', fontsize=10)
        ax_3d.set_ylabel('Y', fontsize=10)
        ax_3d.set_zlabel('Z', fontsize=10)
        
        # 设置固定视角
        ax_3d.view_init(elev=20., azim=45)
        
        # 存储绘图对象
        scatter_plot = None
        line_plots = {}
        
        def update_3d_frame(frame_idx):
            """更新3D帧显示"""
            nonlocal scatter_plot, line_plots
            
            # 清除之前的绘图
            if scatter_plot is not None:
                scatter_plot.remove()
            for line_plot in line_plots.values():
                line_plot.remove()
            line_plots.clear()
            
            # 获取当前帧数据
            if frame_idx >= len(sequence_3d):
                return
            
            frame_data = sequence_3d[frame_idx]
            
            # 检查数据有效性
            if np.any(np.isnan(frame_data)):
                return
            
            # 绘制关节点
            scatter_plot = ax_3d.scatter(
                frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], 
                color='darkred', s=30, alpha=0.8, label='Joints'
            )
            
            # 绘制骨骼连接
            labels_added = set()
            for group_name, group_info in SKELETON_GROUPS.items():
                for edge in group_info['edges']:
                    start_joint, end_joint = edge
                    if (start_joint < len(frame_data) and end_joint < len(frame_data) and
                        not np.any(np.isnan(frame_data[start_joint])) and 
                        not np.any(np.isnan(frame_data[end_joint]))):
                        
                        start_pos = frame_data[start_joint]
                        end_pos = frame_data[end_joint]
                        
                        line_plot, = ax_3d.plot(
                            [start_pos[0], end_pos[0]], 
                            [start_pos[1], end_pos[1]], 
                            [start_pos[2], end_pos[2]], 
                            color=group_info['color'], 
                            linewidth=2, 
                            label=group_info['label'] if group_name not in labels_added else ""
                        )
                        line_plots[f"{group_name}_{edge}"] = line_plot
                        labels_added.add(group_name)
            
            # 设置标题
            ax_3d.set_title(f'{title}\n帧 {frame_idx+1}/{len(sequence_3d)}', 
                           fontsize=12, fontweight='bold', pad=10)
        
        # 初始显示
        update_3d_frame(0)
        
        # 添加图例
        handles, labels = ax_3d.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax_3d.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
        
        # 创建动画
        ani_3d = animation.FuncAnimation(
            fig, update_3d_frame, frames=min(len(sequence_3d), 100), 
            interval=100, repeat=True, blit=False
        )
        
        return fig, ax_3d, ani_3d
    
    def visualize_combined(self, video_path, max_frames=100):
        """组合可视化：左侧显示2D视频，右侧显示3D动画"""
        print("🎬 开始组合可视化...")
        
        # 1. 提取视频帧和2D关键点
        valid_frames = self.extract_video_frames(video_path, max_frames)
        if valid_frames < 16:
            print(f"❌ 有效帧数不足，无法进行3D转换")
            return
        
        # 2. 转换为3D关键点
        if not self.convert_2d_to_3d():
            print("❌ 3D转换失败")
            return
        
        # 3. 创建可视化界面
        import matplotlib
        matplotlib.use('TkAgg')
        
        fig = plt.figure(figsize=(20, 8))
        
        # 左侧：2D视频显示
        ax_2d = fig.add_subplot(121)
        ax_2d.set_title("2D视频与关键点检测", fontsize=14, fontweight='bold')
        ax_2d.axis('off')
        
        # 右侧：3D动画显示
        ax_3d = fig.add_subplot(122, projection='3d')
        ax_3d.set_title("3D姿态估计", fontsize=14, fontweight='bold')
        
        # 计算3D坐标轴范围
        valid_3d_positions = []
        for frame in self.keypoints_3d_sequence:
            if not np.any(np.isnan(frame)):
                valid_3d_positions.extend(frame)
        
        if len(valid_3d_positions) == 0:
            print("❌ 没有有效的3D数据")
            return
        
        valid_3d_positions = np.array(valid_3d_positions)
        max_range = np.array([
            valid_3d_positions[:, 0].max() - valid_3d_positions[:, 0].min(),
            valid_3d_positions[:, 1].max() - valid_3d_positions[:, 1].min(),
            valid_3d_positions[:, 2].max() - valid_3d_positions[:, 2].min()
        ]).max() / 2.0
        
        if max_range == 0:
            max_range = 1.0
        
        mid_x = (valid_3d_positions[:, 0].max() + valid_3d_positions[:, 0].min()) * 0.5
        mid_y = (valid_3d_positions[:, 1].max() + valid_3d_positions[:, 1].min()) * 0.5
        mid_z = (valid_3d_positions[:, 2].max() + valid_3d_positions[:, 2].min()) * 0.5
        
        ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
        ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
        ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.view_init(elev=20., azim=45)
        
        # 存储绘图对象
        img_display = None
        scatter_3d = None
        line_plots_3d = {}
        
        def update_combined_frame(frame_idx):
            """更新组合帧显示"""
            nonlocal img_display, scatter_3d, line_plots_3d
            
            # 更新2D显示
            if frame_idx < len(self.video_frames):
                frame_2d = self.video_frames[frame_idx]
                
                # 绘制2D骨架
                if frame_idx < len(self.keypoints_2d_sequence):
                    frame_with_skeleton = self.draw_2d_skeleton(frame_2d, self.keypoints_2d_sequence[frame_idx])
                else:
                    frame_with_skeleton = frame_2d
                
                # 转换为RGB格式
                frame_rgb = cv2.cvtColor(frame_with_skeleton, cv2.COLOR_BGR2RGB)
                
                if img_display is None:
                    img_display = ax_2d.imshow(frame_rgb)
                else:
                    img_display.set_data(frame_rgb)
                
                ax_2d.set_title(f"2D视频 (帧 {frame_idx+1}/{len(self.video_frames)})", 
                               fontsize=12, fontweight='bold')
            
            # 更新3D显示
            if frame_idx < len(self.keypoints_3d_sequence):
                frame_3d = self.keypoints_3d_sequence[frame_idx]
                
                # 清除之前的3D绘图
                if scatter_3d is not None:
                    scatter_3d.remove()
                for line_plot in line_plots_3d.values():
                    line_plot.remove()
                line_plots_3d.clear()
                
                # 检查3D数据有效性
                if not np.any(np.isnan(frame_3d)):
                    # 绘制3D关节点
                    scatter_3d = ax_3d.scatter(
                        frame_3d[:, 0], frame_3d[:, 1], frame_3d[:, 2], 
                        color='darkred', s=30, alpha=0.8, label='Joints'
                    )
                    
                    # 绘制3D骨骼连接
                    labels_added = set()
                    for group_name, group_info in SKELETON_GROUPS.items():
                        for edge in group_info['edges']:
                            start_joint, end_joint = edge
                            if (start_joint < len(frame_3d) and end_joint < len(frame_3d) and
                                not np.any(np.isnan(frame_3d[start_joint])) and 
                                not np.any(np.isnan(frame_3d[end_joint]))):
                                
                                start_pos = frame_3d[start_joint]
                                end_pos = frame_3d[end_joint]
                                
                                line_plot, = ax_3d.plot(
                                    [start_pos[0], end_pos[0]], 
                                    [start_pos[1], end_pos[1]], 
                                    [start_pos[2], end_pos[2]], 
                                    color=group_info['color'], 
                                    linewidth=2, 
                                    label=group_info['label'] if group_name not in labels_added else ""
                                )
                                line_plots_3d[f"{group_name}_{edge}"] = line_plot
                                labels_added.add(group_name)
                
                ax_3d.set_title(f"3D姿态估计 (帧 {frame_idx+1}/{len(self.keypoints_3d_sequence)})", 
                               fontsize=12, fontweight='bold')
        
        # 初始显示
        update_combined_frame(0)
        
        # 添加3D图例
        handles, labels = ax_3d.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax_3d.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
        
        # 添加控制面板
        plt.subplots_adjust(bottom=0.15)
        
        # 滑块控制
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgoldenrodyellow')
        total_frames = max(len(self.video_frames), len(self.keypoints_3d_sequence))
        frame_slider = Slider(ax_slider, '帧', 0, total_frames-1, valinit=0, valstep=1)
        
        def update_slider(val):
            frame_idx = int(frame_slider.val)
            update_combined_frame(frame_idx)
            fig.canvas.draw_idle()
        
        frame_slider.on_changed(update_slider)
        
        # 播放/暂停按钮
        ax_play = plt.axes([0.15, 0.01, 0.1, 0.04])
        play_button = Button(ax_play, '▶ 播放/暂停', color='lightblue', hovercolor='lightcyan')
        
        playing = [True]
        ani_combined = animation.FuncAnimation(
            fig, update_combined_frame, frames=min(total_frames, 100), 
            interval=100, repeat=True, blit=False
        )
        
        def toggle_animation(event):
            if playing[0]:
                ani_combined.event_source.stop()
                play_button.label.set_text('▶ 播放')
            else:
                ani_combined.event_source.start()
                play_button.label.set_text('⏸ 暂停')
            playing[0] = not playing[0]
        
        play_button.on_clicked(toggle_animation)
        
        # 重置按钮
        ax_reset = plt.axes([0.27, 0.01, 0.1, 0.04])
        reset_button = Button(ax_reset, '↺ 重置', color='lightgreen', hovercolor='lightcyan')
        
        def reset_animation(event):
            frame_slider.set_val(0)
            update_combined_frame(0)
            if not playing[0]:
                ani_combined.event_source.start()
                play_button.label.set_text('⏸ 暂停')
                playing[0] = True
            fig.canvas.draw_idle()
        
        reset_button.on_clicked(reset_animation)
        
        print("🎉 组合可视化已创建!")
        print("💡 提示: 使用滑块控制帧，按钮控制播放，鼠标拖动旋转3D视角")
        
        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("🎯 视频到3D关键点可视化工具")
    print("=" * 70)
    
    # 配置路径
    MODEL_CHECKPOINT = "checkpoint/best_model.pt"
    ONNX_MODEL_PATH = "model/ap10k/end2end.onnx"
    VIDEO_PATH = "video/test_video_yang.mp4"  # 替换为你的视频路径
    
    # 检查文件是否存在
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"❌ ONNX模型文件不存在: {ONNX_MODEL_PATH}")
        print("💡 请确保模型文件路径正确")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ 视频文件不存在: {VIDEO_PATH}")
        print("💡 请提供有效的视频文件路径")
        return
    
    # 创建可视化器
    visualizer = VideoTo3DVisualizer(MODEL_CHECKPOINT, ONNX_MODEL_PATH)
    
    if visualizer.model_3d is None:
        print("❌ 3D模型加载失败，无法继续")
        return
    
    # 执行组合可视化
    try:
        visualizer.visualize_combined(VIDEO_PATH, max_frames=100)
    except Exception as e:
        print(f"❌ 可视化过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()