import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import sys
import os

# 添加common目录路径
sys.path.append('./common')

# 尝试导入训练时的模型定义
try:
    from common.transformer_model import UltraLightAnimalPoseTransformer
    print("✅ 使用训练时的模型定义")
except ImportError:
    print("⚠️ 无法导入ultra_light_transformer，使用本地定义")
    # 使用修复后的模型定义

# 将骨架按部位分组并定义颜色
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

# 如果需要，这里提供修复的模型定义
# 直接使用训练时的模型定义
from common.transformer_model import UltraLightAnimalPoseTransformer


def load_model(checkpoint_path, device):
    """加载模型"""
    print("📦 加载模型...")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查检查点结构
    if isinstance(checkpoint, dict):
        print("📋 检测到模型权重")
        print(f"检查点键: {list(checkpoint.keys())}")
        
        # 如果包含模型状态字典
        if 'model_state_dict' in checkpoint:
            checkpoint = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
    
    # 创建模型 - 使用与训练时相同的参数
    model = UltraLightAnimalPoseTransformer(
        num_joints=17,
        in_dim=2,
        embed_dim=96,
        depth=2,
        num_heads=4,
        seq_len=16,
        dropout=0.1
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint)
    
    # 设置为评估模式
    model.eval()
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型加载成功，参数数量: {total_params:,}")
    
    return model


def load_2d_data(data_2d_path, target_subject="Addax_Male", target_action="standtowalk"):
    """
    加载2D数据，支持选择动物和动作
    
    Args:
        data_2d_path: 数据文件路径
        target_subject: 目标动物名称，默认Addax_Male
        target_action: 目标动作名称，默认standtowalk
    """
    print("📂 加载2D数据...")
    
    try:
        data_2d = np.load(data_2d_path, allow_pickle=True)['positions_2d'].item()
    except Exception as e:
        print(f"❌ 加载2D数据失败: {e}")
        return None, None, None
    
    # 获取所有可用的动物
    available_subjects = list(data_2d.keys())
    print(f"🦓 可用动物: {available_subjects}")
    
    # 检查目标动物是否存在
    if target_subject not in available_subjects:
        print(f"⚠️  目标动物 '{target_subject}' 不存在，使用第一个动物: {available_subjects[0]}")
        target_subject = available_subjects[0]
    
    # 获取目标动物的所有动作
    available_actions = list(data_2d[target_subject].keys())
    print(f"🏃 动物 '{target_subject}' 的可用动作: {available_actions}")
    
    # 检查目标动作是否存在
    if target_action not in available_actions:
        print(f"⚠️  目标动作 '{target_action}' 不存在，寻找替代动作...")
        
        # 寻找具有足够帧数的动作
        suitable_actions = []
        for action in available_actions:
            # 获取该动作的所有视角
            views = data_2d[target_subject][action]
            if len(views) > 0:
                sequence_2d = np.array(views[0])  # 使用第一个视角
                if len(sequence_2d) >= 16:  # 至少16帧
                    suitable_actions.append((action, len(sequence_2d)))
        
        if not suitable_actions:
            print("⚠️  没有找到足够长的序列，尝试所有序列")
            # 尝试所有序列，即使长度不足
            for action in available_actions:
                views = data_2d[target_subject][action]
                if len(views) > 0:
                    sequence_2d = np.array(views[0])
                    suitable_actions.append((action, len(sequence_2d)))
        
        if not suitable_actions:
            print("❌ 没有可用的数据序列")
            return None, None, None
        
        # 选择最长的动作
        target_action, frame_count = max(suitable_actions, key=lambda x: x[1])
        print(f"✅ 自动选择动作: {target_action} ({frame_count}帧)")
    
    # 获取2D序列数据
    views = data_2d[target_subject][target_action]
    sequence_2d = np.array(views[0])  # 使用第一个视图
    
    print(f"✅ 加载数据: {target_subject} - {target_action}")
    print(f"   序列长度: {len(sequence_2d)} 帧")
    print(f"   关节数量: {sequence_2d.shape[1]}")
    print(f"   数据范围: X [{sequence_2d[:,:,0].min():.3f}, {sequence_2d[:,:,0].max():.3f}]")
    print(f"             Y [{sequence_2d[:,:,1].min():.3f}, {sequence_2d[:,:,1].max():.3f}]")
    
    return sequence_2d, target_subject, target_action


def convert_2d_to_3d(model, sequence_2d, device, seq_len=16):
    """将2D序列转换为3D预测"""
    print("🔄 进行2D到3D转换...")
    
    # 预处理2D数据
    sequence_2d = sequence_2d.astype(np.float32)
    
    total_frames = len(sequence_2d)
    
    if total_frames < seq_len:
        print(f"⚠️  序列长度({total_frames})小于模型要求({seq_len})，进行填充")
        # 重复最后一帧直到达到seq_len
        padding = np.repeat(sequence_2d[-1:], seq_len - total_frames, axis=0)
        sequence_2d = np.concatenate([sequence_2d, padding], axis=0)
        total_frames = len(sequence_2d)
    
    # 如果序列很长，只处理前100帧以避免内存问题
    max_frames = 100
    if total_frames > max_frames:
        print(f"⚠️  序列太长({total_frames}帧)，只处理前{max_frames}帧")
        sequence_2d = sequence_2d[:max_frames]
        total_frames = max_frames
    
    # 将序列分割为多个子序列
    num_subsequences = total_frames - seq_len + 1
    
    print(f"   处理 {num_subsequences} 个子序列")
    
    # 存储预测结果
    predictions = []
    
    with torch.no_grad():
        for i in range(num_subsequences):
            # 提取子序列
            sub_seq = sequence_2d[i:i+seq_len]
            
            # 转换为PyTorch张量并添加批次维度
            input_tensor = torch.from_numpy(sub_seq).unsqueeze(0).to(device)
            
            # 模型预测
            pred_3d = model(input_tensor)
            
            # 转换为numpy并移除批次维度
            pred_3d_np = pred_3d.cpu().numpy()[0]
            
            # 第一个子序列：保留所有帧
            if i == 0:
                predictions.extend(pred_3d_np[:seq_len-1])
            
            # 每个子序列：只保留最后一个帧
            predictions.append(pred_3d_np[-1])
    
    # 转换为完整的3D序列
    sequence_3d = np.array(predictions)
    
    print(f"✅ 转换完成，生成 {len(sequence_3d)} 帧3D数据")
    print(f"   3D数据范围: X [{sequence_3d[:,:,0].min():.3f}, {sequence_3d[:,:,0].max():.3f}]")
    print(f"               Y [{sequence_3d[:,:,1].min():.3f}, {sequence_3d[:,:,1].max():.3f}]")
    print(f"               Z [{sequence_3d[:,:,2].min():.3f}, {sequence_3d[:,:,2].max():.3f}]")
    
    return sequence_3d


def create_3d_motion_visualization(sequence_3d, subject, action):
    """创建3D运动可视化"""
    print("🎨 创建可视化...")
    
    # 使用更稳定的后端
    import matplotlib
    matplotlib.use('TkAgg')  # 使用TkAgg后端，更稳定
    
    fig = plt.figure(figsize=(16, 10))
    
    # 创建3D子图
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置初始视图
    total_frames = len(sequence_3d)
    
    # 计算合适的坐标轴范围
    all_positions = sequence_3d.reshape(-1, 3)
    
    if all_positions.shape[0] == 0:
        print("❌ 没有有效的数据点")
        return
    
    max_range = np.array([
        all_positions[:, 0].max() - all_positions[:, 0].min(),
        all_positions[:, 1].max() - all_positions[:, 1].min(),
        all_positions[:, 2].max() - all_positions[:, 2].min()
    ]).max() / 2.0
    
    if max_range == 0:
        max_range = 1.0  # 避免除以零
    
    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置坐标轴标签
    ax.set_xlabel('X (米)', fontsize=10)
    ax.set_ylabel('Y (米)', fontsize=10)
    ax.set_zlabel('Z (米)', fontsize=10)
    
    # 设置固定的视角（防止自动旋转）
    ax.view_init(elev=20., azim=45)
    
    # 存储绘图对象
    scatter_plot = None
    line_plots = {}
    
    def update_frame(frame_idx):
        """更新当前帧的显示"""
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
        
        # 绘制关节点
        scatter_plot = ax.scatter(
            frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], 
            color='darkred', s=30, alpha=0.8, label='Joints'
        )
        
        # 绘制骨骼连接
        labels_added = set()
        for group_name, group_info in SKELETON_GROUPS.items():
            for edge in group_info['edges']:
                start_joint, end_joint = edge
                if start_joint < len(frame_data) and end_joint < len(frame_data):
                    start_pos = frame_data[start_joint]
                    end_pos = frame_data[end_joint]
                    
                    # 检查是否有效点
                    if not (np.any(np.isnan(start_pos)) or np.any(np.isnan(end_pos))):
                        line_plot, = ax.plot(
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
        ax.set_title(f'{subject} - {action}\n帧 {frame_idx+1}/{total_frames}', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # 保持固定视角，不自动旋转
        # ax.view_init(elev=20., azim=45)  # 固定视角
    
    # 初始显示
    update_frame(0)
    
    # 添加图例（只添加一次）
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=8)
    
    # 创建动画
    ani = animation.FuncAnimation(
        fig, update_frame, frames=min(total_frames, 200), 
        interval=100, repeat=True, blit=False
    )
    
    # 添加更稳定的控制面板
    plt.subplots_adjust(bottom=0.25)
    
    # 添加更稳定的滑块控制
    ax_slider = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(ax_slider, '帧', 0, total_frames-1, valinit=0, valstep=1)
    
    def update_slider(val):
        frame_idx = int(frame_slider.val)
        update_frame(frame_idx)
        fig.canvas.draw_idle()
    
    frame_slider.on_changed(update_slider)
    
    # 添加更稳定的播放/暂停按钮
    ax_play = plt.axes([0.15, 0.05, 0.1, 0.04])
    play_button = Button(ax_play, '▶ 播放/暂停', color='lightblue', hovercolor='lightcyan')
    
    playing = [True]  # 使用列表以便在闭包中修改
    
    def toggle_animation(event):
        if playing[0]:
            ani.event_source.stop()
            play_button.label.set_text('▶ 播放')
        else:
            ani.event_source.start()
            play_button.label.set_text('⏸ 暂停')
        playing[0] = not playing[0]
    
    play_button.on_clicked(toggle_animation)
    
    # 添加重置按钮
    ax_reset = plt.axes([0.27, 0.05, 0.1, 0.04])
    reset_button = Button(ax_reset, '↺ 重置', color='lightgreen', hovercolor='lightcyan')
    
    def reset_animation(event):
        frame_slider.set_val(0)
        update_frame(0)
        if not playing[0]:
            ani.event_source.start()
            play_button.label.set_text('⏸ 暂停')
            playing[0] = True
        fig.canvas.draw_idle()
    
    reset_button.on_clicked(reset_animation)
    
    # 添加保存按钮
    ax_save = plt.axes([0.39, 0.05, 0.1, 0.04])
    save_button = Button(ax_save, '💾 保存GIF', color='lightyellow', hovercolor='lightcyan')
    
    def save_gif(event):
        print("💾 保存GIF动画...")
        try:
            ani.save('animal_3d_pose.gif', writer='pillow', fps=15, dpi=100)
            print("✅ GIF保存成功: animal_3d_pose.gif")
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    save_button.on_clicked(save_gif)
    
    # 添加视角控制按钮
    ax_view_front = plt.axes([0.51, 0.05, 0.08, 0.04])
    view_front_button = Button(ax_view_front, '前视图', color='lightgray', hovercolor='lightcyan')
    
    def set_front_view(event):
        ax.view_init(elev=20., azim=0)
        fig.canvas.draw_idle()
    
    view_front_button.on_clicked(set_front_view)
    
    ax_view_side = plt.axes([0.60, 0.05, 0.08, 0.04])
    view_side_button = Button(ax_view_side, '侧视图', color='lightgray', hovercolor='lightcyan')
    
    def set_side_view(event):
        ax.view_init(elev=20., azim=90)
        fig.canvas.draw_idle()
    
    view_side_button.on_clicked(set_side_view)
    
    ax_view_top = plt.axes([0.69, 0.05, 0.08, 0.04])
    view_top_button = Button(ax_view_top, '顶视图', color='lightgray', hovercolor='lightcyan')
    
    def set_top_view(event):
        ax.view_init(elev=90., azim=0)
        fig.canvas.draw_idle()
    
    view_top_button.on_clicked(set_top_view)
    
    # 添加说明文本
    ax_info = plt.axes([0.15, 0.01, 0.7, 0.03])
    ax_info.axis('off')
    ax_info.text(0.5, 0.5, '使用鼠标拖动旋转视角 | 滚轮缩放', 
                ha='center', va='center', fontsize=10, color='gray')
    
    print("🎬 动画已创建，显示窗口中...")
    print("💡 提示: 使用鼠标拖动旋转视角，滚轮缩放，按钮控制播放")
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("🎯 动物3D姿态估计模型验证工具")
    print("=" * 70)
    
    # 配置
    checkpoint_path = r'checkpoint\best_model.pt'
    data_2d_path = r'npz\real_npz\data_2d_animals_gt.npz'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"🔧 设备: {device}")
    print(f"📁 模型路径: {checkpoint_path}")
    print(f"📁 数据路径: {data_2d_path}")
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"❌ 模型文件不存在: {checkpoint_path}")
        # 尝试其他可能的位置
        possible_paths = [
            'checkpoints_light/best_model.pt',
            'checkpoint/best_model.pt',
            'best_model.pt'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"✅ 找到模型文件: {path}")
                break
    
    if not os.path.exists(data_2d_path):
        print(f"❌ 数据文件不存在: {data_2d_path}")
        return
    
    # 1. 加载模型
    model = load_model(checkpoint_path, device)
    if model is None:
        print("❌ 模型加载失败")
        return
    
    # 2. 加载2D数据
    sequence_2d, subject, action = load_2d_data(data_2d_path,"Addax_Male","fightattack")
    if sequence_2d is None:
        print("❌ 2D数据加载失败")
        return
    
    # 3. 转换为3D
    sequence_3d_pred = convert_2d_to_3d(model, sequence_2d, device)
    
    if sequence_3d_pred is None:
        print("❌ 转换失败")
        return
    
    # 4. 可视化结果
    print("\n" + "=" * 70)
    print("🎨 开始可视化...")
    
    create_3d_motion_visualization(sequence_3d_pred, subject, action)
    
    print("\n✅ 验证完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()