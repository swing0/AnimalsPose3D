import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

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


def load_animal_data(data_3d_path, subject=None, action=None):
    """加载动物3D数据"""
    data_3d = np.load(data_3d_path, allow_pickle=True)['positions_3d'].item()
    
    # 如果没有指定，使用第一个动物和动作
    if subject is None:
        subject = list(data_3d.keys())[0]
    if action is None:
        action = list(data_3d[subject].keys())[0]
    
    # 获取3D序列数据
    sequence_3d = data_3d[subject][action]
    
    print(f"加载数据: {subject} - {action}")
    print(f"序列长度: {len(sequence_3d)} 帧")
    print(f"关节数量: {sequence_3d.shape[1]}")
    print(f"数据范围: [{sequence_3d.min():.3f}, {sequence_3d.max():.3f}] 米")
    
    return sequence_3d, subject, action


def create_3d_motion_visualization(sequence_3d, subject, action):
    """创建3D运动可视化"""
    fig = plt.figure(figsize=(16, 10))
    
    # 创建3D子图
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置初始视图
    current_frame = 0
    total_frames = len(sequence_3d)
    
    # 计算合适的坐标轴范围
    all_positions = sequence_3d.reshape(-1, 3)
    max_range = np.array([
        all_positions[:, 0].max() - all_positions[:, 0].min(),
        all_positions[:, 1].max() - all_positions[:, 1].min(),
        all_positions[:, 2].max() - all_positions[:, 2].min()
    ]).max() / 2.0
    
    mid_x = (all_positions[:, 0].max() + all_positions[:, 0].min()) * 0.5
    mid_y = (all_positions[:, 1].max() + all_positions[:, 1].min()) * 0.5
    mid_z = (all_positions[:, 2].max() + all_positions[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
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
        frame_data = sequence_3d[frame_idx]
        
        # 绘制关节点
        scatter_plot = ax.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], 
                                 color='darkred', s=50, alpha=0.8, label='Joints')
        
        # 绘制骨架连线
        for group_name, group_info in SKELETON_GROUPS.items():
            for edge in group_info['edges']:
                line_data = frame_data[list(edge), :]
                line_plot = ax.plot(line_data[:, 0], line_data[:, 1], line_data[:, 2],
                                   color=group_info['color'], linewidth=3, alpha=0.8,
                                   label=group_info['label'] if edge == group_info['edges'][0] else "")
                line_plots[f"{group_name}_{edge}"] = line_plot[0]
        
        # 更新标题
        ax.set_title(f"{subject} - {action}\nFrame: {frame_idx + 1}/{total_frames}", 
                    fontsize=14, fontweight='bold')
        
        ax.set_xlabel('X (Side)', fontsize=12)
        ax.set_ylabel('Y (Height)', fontsize=12)
        ax.set_zlabel('Z (Depth)', fontsize=12)
        
        # 添加图例
        if frame_idx == 0:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        return scatter_plot, *line_plots.values()
    
    # 创建滑块控制
    plt.subplots_adjust(bottom=0.2)
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    frame_slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, 
                         valinit=0, valstep=1)
    
    # 创建控制按钮
    ax_play = plt.axes([0.8, 0.02, 0.08, 0.04])
    play_button = Button(ax_play, 'Play')
    
    ax_pause = plt.axes([0.9, 0.02, 0.08, 0.04])
    pause_button = Button(ax_pause, 'Pause')
    
    # 动画状态
    anim_running = False
    anim = None
    
    def animate(frame):
        """动画函数"""
        frame_slider.set_val(frame % total_frames)
        return update_frame(int(frame_slider.val))
    
    def play_animation(event):
        """播放动画"""
        nonlocal anim_running, anim
        if not anim_running:
            anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                         interval=100, blit=False, repeat=True)
            anim_running = True
            plt.draw()
    
    def pause_animation(event):
        """暂停动画"""
        nonlocal anim_running, anim
        if anim_running and anim is not None:
            anim.event_source.stop()
            anim_running = False
    
    def update_slider(val):
        """滑块更新"""
        frame_idx = int(frame_slider.val)
        update_frame(frame_idx)
        fig.canvas.draw_idle()
    
    # 绑定事件
    frame_slider.on_changed(update_slider)
    play_button.on_clicked(play_animation)
    pause_button.on_clicked(pause_animation)
    
    # 初始显示
    update_frame(0)
    
    # 添加键盘控制
    def on_key(event):
        nonlocal current_frame
        if event.key == 'right':
            current_frame = min(current_frame + 1, total_frames - 1)
            frame_slider.set_val(current_frame)
        elif event.key == 'left':
            current_frame = max(current_frame - 1, 0)
            frame_slider.set_val(current_frame)
        elif event.key == ' ':
            if anim_running:
                pause_animation(None)
            else:
                play_animation(None)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # 显示操作说明
    print("\nControls:")
    print("- Use slider to control frame")
    print("- Click 'Play' button to start animation")
    print("- Click 'Pause' button to stop animation")
    print("- Use left/right arrow keys for frame-by-frame control")
    print("- Spacebar to play/pause")
    
    plt.show()


def visualize_multiple_views(sequence_3d, subject, action):
    """多视角可视化"""
    fig = plt.figure(figsize=(18, 6))
    
    # 创建三个不同视角的子图
    views = [
        ('Front View', (0, 0)),
        ('Side View', (90, 0)),
        ('Top View', (0, 90))
    ]
    
    axes = []
    for i, (view_name, (elev, azim)) in enumerate(views):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(view_name, fontsize=12)
        axes.append(ax)
    
    # 显示第一帧
    frame_data = sequence_3d[0]
    
    for ax in axes:
        # 绘制关节点
        ax.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], 
                  color='darkred', s=30, alpha=0.8)
        
        # 绘制骨架连线
        for group_name, group_info in SKELETON_GROUPS.items():
            for edge in group_info['edges']:
                line_data = frame_data[list(edge), :]
                ax.plot(line_data[:, 0], line_data[:, 1], line_data[:, 2],
                       color=group_info['color'], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.suptitle(f"{subject} - {action} - Multiple Views", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 加载数据
    data_3d_path = r'npz\real_npz\data_3d_animals.npz'
    subject = 'Addax_Male'
    action = 'attackfence'
    
    try:
        sequence_3d, subject, action = load_animal_data(data_3d_path, subject, action)
        
        print("\n选择可视化模式:")
        print("1. 3D运动动画（交互式）")
        print("2. 多视角静态图")
        
        choice = input("请输入选择 (1 或 2, 默认1): ").strip()
        
        if choice == '2':
            visualize_multiple_views(sequence_3d, subject, action)
        else:
            create_3d_motion_visualization(sequence_3d, subject, action)
            
    except Exception as e:
        print(f"错误: {e}")
        print("请检查数据文件路径是否正确")