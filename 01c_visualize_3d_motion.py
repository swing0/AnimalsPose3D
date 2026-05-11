import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# ========== 骨架配置 ==========
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
        'edges': [(4, 8), (8, 9), (10, 9)], # 修正部分可能的索引顺序
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
    data = np.load(data_3d_path, allow_pickle=True)
    # 兼容两种可能的 key 格式
    key = 'positions_3d' if 'positions_3d' in data else 'positions'
    data_3d = data[key].item()
    
    if subject is None:
        subject = list(data_3d.keys())[0]
    if action is None:
        action = list(data_3d[subject].keys())[0]
    
    sequence_3d = data_3d[subject][action]
    
    print(f"✅ 加载成功: {subject} - {action} ({len(sequence_3d)} 帧)")
    return sequence_3d, subject, action

def create_interactive_3d_vis(sequence_3d, subject, action):
    """创建纯净版交互式3D运动动画"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.15) # 预留滑块空间

    total_frames = len(sequence_3d)
    
    # 存储绘图对象的列表，方便清除
    lines = []
    scatters = []

    def update_frame(frame_idx):
        frame_idx = int(frame_idx)
        # 清除旧线条
        while lines:
            lines.pop(0).remove()
        while scatters:
            scatters.pop(0).remove()

        frame_data = sequence_3d[frame_idx]
        
        # 1. 绘制关节点
        sc = ax.scatter(frame_data[:, 0], frame_data[:, 1], frame_data[:, 2], 
                        c='black', s=20, alpha=0.6)
        scatters.append(sc)

        # 2. 绘制骨架
        for name, info in SKELETON_GROUPS.items():
            for edge in info['edges']:
                line_pts = frame_data[list(edge), :]
                ln, = ax.plot(line_pts[:, 0], line_pts[:, 1], line_pts[:, 2], 
                             color=info['color'], linewidth=2, 
                             label=info['label'] if edge == info['edges'][0] else "")
                lines.append(ln)

        # 3. 动态更新坐标轴（可选：固定范围或跟随移动）
        # 这里使用局部跟随，让视野始终聚焦于动物
        center = frame_data[0] # 以根节点为中心
        r = 1.0 # 视角半径(米)
        ax.set_xlim(center[0]-r, center[0]+r)
        ax.set_ylim(center[1]-r, center[1]+r)
        ax.set_zlim(center[2]-r, center[2]+r)

        ax.set_title(f"Animal: {subject} | Action: {action}\nFrame: {frame_idx}/{total_frames}")
        
        # 仅在第一帧添加图例
        if frame_idx == 0 and not ax.get_legend():
            ax.legend(loc='upper left', prop={'size': 8})

    # --- 交互组件 ---
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, total_frames-1, valinit=0, valstep=1)

    anim_running = [False] # 使用列表包裹以在函数内修改
    anim = [None]

    def update_slider(val):
        update_frame(val)
        fig.canvas.draw_idle()

    slider.on_changed(update_slider)

    # 自动播放逻辑
    def animate(i):
        new_val = (slider.val + 1) % total_frames
        slider.set_val(new_val)
        return []

    def toggle_play(event):
        if anim_running[0]:
            anim[0].event_source.stop()
            btn_play.label.set_text('▶ Play')
        else:
            anim[0] = animation.FuncAnimation(fig, animate, interval=50, cache_frame_data=False)
            btn_play.label.set_text('⏸ Pause')
        anim_running[0] = not anim_running[0]

    ax_play = plt.axes([0.85, 0.04, 0.1, 0.05])
    btn_play = Button(ax_play, '▶ Play')
    btn_play.on_clicked(toggle_play)

    # 初始帧
    update_frame(0)
    
    print("\n💡 交互指南:")
    print("- 拖动下方进度条查看特定帧")
    print("- 点击右下角 Play/Pause 开始或停止动画")
    print("- 鼠标左键旋转 3D 视角，右键缩放")
    
    plt.show()

if __name__ == '__main__':
    # 路径根据你的项目结构调整
    DATA_PATH = r'npz\real_npz\data_3d_animals.npz'
    
    # 你可以手动指定，或留空(None)自动选择第一个
    SUB = 'Giant_Panda_Male'
    ACT = 'attackfence'

    try:
        seq, sub, act = load_animal_data(DATA_PATH, SUB, ACT)
        create_interactive_3d_vis(seq, sub, act)
    except Exception as e:
        print(f"❌ 运行失败: {e}")