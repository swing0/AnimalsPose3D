import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def visualize(data_3d_path, data_2d_path):
    # 加载数据
    d3 = np.load(data_3d_path, allow_pickle=True)['positions_3d'].item()
    d2 = np.load(data_2d_path, allow_pickle=True)['positions_2d'].item()

    # 自动获取第一个动物和第一个动作
    sub = list(d3.keys())[0]
    act = list(d3[sub].keys())[0]  # 使用第一个动作

    # 抽取一帧数据
    frame_3d = d3[sub][act][0]  # 取第0帧

    fig = plt.figure(figsize=(20, 10))

    # --- 左图：3D 骨架 ---
    ax1 = fig.add_subplot(231, projection='3d')
    ax1.set_title(f"3D Skeleton: {sub} ({act})")

    for group_name, group_info in SKELETON_GROUPS.items():
        for edge in group_info['edges']:
            ax1.plot(frame_3d[edge, 0], frame_3d[edge, 1], frame_3d[edge, 2],
                     color=group_info['color'], lw=2)

    ax1.scatter(frame_3d[:, 0], frame_3d[:, 1], frame_3d[:, 2], color='gray', s=20)
    ax1.set_xlabel('X (Side)'); ax1.set_ylabel('Y (Depth)'); ax1.set_zlabel('Z (Height)')
    max_range = np.array([frame_3d[:, 0].max() - frame_3d[:, 0].min(),
                          frame_3d[:, 1].max() - frame_3d[:, 1].min(),
                          frame_3d[:, 2].max() - frame_3d[:, 2].min()]).max() / 2.0
    ax1.set_xlim(-max_range, max_range)
    ax1.set_ylim(-max_range, max_range)
    ax1.set_zlim(-max_range, max_range)

    # --- 四个摄像机视角 ---
    view_titles = ['Camera 0: Front (0°)', 'Camera 1: Right (90°)', 
                   'Camera 2: Back (180°)', 'Camera 3: Left (270°)']
    
    for i in range(4):
        ax = fig.add_subplot(2, 3, i+2)
        frame_2d = d2[sub][act][i][0]  # 第i个视角的第0帧
        
        ax.set_title(view_titles[i])
        
        for group_name, group_info in SKELETON_GROUPS.items():
            is_first = True
            for edge in group_info['edges']:
                label = group_info['label'] if is_first else None
                ax.plot(frame_2d[edge, 0], frame_2d[edge, 1],
                         color=group_info['color'], lw=2, label=label)
                is_first = False

        ax.scatter(frame_2d[:, 0], frame_2d[:, 1], color='gray', s=20, zorder=5)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 只在第一个2D图中显示图例
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize(r'npz\real_npz\data_3d_animals.npz', r'npz\real_npz\data_2d_animals_gt.npz')