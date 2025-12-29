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

    # 自动获取第一个动物和第一个动作（也可以手动指定）
    sub = list(d3.keys())[0]
    act = list(d3[sub].keys())[3]

    # 抽取一帧数据
    frame_3d = d3[sub][act][40]  # 取第10帧
    frame_2d = d2[sub][act][1][40]  # 第一个视角的第10帧

    fig = plt.figure(figsize=(15, 7))

    # --- 左图：3D 骨架 ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"3D Skeleton: {sub} ({act})")

    for group_name, group_info in SKELETON_GROUPS.items():
        for edge in group_info['edges']:
            # 注意：绘图时将 3D 的 Y 放在 Z 轴位置，以对齐视觉上的“高度”
            ax1.plot(frame_3d[edge, 0], frame_3d[edge, 2], frame_3d[edge, 1],
                     color=group_info['color'], lw=2)

    # 画出所有关节点
    ax1.scatter(frame_3d[:, 0], frame_3d[:, 2], frame_3d[:, 1], color='gray', s=20)

    ax1.set_xlabel('X (Side)');
    ax1.set_ylabel('Z (Depth)');
    ax1.set_zlabel('Y (Height)')
    # 设置相等的比例尺
    max_range = np.array([frame_3d[:, 0].max() - frame_3d[:, 0].min(),
                          frame_3d[:, 1].max() - frame_3d[:, 1].min(),
                          frame_3d[:, 2].max() - frame_3d[:, 2].min()]).max() / 2.0
    ax1.set_xlim(-max_range, max_range)
    ax1.set_ylim(-max_range, max_range)
    ax1.set_zlim(-max_range, max_range)

    # --- 右图：2D 投影 ---
    ax2 = fig.add_subplot(122)
    ax2.set_title("2D Perspective Projection (Color Coded)")

    for group_name, group_info in SKELETON_GROUPS.items():
        is_first = True  # 用于图例展示
        for edge in group_info['edges']:
            label = group_info['label'] if is_first else None
            ax2.plot(frame_2d[edge, 0], -frame_2d[edge, 1],
                     color=group_info['color'], lw=2, label=label)
            is_first = False

    ax2.scatter(frame_2d[:, 0], -frame_2d[:, 1], color='gray', s=20, zorder=5)
    ax2.set_aspect('equal')
    ax2.legend(loc='upper right', fontsize='small')
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualize(r'npz\real_npz\data_3d_animals.npz', r'npz\real_npz\data_2d_animals_gt.npz')