import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

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


def normalize_2d(pose_2d):
    max_vals = np.max(np.abs(pose_2d), axis=(1, 2), keepdims=True)
    max_vals[max_vals < 1e-5] = 1.0
    return pose_2d / max_vals


def load_animal_data(data_path, subject=None, action=None):
    data = np.load(data_path, allow_pickle=True)
    key = 'positions_3d' if 'positions_3d' in data else 'positions'
    data_3d = data[key].item()

    if subject is None:
        subject = list(data_3d.keys())[0]
    if action is None:
        action = list(data_3d[subject].keys())[0]

    sequence_3d = data_3d[subject][action]
    sequence_3d = sequence_3d - sequence_3d[:, 0:1, :]

    print(f"加载成功: {subject} - {action} ({len(sequence_3d)} 帧)")
    return sequence_3d, subject, action


def create_interactive_vis(sequence_3d, subject, action):
    total_frames = len(sequence_3d)

    fig = plt.figure(figsize=(16, 7))
    ax_3d = fig.add_subplot(121, projection='3d')
    ax_2d = fig.add_subplot(122)
    plt.subplots_adjust(bottom=0.22, left=0.05, right=0.95)

    state = {
        'theta_deg': 0,
        'noise': 0.0,
    }

    lines_3d = []
    scatters_3d = []
    lines_2d = []
    scatters_2d = []

    def update_frame(frame_idx):
        frame_idx = int(frame_idx)
        pos_3d = sequence_3d[frame_idx]

        theta = np.deg2rad(state['theta_deg'])
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        pos_3d_rotated = np.matmul(pos_3d, Rz)

        pos_2d = pos_3d_rotated[..., [0, 2]].copy()
        pos_2d_norm = normalize_2d(pos_2d[np.newaxis, ...])[0]

        if state['noise'] > 0:
            pos_2d_norm += np.random.normal(0, state['noise'], pos_2d_norm.shape).astype(np.float32)

        while lines_3d:
            lines_3d.pop(0).remove()
        while scatters_3d:
            scatters_3d.pop(0).remove()
        while lines_2d:
            lines_2d.pop(0).remove()
        while scatters_2d:
            scatters_2d.pop(0).remove()

        sc_3d = ax_3d.scatter(pos_3d_rotated[:, 0], pos_3d_rotated[:, 1], pos_3d_rotated[:, 2],
                              c='black', s=20, alpha=0.6)
        scatters_3d.append(sc_3d)

        for name, info in SKELETON_GROUPS.items():
            for edge in info['edges']:
                line_pts_3d = pos_3d_rotated[list(edge), :]
                ln_3d, = ax_3d.plot(line_pts_3d[:, 0], line_pts_3d[:, 1], line_pts_3d[:, 2],
                                    color=info['color'], linewidth=2,
                                    label=(info['label'] if edge == info['edges'][0] else ""))
                lines_3d.append(ln_3d)

                line_pts_2d = pos_2d_norm[list(edge), :]
                ln_2d, = ax_2d.plot(line_pts_2d[:, 0], line_pts_2d[:, 1],
                                    color=info['color'], linewidth=2)
                lines_2d.append(ln_2d)

        sc_2d = ax_2d.scatter(pos_2d_norm[:, 0], pos_2d_norm[:, 1], c='black', s=30, alpha=0.7)
        scatters_2d.append(sc_2d)

        r = 1.0
        ax_3d.set_xlim(-r, r)
        ax_3d.set_ylim(-r, r)
        ax_3d.set_zlim(-r, r)
        ax_3d.set_xlabel('X')
        ax_3d.set_ylabel('Y')
        ax_3d.set_zlabel('Z')
        ax_3d.set_title(f"3D (Rotated, Z-up)\nAzimuth = {state['theta_deg']}°")

        ax_2d.set_xlim(-1.2, 1.2)
        ax_2d.set_ylim(-1.2, 1.2)
        ax_2d.set_aspect('equal')
        ax_2d.set_xlabel('u (= X after rotate)')
        ax_2d.set_ylabel('v (= Z after rotate)')
        ax_2d.set_title(f"2D Ortho Projection [X, Z]\n{subject} | {action} | Frame {frame_idx}/{total_frames}")
        ax_2d.grid(True, alpha=0.3)

        if frame_idx == 0 and not ax_3d.get_legend():
            ax_3d.legend(loc='upper left', prop={'size': 7})

    ax_slider = plt.axes([0.2, 0.12, 0.6, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, total_frames - 1, valinit=0, valstep=1)

    ax_theta = plt.axes([0.2, 0.07, 0.25, 0.03])
    slider_theta = Slider(ax_theta, 'Azimuth', 0, 360, valinit=state['theta_deg'], valstep=1)

    ax_noise = plt.axes([0.55, 0.07, 0.25, 0.03])
    slider_noise = Slider(ax_noise, 'Noise', 0, 0.05, valinit=state['noise'], valstep=0.001)

    anim_running = [False]
    anim_obj = [None]

    def on_frame_change(val):
        update_frame(val)
        fig.canvas.draw_idle()

    def on_theta_change(val):
        state['theta_deg'] = val
        update_frame(slider.val)
        fig.canvas.draw_idle()

    def on_noise_change(val):
        state['noise'] = val
        update_frame(slider.val)
        fig.canvas.draw_idle()

    slider.on_changed(on_frame_change)
    slider_theta.on_changed(on_theta_change)
    slider_noise.on_changed(on_noise_change)

    def animate_play(i):
        new_val = (slider.val + 1) % total_frames
        slider.set_val(new_val)
        return []

    def toggle_play(event):
        if anim_running[0]:
            anim_obj[0].event_source.stop()
            btn_play.label.set_text('Play')
        else:
            anim_obj[0] = animation.FuncAnimation(fig, animate_play, interval=50, cache_frame_data=False)
            btn_play.label.set_text('Pause')
        anim_running[0] = not anim_running[0]

    ax_play = plt.axes([0.88, 0.015, 0.08, 0.04])
    btn_play = Button(ax_play, 'Play')
    btn_play.on_clicked(toggle_play)

    update_frame(0)

    print("\n交互指南:")
    print("- 拖动 Frame 滑块浏览帧")
    print("- 拖动 Azimuth 滑块调整绕Z轴旋转角度")
    print("- 拖动 Noise 滑块模拟关键点检测噪声")
    print("- 点击 Play/Pause 自动播放动画")
    print("- 左侧:  绕Z轴旋转后的 3D 骨架")
    print("- 右侧:  丢弃Y轴后的正交投影 2D (X→u, Z→v)")

    plt.show()


if __name__ == '__main__':
    DATA_PATH = r'npz\real_npz\animals_test_3d.npz'
    SUB = "Addax_Male"
    ACT = "runtostandturnl"

    try:
        seq, sub, act = load_animal_data(DATA_PATH, SUB, ACT)
        create_interactive_vis(seq, sub, act)
    except Exception as e:
        print(f"运行失败: {e}")
