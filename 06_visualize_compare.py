import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from tqdm import tqdm

sys.path.append('./common')

SKELETON_EDGES = [
    (0, 4), (4, 3), (3, 1), (3, 2), (4, 5), (5, 6), (6, 7),
    (4, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16)
]

EDGE_COLORS_3D = [
    'black', 'black', 'black', 'black',
    'red', 'red', 'red',
    'orange', 'orange', 'orange',
    'blue', 'blue', 'blue',
    'cyan', 'cyan', 'cyan',
]

EDGE_COLORS_2D = [
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (1, 0, 0), (1, 0, 0), (1, 0, 0),
    (1, 0.65, 0), (1, 0.65, 0), (1, 0.65, 0),
    (0, 0, 1), (0, 0, 1), (0, 0, 1),
    (0, 1, 1), (0, 1, 1), (0, 1, 1),
]

MODEL_CFG = {
    'animalposeformer': {'seq_len': 27, 'ckpt': 'checkpoints/animal_poseformer_best_model.pt'},
    'poseformer':        {'seq_len': 27, 'ckpt': 'checkpoints/compare_poseformer_best.pt'},
    'poseformerv2':      {'seq_len': 27, 'ckpt': 'checkpoints/compare_poseformerv2_best.pt'},
    'videopose3d':       {'seq_len': 27, 'ckpt': 'checkpoints/compare_videopose3d_best.pt'},
    'dstformer':         {'seq_len': 27, 'ckpt': 'checkpoints/compare_dstformer_best.pt'},
    'mixste':            {'seq_len': 27, 'ckpt': 'checkpoints/compare_mixste_best.pt'},
    'stcformer':         {'seq_len': 27, 'ckpt': 'checkpoints/compare_stcformer_best.pt'},
}


def build_model(model_name, seq_len, device):
    if model_name == 'animalposeformer':
        from common.animal_poseformer import AnimalPoseFormer
        return AnimalPoseFormer(
            num_frame=seq_len, num_joints=17, in_chans=2,
            embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.2, use_lme=True,
            num_frame_kept=seq_len, num_coeff_kept=seq_len
        ).to(device)
    elif model_name == 'poseformer':
        from common.poseformer.model_poseformer import PoseTransformer
        return PoseTransformer(
            num_frame=seq_len, num_joints=17, in_chans=2,
            embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.2
        ).to(device)
    elif model_name == 'poseformerv2':
        from common.poseformerv2.model_poseformerV2 import PoseTransformerV2
        args_ns = argparse.Namespace(
            embed_dim_ratio=32, depth=4,
            number_of_kept_frames=seq_len, number_of_kept_coeffs=seq_len
        )
        return PoseTransformerV2(
            num_frame=seq_len, num_joints=17, in_chans=2,
            num_heads=8, mlp_ratio=2., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, args=args_ns
        ).to(device)
    elif model_name == 'videopose3d':
        from common.videopose3D.model import TemporalModel
        return TemporalModel(
            num_joints_in=17, in_features=2, num_joints_out=17,
            filter_widths=[3, 3, 3], dropout=0.25, channels=1024
        ).to(device)
    elif model_name == 'dstformer':
        from common.MotionBERT.DSTformer import DSTformer
        return DSTformer(
            dim_in=2, dim_out=3, dim_feat=448, depth=4, num_heads=8,
            mlp_ratio=2, num_joints=17, maxlen=seq_len
        ).to(device)
    elif model_name == 'mixste':
        from common.MixSTE.model_cross import MixSTE2
        return MixSTE2(
            num_frame=seq_len, num_joints=17, in_chans=2,
            embed_dim_ratio=512, depth=4, num_heads=8, mlp_ratio=2.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.2
        ).to(device)
    elif model_name == 'stcformer':
        from common.STCFormer.stcformer import Model
        stc_args = argparse.Namespace(
            layers=8, d_hid=320, frames=seq_len,
            n_joints=17, out_joints=17
        )
        return Model(stc_args).to(device)
    raise ValueError(f"Unknown: {model_name}")


def normalize_2d(pose_2d):
    max_vals = np.max(np.abs(pose_2d), axis=(1, 2), keepdims=True)
    max_vals[max_vals < 1e-5] = 1.0
    return pose_2d / max_vals


def calculate_species_scales(data_dict, species_to_id):
    species_scales = {}
    for animal_name, animal_id in species_to_id.items():
        if animal_name in data_dict:
            bone_lengths = []
            for action in data_dict[animal_name].values():
                if len(action) > 0:
                    for e in SKELETON_EDGES:
                        if e[0] < action.shape[1] and e[1] < action.shape[1]:
                            bone_vec = action[:, e[1]] - action[:, e[0]]
                            bone_lengths.extend(np.linalg.norm(bone_vec, axis=-1))
            species_scales[animal_id] = np.mean(bone_lengths) if bone_lengths else 1.0
        else:
            species_scales[animal_id] = 1.0
    return species_scales


def infer_full_sequence(model, model_name, pos_2d_norm, seq_len, device):
    num_frames = len(pos_2d_norm)
    pad_len = seq_len // 2
    padded = np.pad(pos_2d_norm, ((pad_len, pad_len), (0, 0), (0, 0)), mode='edge')

    windows = []
    for i in range(num_frames):
        w = padded[i:i + seq_len]
        w = np.nan_to_num(w, 0.0)
        windows.append(w)
    windows = np.array(windows)

    batch_t = torch.tensor(windows, dtype=torch.float32).to(device).contiguous()
    with torch.no_grad():
        pred = model(batch_t)

    if pred.shape[1] == 1:
        pred_3d_norm = pred[:, 0, :, :].cpu().numpy()
    else:
        center = pred.shape[1] // 2
        pred_3d_norm = pred[:, center, :, :].cpu().numpy()

    return pred_3d_norm


def draw_2d_skeleton(ax, x, y, edges, colors, title):
    ax.clear()
    ax.scatter(x, y, c='green', s=20, zorder=3)
    for (s, e), col in zip(edges, colors):
        ax.plot([x[s], x[e]], [y[s], y[e]], color=col, linewidth=1.5, zorder=2)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.axis('off')


def draw_3d_skeleton(ax, kps, edges, colors, title, fixed_lim=None):
    ax.clear()
    ax.scatter(kps[:, 0], kps[:, 1], kps[:, 2], c='green', s=15)
    for (s, e), col in zip(edges, colors):
        ax.plot([kps[s, 0], kps[e, 0]],
                [kps[s, 1], kps[e, 1]],
                [kps[s, 2], kps[e, 2]], color=col, linewidth=1.5)
    if fixed_lim is not None:
        ax.set_xlim(-fixed_lim, fixed_lim)
        ax.set_ylim(-fixed_lim, fixed_lim)
        ax.set_zlim(-fixed_lim, fixed_lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=11)


def compute_mpjpe(pred, gt):
    pred_root = pred[0:1, :]
    gt_root = gt[0:1, :]
    pred_rel = pred - pred_root
    gt_rel = gt - gt_root
    return float(np.mean(np.sqrt(np.sum((pred_rel - gt_rel) ** 2, axis=-1))))


def prepare_sequence(data_dict, sub, act, theta, seq_len):
    pos_3d_raw = data_dict[sub][act]

    n_frames = len(pos_3d_raw)
    if n_frames >= seq_len:
        start = (n_frames - seq_len) // 2
        pos_3d = pos_3d_raw[start:start + seq_len]
    else:
        pad_len = seq_len - n_frames
        pos_3d = np.pad(pos_3d_raw, ((0, pad_len), (0, 0), (0, 0)), mode='edge')

    pos_3d = pos_3d - pos_3d[:, 0:1, :]

    c, s = np.cos(theta), np.sin(theta)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    pos_3d_rot = np.matmul(pos_3d, Rz)
    pos_2d = pos_3d_rot[..., [0, 2]]
    pos_2d_norm = normalize_2d(pos_2d)

    return pos_2d_norm.astype(np.float32), pos_3d_rot.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='animalposeformer',
                        choices=list(MODEL_CFG.keys()))
    args = parser.parse_args()

    cfg = MODEL_CFG[args.model]
    SEQ_LEN = cfg['seq_len']
    CKPT = cfg['ckpt']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"模型: {args.model}, seq_len={SEQ_LEN}")
    print(f"Checkpoint: {CKPT}")

    val_data = np.load('npz/real_npz/animals_val_3d.npz', allow_pickle=True)['positions_3d'].item()
    train_data = np.load('npz/real_npz/animals_train_3d.npz', allow_pickle=True)['positions_3d'].item()

    subjects = sorted(train_data.keys())
    species_to_id = {s: i for i, s in enumerate(subjects)}
    scales_dict = calculate_species_scales(train_data, species_to_id)

    v_subjects = sorted(val_data.keys())

    model = build_model(args.model, SEQ_LEN, device)
    state_dict = torch.load(CKPT, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    subject_actions = {}
    for sub in v_subjects:
        subject_actions[sub] = sorted(val_data[sub].keys())

    state = {
        'sub': v_subjects[0],
        'act': subject_actions[v_subjects[0]][0],
        'view_idx': 0,
        'frame': 0,
        'paused': False,
        'pos_2d_norm': None,
        'pos_3d_rot': None,
        'pred_3d': None,
        'fixed_limit': None,
    }

    def load_sequence(sub, act, theta):
        pos_2d_norm, pos_3d_rot = prepare_sequence(val_data, sub, act, theta, SEQ_LEN)
        species_id = species_to_id[sub]
        scale = scales_dict.get(species_id, 1.0)

        pred_3d_norm = infer_full_sequence(model, args.model, pos_2d_norm, SEQ_LEN, device)
        pred_3d = pred_3d_norm * scale

        all_kps = np.concatenate([pos_3d_rot, pred_3d], axis=0)
        lim = np.max(np.abs(all_kps)) * 1.2

        return pos_2d_norm, pos_3d_rot, pred_3d, lim

    views = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    view_labels = ['0° (正面)', '90° (侧面)', '180° (背面)', '270° (另一侧)']

    fig = plt.figure(figsize=(20, 7))
    ax_2d = fig.add_subplot(131)
    ax_pred = fig.add_subplot(132, projection='3d')
    ax_gt = fig.add_subplot(133, projection='3d')

    ax_pred.view_init(elev=20, azim=-90)
    ax_gt.view_init(elev=20, azim=-90)

    species_idx = [0]
    action_idx = [0]
    view_idx = [0]

    ax_spec_prev = plt.axes([0.005, 0.94, 0.04, 0.035])
    ax_spec_next = plt.axes([0.17, 0.94, 0.04, 0.035])
    btn_spec_prev = Button(ax_spec_prev, '◀', color='0.9')
    btn_spec_next = Button(ax_spec_next, '▶', color='0.9')

    ax_act_prev = plt.axes([0.005, 0.90, 0.04, 0.035])
    ax_act_next = plt.axes([0.17, 0.90, 0.04, 0.035])
    btn_act_prev = Button(ax_act_prev, '◀', color='0.9')
    btn_act_next = Button(ax_act_next, '▶', color='0.9')

    ax_vw_prev = plt.axes([0.005, 0.86, 0.04, 0.035])
    ax_vw_next = plt.axes([0.17, 0.86, 0.04, 0.035])
    btn_vw_prev = Button(ax_vw_prev, '◀', color='0.9')
    btn_vw_next = Button(ax_vw_next, '▶', color='0.9')

    ax_spec_label = plt.axes([0.045, 0.94, 0.125, 0.035])
    ax_act_label = plt.axes([0.045, 0.90, 0.125, 0.035])
    ax_vw_label = plt.axes([0.045, 0.86, 0.125, 0.035])

    ax_spec_label.set_xticks([])
    ax_spec_label.set_yticks([])
    ax_act_label.set_xticks([])
    ax_act_label.set_yticks([])
    ax_vw_label.set_xticks([])
    ax_vw_label.set_yticks([])

    def update_labels():
        ax_spec_label.clear()
        ax_spec_label.set_xticks([])
        ax_spec_label.set_yticks([])
        ax_spec_label.text(0.5, 0.5, state['sub'], transform=ax_spec_label.transAxes,
                          ha='center', va='center', fontsize=7, fontweight='bold')
        acts = subject_actions[state['sub']]
        ax_act_label.clear()
        ax_act_label.set_xticks([])
        ax_act_label.set_yticks([])
        ax_act_label.text(0.5, 0.5, state['act'], transform=ax_act_label.transAxes,
                         ha='center', va='center', fontsize=7)

        view_names = view_labels
        ax_vw_label.clear()
        ax_vw_label.set_xticks([])
        ax_vw_label.set_yticks([])
        ax_vw_label.text(0.5, 0.5, view_names[state['view_idx']], transform=ax_vw_label.transAxes,
                        ha='center', va='center', fontsize=7, color='darkblue')

        for spine in ax_spec_label.spines.values():
            spine.set_color('gray')
            spine.set_linewidth(0.5)
        for spine in ax_act_label.spines.values():
            spine.set_color('gray')
            spine.set_linewidth(0.5)
        for spine in ax_vw_label.spines.values():
            spine.set_color('gray')
            spine.set_linewidth(0.5)

    def on_spec_prev(e):
        sub = state['sub']
        idx = v_subjects.index(sub)
        idx = (idx - 1) % len(v_subjects)
        state['sub'] = v_subjects[idx]
        state['act'] = subject_actions[state['sub']][0]
        state['frame'] = 0
        update_labels()
        reload_data()
        fig.canvas.draw_idle()

    def on_spec_next(e):
        sub = state['sub']
        idx = v_subjects.index(sub)
        idx = (idx + 1) % len(v_subjects)
        state['sub'] = v_subjects[idx]
        state['act'] = subject_actions[state['sub']][0]
        state['frame'] = 0
        update_labels()
        reload_data()
        fig.canvas.draw_idle()

    def on_act_prev(e):
        acts = subject_actions[state['sub']]
        idx = acts.index(state['act'])
        idx = (idx - 1) % len(acts)
        state['act'] = acts[idx]
        state['frame'] = 0
        update_labels()
        reload_data()
        fig.canvas.draw_idle()

    def on_act_next(e):
        acts = subject_actions[state['sub']]
        idx = acts.index(state['act'])
        idx = (idx + 1) % len(acts)
        state['act'] = acts[idx]
        state['frame'] = 0
        update_labels()
        reload_data()
        fig.canvas.draw_idle()

    def on_vw_prev(e):
        state['view_idx'] = (state['view_idx'] - 1) % 4
        state['frame'] = 0
        update_labels()
        reload_data()
        fig.canvas.draw_idle()

    def on_vw_next(e):
        state['view_idx'] = (state['view_idx'] + 1) % 4
        state['frame'] = 0
        update_labels()
        reload_data()
        fig.canvas.draw_idle()

    btn_spec_prev.on_clicked(on_spec_prev)
    btn_spec_next.on_clicked(on_spec_next)
    btn_act_prev.on_clicked(on_act_prev)
    btn_act_next.on_clicked(on_act_next)
    btn_vw_prev.on_clicked(on_vw_prev)
    btn_vw_next.on_clicked(on_vw_next)

    update_labels()

    def reload_data():
        theta = views[state['view_idx']]
        pos_2d_norm, pos_3d_rot, pred_3d, lim = load_sequence(
            state['sub'], state['act'], theta
        )
        state['pos_2d_norm'] = pos_2d_norm
        state['pos_3d_rot'] = pos_3d_rot
        state['pred_3d'] = pred_3d
        state['fixed_limit'] = lim
        slider_frame.valmax = SEQ_LEN - 1
        slider_frame.set_val(0)

    ax_pause = plt.axes([0.25, 0.02, 0.1, 0.04])
    btn_pause = Button(ax_pause, 'pause', color='lightblue', hovercolor='0.975')

    def toggle_pause(event):
        state['paused'] = not state['paused']
        btn_pause.label.set_text('continue' if state['paused'] else 'pause')
        btn_pause.color = 'lightcoral' if state['paused'] else 'lightblue'

    btn_pause.on_clicked(toggle_pause)

    ax_slider = plt.axes([0.38, 0.02, 0.4, 0.04])
    slider_frame = Slider(ax_slider, 'frame', 0, SEQ_LEN - 1,
                          valinit=0, valfmt='%d', valstep=1)

    reload_data()

    def render(f):
        idx = int(slider_frame.val) if state['paused'] else f % SEQ_LEN

        if state['pos_2d_norm'] is not None and idx < len(state['pos_2d_norm']):
            x = state['pos_2d_norm'][idx, :, 0]
            y = state['pos_2d_norm'][idx, :, 1]
            draw_2d_skeleton(ax_2d, x, y, SKELETON_EDGES, EDGE_COLORS_2D,
                             f"2D Input\n{state['sub']} | {state['act']}")

        if state['pred_3d'] is not None and idx < len(state['pred_3d']):
            draw_3d_skeleton(ax_pred, state['pred_3d'][idx], SKELETON_EDGES, EDGE_COLORS_3D,
                             f"Predicted 3D\nMPJPE={compute_mpjpe(state['pred_3d'][idx], state['pos_3d_rot'][idx]) * 1000:.1f}mm",
                             fixed_lim=state['fixed_limit'])

        if state['pos_3d_rot'] is not None and idx < len(state['pos_3d_rot']):
            draw_3d_skeleton(ax_gt, state['pos_3d_rot'][idx], SKELETON_EDGES, EDGE_COLORS_3D,
                             "Ground Truth 3D",
                             fixed_lim=state['fixed_limit'])
            ax_gt.view_init(elev=ax_pred.elev, azim=ax_pred.azim)

        if not state['paused']:
            slider_frame.set_val(idx)

        return []

    ani = animation.FuncAnimation(fig, render, frames=SEQ_LEN,
                                   interval=80, blit=False, repeat=True,
                                   cache_frame_data=False)

    model_label = f"model: {args.model}  |  checkpoint: {os.path.basename(CKPT)}"
    fig.text(0.25, 0.96, model_label, fontsize=9, color='gray')

    plt.show()


if __name__ == '__main__':
    main()
