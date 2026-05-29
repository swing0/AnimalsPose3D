import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

sys.path.append('./common')
from common.loss import mpjpe
from common.quadVideo3D import QuadVideo3D

SKELETON_EDGES = [
    (0, 4), (4, 3), (3, 1), (3, 2), (4, 5), (5, 6), (6, 7),
    (4, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16)
]


def normalize_2d(pose_2d):
    max_vals = np.max(np.abs(pose_2d), axis=(1, 2), keepdims=True)
    max_vals[max_vals < 1e-5] = 1.0
    return pose_2d / max_vals


def calculate_species_scales(data_dict):
    species_scales = {}
    for animal_name in data_dict:
        bone_lengths = []
        for action in data_dict[animal_name]:
            positions = data_dict[animal_name][action]
            if len(positions) > 0:
                for e in SKELETON_EDGES:
                    if e[0] < positions.shape[1] and e[1] < positions.shape[1]:
                        bone_vec = positions[:, e[1]] - positions[:, e[0]]
                        bone_len = np.linalg.norm(bone_vec, axis=-1)
                        bone_lengths.extend(bone_len)
        species_scales[animal_name] = np.mean(bone_lengths) if bone_lengths else 1.0
    return species_scales


def calculate_torso_lengths(data_dict):
    torso_lengths = {}
    for animal_name in data_dict:
        lengths = []
        for action in data_dict[animal_name]:
            positions = data_dict[animal_name][action]
            if len(positions) > 0 and positions.shape[1] > 3:
                root_neck_vec = positions[:, 3] - positions[:, 0]
                lengths.extend(np.linalg.norm(root_neck_vec, axis=-1))
        torso_lengths[animal_name] = np.mean(lengths) if lengths else 1.0
    return torso_lengths


def batch_rigid_align(pred, gt):
    """刚性对齐: rotation + translation only. pred, gt: (B, J, 3)"""
    mu_gt = gt.mean(dim=1, keepdim=True)
    mu_pred = pred.mean(dim=1, keepdim=True)
    gt_c = gt - mu_gt
    pred_c = pred - mu_pred

    H = torch.bmm(pred_c.transpose(1, 2), gt_c)
    U, _, Vt = torch.svd(H)
    V = Vt.transpose(1, 2)
    R = torch.bmm(V, U.transpose(1, 2))

    det = torch.det(R)
    diag = torch.ones(pred.shape[0], 3, device=pred.device)
    diag[:, 2] = torch.sign(det)
    R = torch.bmm(torch.bmm(V, torch.diag_embed(diag)), U.transpose(1, 2))

    aligned = torch.bmm(pred_c, R.transpose(1, 2)) + mu_gt
    return aligned


def batch_procrustes_np(pred_np, gt_np):
    """Procrustes alignment using numpy (stable). pred_np, gt_np: (J, 3)"""
    mu_pred = pred_np.mean(axis=0, keepdims=True)
    mu_gt = gt_np.mean(axis=0, keepdims=True)
    pred_c = pred_np - mu_pred
    gt_c = gt_np - mu_gt

    norm_pred = np.sqrt(np.sum(pred_c ** 2) + 1e-8)
    norm_gt = np.sqrt(np.sum(gt_c ** 2) + 1e-8)
    pred_n = pred_c / norm_pred
    gt_n = gt_c / norm_gt

    H = gt_n.T @ pred_n
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T

    if np.linalg.det(R) < 0:
        V[:, -1] *= -1
        R = V @ U.T

    s = np.sum(S) * norm_gt / norm_pred
    t = mu_gt - s * mu_pred @ R
    aligned = s * pred_np @ R + t
    return aligned


def compute_pa_mpjpe(pred, gt):
    """Compute PA-MPJPE for a batch using numpy Procrustes."""
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()
    errors = []
    for i in range(pred_np.shape[0]):
        aligned = batch_procrustes_np(pred_np[i], gt_np[i])
        err = np.sqrt(np.sum((aligned - gt_np[i]) ** 2, axis=-1))
        errors.append(np.mean(err))
    return np.mean(errors) * 1000


def compute_bone_lengths(kps, edges):
    lengths = []
    for s, e in edges:
        if s < kps.shape[1] and e < kps.shape[1]:
            lengths.append(torch.norm(kps[:, e] - kps[:, s], dim=-1))
    return torch.stack(lengths, dim=1)


class Logger:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)
        print(msg)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
        print(f"\n[结果已保存至] {path}")


def evaluate(checkpoint_path, data_path, train_data_path, device, output_txt=None):
    log = Logger()

    log.log("=" * 70)
    log.log("QuadVideo3D 评估")
    log.log(f"Checkpoint: {checkpoint_path}")
    log.log(f"数据集: {data_path}")
    log.log("=" * 70)

    data_dict = np.load(data_path, allow_pickle=True)['positions_3d'].item()
    train_data = np.load(train_data_path, allow_pickle=True)['positions_3d'].item()
    species_scales = calculate_species_scales(train_data)
    torso_lengths = calculate_torso_lengths(train_data)
    for name in data_dict:
        if name not in species_scales:
            species_scales[name] = 1.0
        if name not in torso_lengths:
            torso_lengths[name] = 1.0

    subjects = sorted(data_dict.keys())
    log.log(f"物种: {len(subjects)}, 动作总数: {sum(len(v) for v in data_dict.values())}")

    model = QuadVideo3D(
        num_frame=27, num_joints=17, in_chans=2,
        embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2.,
        qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=0.2, use_lme=True, num_frame_kept=27, num_coeff_kept=27
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    VIEW_ANGLES = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    SEQ_LEN = 27
    CENTER = SEQ_LEN // 2
    BATCH_SIZE = 32

    all_mpjpe = []
    all_pa_mpjpe = []
    all_n_mpjpe = []
    all_per_joint_errors = []
    all_per_joint_pa_errors = []
    all_bone_errors = []
    species_mpjpe = defaultdict(list)
    species_pa_mpjpe = defaultdict(list)
    species_n_mpjpe = defaultdict(list)
    species_bone_errors = defaultdict(list)

    with torch.no_grad():
        for sub in tqdm(subjects, desc="物种", ncols=80):
            scale_val = species_scales.get(sub, 1.0)
            for act in data_dict[sub]:
                pos_3d_raw = data_dict[sub][act]
                pos_3d_raw = pos_3d_raw - pos_3d_raw[:, 0:1, :]

                n_frames = len(pos_3d_raw)
                if n_frames >= SEQ_LEN:
                    start = (n_frames - SEQ_LEN) // 2
                    pos_3d = pos_3d_raw[start:start + SEQ_LEN]
                else:
                    pad = SEQ_LEN - n_frames
                    pos_3d = np.pad(pos_3d_raw, ((0, pad), (0, 0), (0, 0)), mode='edge')

                for theta in VIEW_ANGLES:
                    c, s = np.cos(theta), np.sin(theta)
                    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
                    pos_3d_rot = np.matmul(pos_3d, Rz)
                    pos_2d = pos_3d_rot[..., [0, 2]]
                    pos_2d_norm = normalize_2d(pos_2d)

                    input_2d = torch.tensor(pos_2d_norm, dtype=torch.float32).unsqueeze(0).to(device)
                    pred_norm = model(input_2d)
                    pred_3d = (pred_norm[:, 0] * scale_val).cpu()

                    gt_3d_rot = torch.tensor(pos_3d_rot[CENTER:CENTER + 1], dtype=torch.float32)

                    root_pred = pred_3d[:, 0:1, :]
                    root_gt = gt_3d_rot[:, 0:1, :]
                    pred_rootrel = pred_3d - root_pred
                    gt_rootrel = gt_3d_rot - root_gt

                    mpjpe_val = mpjpe(pred_rootrel, gt_rootrel).item() * 1000
                    all_mpjpe.append(mpjpe_val)
                    species_mpjpe[sub].append(mpjpe_val)

                    torso_len = torso_lengths.get(sub, 1.0)
                    n_mpjpe_val = (mpjpe_val / 1000) / torso_len * 100
                    all_n_mpjpe.append(n_mpjpe_val)
                    species_n_mpjpe[sub].append(n_mpjpe_val)

                    pa_val = compute_pa_mpjpe(pred_3d, gt_3d_rot)
                    all_pa_mpjpe.append(pa_val)
                    species_pa_mpjpe[sub].append(pa_val)

                    per_joint_err = torch.norm(pred_rootrel - gt_rootrel, dim=-1).squeeze(0) * 1000
                    all_per_joint_errors.append(per_joint_err.numpy())

                    pred_np = pred_3d.cpu().numpy()[0]
                    gt_np = gt_3d_rot.cpu().numpy()[0]
                    aligned_np = batch_procrustes_np(pred_np, gt_np)
                    per_joint_pa_err = np.sqrt(np.sum((aligned_np - gt_np) ** 2, axis=-1)) * 1000
                    all_per_joint_pa_errors.append(per_joint_pa_err)

                    pred_bones = compute_bone_lengths(pred_rootrel, SKELETON_EDGES)
                    gt_bones = compute_bone_lengths(gt_rootrel, SKELETON_EDGES)
                    bone_err = (torch.abs(pred_bones - gt_bones) * 1000).squeeze(0).numpy()
                    all_bone_errors.append(bone_err)
                    species_bone_errors[sub].append(np.mean(bone_err))

    per_joint_errors = np.array(all_per_joint_errors)
    per_joint_pa_errors = np.array(all_per_joint_pa_errors)

    # Procrustes 自洽验证 (numpy)
    test_t = np.random.randn(4, 17, 3).astype(np.float32) * 0.5
    self_check = 0.0
    for i in range(len(test_t)):
        aligned_np = batch_procrustes_np(test_t[i], test_t[i])
        self_check += np.mean(np.sqrt(np.sum((aligned_np - test_t[i]) ** 2, axis=-1)))
    self_check = (self_check / len(test_t)) * 1000
    log.log(f"\n[Procrustes 自洽验证] pred==gt 误差: {self_check:.6f} mm {'(OK)' if self_check < 0.1 else '(WARN)'}")

    log.log("\n" + "=" * 70)
    log.log("1. 核心精度指标")
    log.log("=" * 70)
    log.log(f"  MPJPE:       {np.mean(all_mpjpe):.2f} mm")
    log.log(f"  PA-MPJPE:    {np.mean(all_pa_mpjpe):.2f} mm")
    log.log(f"  N-MPJPE:     {np.mean(all_n_mpjpe):.2f}%")

    thresholds = [50, 100, 150, 200]
    log.log("\n" + "=" * 70)
    log.log("2. 鲁棒性指标 — PCK / CPS")
    log.log("=" * 70)
    for th in thresholds:
        pck = np.mean(per_joint_errors < th) * 100
        pa_pck = np.mean(per_joint_pa_errors < th) * 100
        cps = np.mean(np.all(per_joint_errors < th, axis=1)) * 100
        pa_cps = np.mean(np.all(per_joint_pa_errors < th, axis=1)) * 100
        log.log(f"  阈值 {th:3d}mm | PCK: {pck:6.2f}% | PA-PCK: {pa_pck:6.2f}% | CPS: {cps:6.2f}% | PA-CPS: {pa_cps:6.2f}%")

    log.log("\n" + "=" * 70)
    log.log("3. 骨骼长度误差 (BLE)")
    log.log("=" * 70)
    all_bone = np.concatenate([b.flatten() for b in all_bone_errors])
    log.log(f"  Mean BLE: {np.mean(all_bone):.2f} mm")
    log.log(f"  Median BLE: {np.median(all_bone):.2f} mm")

    log.log("\n" + "=" * 70)
    log.log("4. 逐物种精度")
    log.log("=" * 70)
    log.log(f"  {'物种':<25s} {'样本':>6s} {'MPJPE':>8s} {'PA-MPJPE':>10s} {'N-MPJPE':>10s} {'BLE':>8s}")
    log.log(f"  {'-' * 69}")
    for sub in sorted(species_mpjpe.keys()):
        n = len(species_mpjpe[sub])
        mpjpe_s = np.mean(species_mpjpe[sub])
        pa_s = np.mean(species_pa_mpjpe[sub])
        n_s = np.mean(species_n_mpjpe[sub])
        ble_s = np.mean(species_bone_errors[sub])
        log.log(f"  {sub:<25s} {n:>6d} {mpjpe_s:>7.1f}mm {pa_s:>9.1f}mm {n_s:>8.1f}%  {ble_s:>7.1f}mm")

    log.log("\n" + "=" * 70)
    log.log("5. Per-Joint Error (MPJPE per joint, mm)")
    log.log("=" * 70)
    joint_names = [
        "Root(0)", "Hip(1)", "Hip(2)", "Neck(3)", "Head(4)",
        "LShoulder(5)", "LElbow(6)", "LPaw(7)",
        "RShoulder(8)", "RElbow(9)", "RPaw(10)",
        "LHip(11)", "LKnee(12)", "LFoot(13)",
        "RHip(14)", "RKnee(15)", "RFoot(16)"
    ]
    row_fmt = "  {:>5s}  {:>20s}  {:>8s}  {:>8s}"
    log.log(row_fmt.format("Idx", "Name", "MPJPE", "PA-MPJPE"))
    log.log("  " + "-" * 48)
    for j in range(17):
        name = joint_names[j] if j < len(joint_names) else f"Joint({j})"
        e1 = np.mean(per_joint_errors[:, j])
        e2 = np.mean(per_joint_pa_errors[:, j])
        log.log(row_fmt.format(f"{j:02d}", name, f"{e1:.1f}mm", f"{e2:.1f}mm"))

    log.log("\n" + "=" * 70)
    log.log("评估完成")
    log.log("=" * 70)

    if output_txt:
        log.save(output_txt)


if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    evaluate(
        checkpoint_path='checkpoints/quadVideo3D_best_model.pt',
        data_path='npz/real_npz/animals_val_3d.npz',
        train_data_path='npz/real_npz/animals_train_3d.npz',
        device=device,
        output_txt='evaluation_results/quadVideo3D_best_model_eval.txt',
    )
