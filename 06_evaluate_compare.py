import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

sys.path.append('./common')
from common.loss import mpjpe

SKELETON_EDGES = [
    (0, 4), (4, 3), (3, 1), (3, 2), (4, 5), (5, 6), (6, 7),
    (4, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16)
]

MODEL_CFG = {
    'quadVideo3D': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_quadVideo3D_best.pt',
    },
    'poseformer': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_poseformer_best.pt',
    },
    'poseformerv2': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_poseformerv2_best.pt',
    },
    'videopose3d': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_videopose3d_best.pt',
    },
    'dstformer': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_dstformer_best.pt',
    },
    'dstformer_full': {
        'seq_len': 243,
        'ckpt': 'checkpoints/compare_dstformer_full_best.pt',
    },
    'motionagformer': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_motionagformer_best.pt',
    },
    'stcformer': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_stcformer_best.pt',
    },
    'mixste': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_mixste_best.pt',
    },
    'dtf': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_dtf_best.pt',
    },
    'graphmlp': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_graphmlp_best.pt',
    },
    'icfnet': {
        'seq_len': 27,
        'ckpt': 'checkpoints/compare_icfnet_best.pt',
    },
}


def build_eval_model(model_name, seq_len, device):
    if model_name == 'quadVideo3D':
        from common.quadVideo3D import QuadVideo3D
        return QuadVideo3D(
            num_frame=seq_len, num_joints=17, in_chans=2,
            embed_dim_ratio=32, depth=4, num_heads=8, mlp_ratio=2.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.2,
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
            num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
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
    elif model_name == 'dstformer_full':
        from common.MotionBERT.DSTformer import DSTformer
        return DSTformer(
            dim_in=2, dim_out=3, dim_feat=512, dim_rep=512,
            depth=5, num_heads=8, mlp_ratio=4,
            num_joints=17, maxlen=seq_len
        ).to(device)
    elif model_name == 'motionagformer':
        from common.MotionAGFormer.MotionAGFormer import MotionAGFormer
        return MotionAGFormer(
            n_layers=8, dim_in=2, dim_feat=256, dim_rep=512, dim_out=3,
            mlp_ratio=4, num_heads=8, num_joints=17, n_frames=seq_len
        ).to(device)
    elif model_name == 'stcformer':
        from common.STCFormer.stcformer import Model
        stc_args = argparse.Namespace(
            layers=8, d_hid=320, frames=seq_len,
            n_joints=17, out_joints=17
        )
        return Model(stc_args).to(device)
    elif model_name == 'mixste':
        from common.MixSTE.model_cross import MixSTE2
        return MixSTE2(
            num_frame=seq_len, num_joints=17, in_chans=2,
            embed_dim_ratio=512, depth=4, num_heads=8, mlp_ratio=2.,
            qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0.2
        ).to(device)
    elif model_name == 'dtf':
        from common.DTF.dtf import Model
        dtf_args = argparse.Namespace(
            layers=3, channel=512, d_hid=1024, frames=seq_len,
            n_joints=17, out_joints=17, in_chans=2
        )
        return Model(dtf_args).to(device)
    elif model_name == 'graphmlp':
        from common.GraphMLP.graphmlp import Model
        gmlp_args = argparse.Namespace(
            layers=11, channel=512, d_hid=1024, token_dim=256,
            frames=seq_len, n_joints=17
        )
        return Model(gmlp_args).to(device)
    elif model_name == 'icfnet':
        from common.ICFNet.trans import ICFNet
        icf_args = argparse.Namespace(
            layers=3, channel=256, d_hid=512, frames=seq_len,
            n_joints=17, out_joints=17
        )
        return ICFNet(icf_args).to(device)
    raise ValueError(f"Unknown: {model_name}")


def normalize_2d(pose_2d):
    max_vals = np.max(np.abs(pose_2d), axis=(1, 2), keepdims=True)
    max_vals[max_vals < 1e-5] = 1.0
    return pose_2d / max_vals


def calculate_species_scales(data_dict):
    species_scales = {}
    for animal_name in data_dict:
        bone_lengths = []
        for action in data_dict[animal_name].values():
            if len(action) > 0:
                for e in SKELETON_EDGES:
                    if e[0] < action.shape[1] and e[1] < action.shape[1]:
                        bone_vec = action[:, e[1]] - action[:, e[0]]
                        bone_lengths.extend(np.linalg.norm(bone_vec, axis=-1))
        species_scales[animal_name] = np.mean(bone_lengths) if bone_lengths else 1.0
    return species_scales


def calculate_torso_lengths(data_dict):
    torso_lengths = {}
    for animal_name in data_dict:
        lengths = []
        for action in data_dict[animal_name].values():
            if len(action) > 0 and action.shape[1] > 3:
                root_neck_vec = action[:, 3] - action[:, 0]
                lengths.extend(np.linalg.norm(root_neck_vec, axis=-1))
        torso_lengths[animal_name] = np.mean(lengths) if lengths else 1.0
    return torso_lengths


def batch_procrustes_np(pred_np, gt_np):
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
    return s * pred_np @ R + t


def compute_pa_mpjpe(pred, gt):
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()
    errors = []
    for i in range(pred_np.shape[0]):
        aligned = batch_procrustes_np(pred_np[i], gt_np[i])
        errors.append(np.mean(np.sqrt(np.sum((aligned - gt_np[i]) ** 2, axis=-1))))
    return np.mean(errors) * 1000


def compute_bone_lengths(kps, edges):
    lengths = []
    for s, e in edges:
        if s < kps.shape[1] and e < kps.shape[1]:
            lengths.append(torch.norm(kps[:, e] - kps[:, s], dim=-1))
    return torch.stack(lengths, dim=1)


def forward_inference(model, model_name, input_2d, device):
    return model(input_2d.to(device).contiguous())


def get_center_pred(pred, pred_seq_len):
    if pred.shape[1] == 1:
        return pred[:, 0, :, :]
    return pred[:, pred_seq_len // 2, :, :]


class Logger:
    def __init__(self):
        self.lines = []

    def log(self, msg):
        self.lines.append(msg)
        print(msg)

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))


def evaluate(model_name):
    cfg = MODEL_CFG[model_name]
    SEQ_LEN = cfg['seq_len']
    CKPT_PATH = cfg['ckpt']
    CENTER = SEQ_LEN // 2
    OUTPUT_TXT = f"evaluation_results/compare_{model_name}_eval.txt"

    log = Logger()
    log.log("=" * 70)
    log.log(f"{model_name} 评估")
    log.log(f"Checkpoint: {CKPT_PATH}")
    log.log("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = 'npz/real_npz/animals_val_3d.npz'
    train_data_path = 'npz/real_npz/animals_train_3d.npz'

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

    model = build_eval_model(model_name, SEQ_LEN, device)
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    VIEW_ANGLES = [0, np.pi / 2, np.pi, 3 * np.pi / 2]

    all_mpjpe = []
    all_pa_mpjpe = []
    all_n_mpjpe = []
    all_per_joint_errors = []
    all_per_joint_pa_errors = []
    all_per_joint_species = []
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

                    input_2d = torch.tensor(pos_2d_norm, dtype=torch.float32).unsqueeze(0)
                    pred = forward_inference(model, model_name, input_2d, device)
                    pred_center = get_center_pred(pred, SEQ_LEN)
                    pred_3d = (pred_center * scale_val).cpu()

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
                    all_per_joint_species.append(sub)

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

    test_t = np.random.randn(4, 17, 3).astype(np.float32) * 0.5
    self_check = 0.0
    for i in range(len(test_t)):
        aligned_np = batch_procrustes_np(test_t[i], test_t[i])
        self_check += np.mean(np.sqrt(np.sum((aligned_np - test_t[i]) ** 2, axis=-1)))
    self_check = (self_check / len(test_t)) * 1000
    log.log(f"\n[Procrustes自洽] {self_check:.6f} mm {'(OK)' if self_check < 0.1 else '(WARN)'}")

    log.log("\n" + "=" * 70)
    log.log("1. 核心精度")
    log.log("=" * 70)
    log.log(f"  MPJPE:       {np.mean(all_mpjpe):.2f} mm")
    log.log(f"  PA-MPJPE:    {np.mean(all_pa_mpjpe):.2f} mm")
    log.log(f"  TN-MPJPE:    {np.mean(all_n_mpjpe):.2f}%")

    thresholds = [50, 100, 150, 200]
    log.log("\n" + "=" * 70)
    log.log("2a. PCK / CPS (固定阈值)")
    log.log("=" * 70)
    for th in thresholds:
        pck = np.mean(per_joint_errors < th) * 100
        pa_pck = np.mean(per_joint_pa_errors < th) * 100
        cps = np.mean(np.all(per_joint_errors < th, axis=1)) * 100
        pa_cps = np.mean(np.all(per_joint_pa_errors < th, axis=1)) * 100
        log.log(f"  阈值 {th:3d}mm | PCK: {pck:6.2f}% | PA-PCK: {pa_pck:6.2f}% | CPS: {cps:6.2f}% | PA-CPS: {pa_cps:6.2f}%")
    
    species_arr = np.array(all_per_joint_species)
    rp_thresholds_pct = [3, 5, 10]
    log.log("\n" + "=" * 70)
    log.log("2b. RP-PCK / RP-CPS (躯干长度百分比阈值)")
    log.log("=" * 70)
    for pct in rp_thresholds_pct:
        thresholds_mm = np.array([torso_lengths[s] * 1000 * pct / 100 for s in species_arr])
        rp_pck = np.mean(per_joint_errors < thresholds_mm[:, np.newaxis]) * 100
        rp_pa_pck = np.mean(per_joint_pa_errors < thresholds_mm[:, np.newaxis]) * 100
        rp_cps = np.mean(np.all(per_joint_errors < thresholds_mm[:, np.newaxis], axis=1)) * 100
        rp_pa_cps = np.mean(np.all(per_joint_pa_errors < thresholds_mm[:, np.newaxis], axis=1)) * 100
        log.log(f"  阈值 {pct:2d}% | RP-PCK: {rp_pck:6.2f}% | RP-PA-PCK: {rp_pa_pck:6.2f}% | RP-CPS: {rp_cps:6.2f}% | RP-PA-CPS: {rp_pa_cps:6.2f}%")

    log.log("\n" + "=" * 70)
    log.log("3. 骨骼长度误差 (BLE)")
    log.log("=" * 70)
    all_bone = np.concatenate([b.flatten() for b in all_bone_errors])
    log.log(f"  Mean BLE: {np.mean(all_bone):.2f} mm")
    log.log(f"  Median BLE: {np.median(all_bone):.2f} mm")

    log.log("\n" + "=" * 70)
    log.log("4. 逐物种精度")
    log.log("=" * 70)
    log.log(f"  {'物种':<25s} {'样本':>6s} {'MPJPE':>8s} {'PA-MPJPE':>10s} {'TN-MPJPE':>10s} {'BLE':>8s} {'RP-PCK@3':>10s} {'RP-PA-PCK@3':>12s}")
    log.log(f"  {'-' * 89}")
    species_arr_full = np.array(all_per_joint_species)
    for sub in sorted(species_mpjpe.keys()):
        n = len(species_mpjpe[sub])
        mpjpe_s = np.mean(species_mpjpe[sub])
        pa_s = np.mean(species_pa_mpjpe[sub])
        n_s = np.mean(species_n_mpjpe[sub])
        ble_s = np.mean(species_bone_errors[sub])
        
        # Per-species RP-PCK@3
        mask = species_arr_full == sub
        torso_m = torso_lengths.get(sub, 1.0)
        thresh_mm = torso_m * 1000 * 3 / 100
        species_errs = per_joint_errors[mask]
        species_pa_errs = per_joint_pa_errors[mask]
        rp_pck_s = np.mean(species_errs < thresh_mm) * 100 if len(species_errs) > 0 else 0.0
        rp_pa_pck_s = np.mean(species_pa_errs < thresh_mm) * 100 if len(species_pa_errs) > 0 else 0.0
        
        log.log(f"  {sub:<25s} {n:>6d} {mpjpe_s:>7.1f}mm {pa_s:>9.1f}mm {n_s:>8.1f}%  {ble_s:>7.1f}mm {rp_pck_s:>9.1f}% {rp_pa_pck_s:>11.1f}%")

    log.log("\n" + "=" * 70)
    log.log("5. Per-Joint Error (mm)")
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
    os.makedirs('evaluation_results', exist_ok=True)
    log.save(OUTPUT_TXT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="icfnet",
                        choices=list(MODEL_CFG.keys()),
                        help='Model to evaluate')
    args = parser.parse_args()
    evaluate(args.model)
