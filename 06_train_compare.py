import os
import sys
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append('./common')

from common.loss import mpjpe, compute_bone_loss, compute_symmetry_loss

SKELETON_EDGES = [
    (0, 4), (4, 3), (3, 1), (3, 2), (4, 5), (5, 6), (6, 7),
    (4, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13),
    (0, 14), (14, 15), (15, 16)
]

MODEL_CONFIGS = {
    'animalposeformer': {
        'seq_len': 27, 'batch_size': 32, 'lr': 2e-4, 'epochs': 200,
        'ckpt': 'checkpoints/compare_animalposeformer_best.pt',
        'log': 'log/compare_animalposeformer_log.txt',
        'import_name': 'AnimalPoseFormer', 'module': 'common.animal_poseformer',
    },
    'poseformer': {
        'seq_len': 27, 'batch_size': 32, 'lr': 2e-4, 'epochs': 200,
        'ckpt': 'checkpoints/compare_poseformer_best.pt',
        'log': 'log/compare_poseformer_log.txt',
        'import_name': 'PoseTransformer', 'module': 'common.poseformer.model_poseformer',
    },
    'poseformerv2': {
        'seq_len': 27, 'batch_size': 32, 'lr': 1e-4, 'epochs': 200,
        'ckpt': 'checkpoints/compare_poseformerv2_best.pt',
        'log': 'log/compare_poseformerv2_log.txt',
        'import_name': 'PoseTransformerV2', 'module': 'common.poseformerv2.model_poseformerV2',
    },
    'videopose3d': {
        'seq_len': 27, 'batch_size': 32, 'lr': 1e-4, 'epochs': 200,
        'ckpt': 'checkpoints/compare_videopose3d_best.pt',
        'log': 'log/compare_videopose3d_log.txt',
        'import_name': 'TemporalModel', 'module': 'common.videopose3D.model',
    },
    'dstformer': {
        'seq_len': 27, 'batch_size': 32, 'lr': 2e-4, 'epochs': 200,
        'ckpt': 'checkpoints/compare_dstformer_best.pt',
        'log': 'log/compare_dstformer_log.txt',
        'import_name': 'DSTformer', 'module': 'common.MotionBERT.DSTformer',
    },
}

class AnimalDataset(Dataset):
    def __init__(self, data_dict, seq_len=27, noise_std=0.005, mode='train'):
        self.data_dict = data_dict
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.mode = mode
        self.subjects = sorted(data_dict.keys())
        self.species_to_id = {s: i for i, s in enumerate(self.subjects)}
        self.data_source = []
        for sub in self.subjects:
            for act in data_dict[sub].keys():
                self.data_source.append((sub, act))
        self.val_samples = []
        if self.mode in ('val', 'test'):
            for list_idx, (animal, action) in enumerate(self.data_source):
                for view_idx in range(4):
                    self.val_samples.append((list_idx, view_idx))

    def __len__(self):
        if self.mode == 'train':
            return max(len(self.data_source) * 2, 1)
        return len(self.val_samples)

    def __getitem__(self, idx):
        if self.mode == 'train':
            list_idx = idx % len(self.data_source)
            view_idx = -1
        else:
            list_idx, view_idx = self.val_samples[idx]

        animal, action = self.data_source[list_idx]
        species_id = self.species_to_id[animal]
        pos_3d_raw = self.data_dict[animal][action]

        if len(pos_3d_raw) >= self.seq_len:
            if self.mode == 'train':
                start = np.random.randint(0, len(pos_3d_raw) - self.seq_len + 1)
            else:
                start = (len(pos_3d_raw) - self.seq_len) // 2
            pos_3d = pos_3d_raw[start:start + self.seq_len]
        else:
            pad_len = self.seq_len - len(pos_3d_raw)
            pos_3d = np.pad(pos_3d_raw, ((0, pad_len), (0, 0), (0, 0)), mode='edge')

        pos_3d = pos_3d - pos_3d[:, 0:1, :]

        if self.mode == 'train':
            theta = np.random.uniform(0, 2 * np.pi)
        else:
            angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
            theta = angles[view_idx]

        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        pos_3d_rotated = np.matmul(pos_3d, Rz)
        pos_2d = pos_3d_rotated[..., [0, 2]]
        pos_2d_norm = self._normalize_2d(pos_2d)

        if self.mode == 'train' and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, pos_2d_norm.shape).astype(np.float32)
            pos_2d_norm += noise

        return pos_2d_norm.astype(np.float32), pos_3d_rotated.astype(np.float32), species_id

    def _normalize_2d(self, pose_2d):
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


def batch_procrustes_torch(S1, S2):
    trans1 = S1.mean(dim=1, keepdim=True)
    trans2 = S2.mean(dim=1, keepdim=True)
    S1 = S1 - trans1
    S2 = S2 - trans2
    H = torch.matmul(S1.transpose(1, 2), S2)
    U, _, V = torch.svd(H)
    R = torch.matmul(V, U.transpose(1, 2))
    det = torch.det(R)
    diag = torch.ones(S1.shape[0], 3, device=S1.device)
    diag[:, 2] = torch.sign(det)
    R = torch.matmul(torch.matmul(V, torch.diag_embed(diag)), U.transpose(1, 2))
    S1_hat = torch.matmul(S1, R.transpose(1, 2)) + trans2
    return S1_hat


def build_model(model_name, device):
    cfg = MODEL_CONFIGS[model_name]
    seq_len = cfg['seq_len']

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
            dim_in=2, dim_out=3, dim_feat=256, depth=4, num_heads=8,
            mlp_ratio=2, num_joints=17, maxlen=seq_len
        ).to(device)

    raise ValueError(f"Unknown model: {model_name}")


def forward_step(model, model_name, batch_2d, device):
    return model(batch_2d.to(device).contiguous())


def get_pred_center(pred, model_name, seq_len):
    if pred.shape[1] == 1:
        return pred
    center = seq_len // 2
    return pred[:, center:center + 1, :, :]


def train_model(model_name):
    cfg = MODEL_CONFIGS[model_name]
    SEQ_LEN = cfg['seq_len']
    BATCH_SIZE = cfg['batch_size']
    LR = cfg['lr']
    EPOCHS = cfg['epochs']
    CKPT_PATH = cfg['ckpt']
    LOG_PATH = cfg['log']
    CENTER = SEQ_LEN // 2

    print("=" * 70)
    print(f"对比实验训练 — {model_name}")
    print(f"序列长度={SEQ_LEN}, batch={BATCH_SIZE}, lr={LR}")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU")
    device = torch.device("cuda")

    train_data = np.load('npz/real_npz/animals_train_3d.npz', allow_pickle=True)['positions_3d'].item()
    val_data = np.load('npz/real_npz/animals_val_3d.npz', allow_pickle=True)['positions_3d'].item()
    test_data = np.load('npz/real_npz/animals_test_3d.npz', allow_pickle=True)['positions_3d'].item()

    all_subjects = sorted(train_data.keys())
    species_to_id = {name: i for i, name in enumerate(all_subjects)}
    scales_dict = calculate_species_scales(train_data, species_to_id)

    train_dataset = AnimalDataset(train_data, seq_len=SEQ_LEN, noise_std=0.005, mode='train')
    val_dataset = AnimalDataset(val_data, seq_len=SEQ_LEN, noise_std=0.0, mode='val')
    test_dataset = AnimalDataset(test_data, seq_len=SEQ_LEN, noise_std=0.0, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(model_name, device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs('log', exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n{model_name} Started at {start_time}\n{'='*70}\n")

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_loss_mm = 0

        for batch_2d, batch_3d, batch_species in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", ncols=80):
            batch_3d = batch_3d.to(device)
            batch_species = batch_species.to(device)

            batch_scales = torch.tensor(
                [scales_dict[s.item()] for s in batch_species], device=device
            ).view(-1, 1, 1, 1)

            pred = forward_step(model, model_name, batch_2d, device)
            pred_center = get_pred_center(pred, model_name, SEQ_LEN)
            target_norm = batch_3d / batch_scales
            target_center = target_norm[:, CENTER:CENTER + 1, :, :]
            gt_center = batch_3d[:, CENTER:CENTER + 1, :, :]

            loss_mpjpe = mpjpe(pred_center, target_center)
            pred_unscaled = pred_center * batch_scales
            loss_bone = compute_bone_loss(pred_unscaled, gt_center, SKELETON_EDGES)
            loss_sym = compute_symmetry_loss(pred_unscaled)
            loss = loss_mpjpe# + 0.5 * loss_bone + 0.1 * loss_sym

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                loss_mm = mpjpe(pred_unscaled, gt_center)
                train_loss_mm += loss_mm.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_mm = (train_loss_mm / len(train_loader)) * 1000

        model.eval()
        val_mpjpe_mm = 0
        val_pa_mpjpe_mm = 0

        with torch.no_grad():
            for batch_2d, batch_3d, batch_species in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", ncols=80):
                batch_3d = batch_3d.to(device)
                batch_species = batch_species.to(device)

                batch_scales = torch.tensor(
                    [scales_dict[s.item()] for s in batch_species], device=device
                ).view(-1, 1, 1, 1)

                pred = forward_step(model, model_name, batch_2d, device)
                pred_center = get_pred_center(pred, model_name, SEQ_LEN)
                pred_3d = pred_center * batch_scales
                gt_center = batch_3d[:, CENTER:CENTER + 1, :, :]

                val_mpjpe_mm += mpjpe(pred_3d, gt_center).item()

                pred_aligned = batch_procrustes_torch(
                    pred_3d.view(pred_3d.shape[0], -1, 3),
                    gt_center.view(gt_center.shape[0], -1, 3)
                ).view_as(pred_3d)
                val_pa_mpjpe_mm += mpjpe(pred_aligned, gt_center).item()

        avg_val_mpjpe = (val_mpjpe_mm / len(val_loader)) * 1000
        avg_val_pa_mpjpe = (val_pa_mpjpe_mm / len(val_loader)) * 1000

        log_msg = (f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} ({avg_train_loss_mm:.1f}mm) | "
                   f"Val MPJPE: {avg_val_mpjpe:6.1f}mm | Val PA-MPJPE: {avg_val_pa_mpjpe:6.1f}mm")
        print(log_msg)

        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

        scheduler.step(avg_val_pa_mpjpe)

        if avg_val_pa_mpjpe < best_val_loss:
            best_val_loss = avg_val_pa_mpjpe
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"> Saved Best ({best_val_loss:.2f}mm)")
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"> Saved Best ({best_val_loss:.2f}mm)\n")

    print("\n" + "=" * 70)
    print(f"测试集评估 — {model_name}")
    print("=" * 70)

    model.eval()
    test_mpjpe_mm = 0
    test_pa_mpjpe_mm = 0

    with torch.no_grad():
        for batch_2d, batch_3d, batch_species in tqdm(test_loader, desc="Test", ncols=80):
            batch_3d = batch_3d.to(device)
            batch_species = batch_species.to(device)

            batch_scales = torch.tensor(
                [scales_dict[s.item()] for s in batch_species], device=device
            ).view(-1, 1, 1, 1)

            pred = forward_step(model, model_name, batch_2d, device)
            pred_center = get_pred_center(pred, model_name, SEQ_LEN)
            pred_3d = pred_center * batch_scales
            gt_center = batch_3d[:, CENTER:CENTER + 1, :, :]

            test_mpjpe_mm += mpjpe(pred_3d, gt_center).item()

            pred_aligned = batch_procrustes_torch(
                pred_3d.view(pred_3d.shape[0], -1, 3),
                gt_center.view(gt_center.shape[0], -1, 3)
            ).view_as(pred_3d)
            test_pa_mpjpe_mm += mpjpe(pred_aligned, gt_center).item()

    avg_test_mpjpe = (test_mpjpe_mm / len(test_loader)) * 1000
    avg_test_pa_mpjpe = (test_pa_mpjpe_mm / len(test_loader)) * 1000

    test_msg = (f"Test MPJPE: {avg_test_mpjpe:6.1f}mm | Test PA-MPJPE: {avg_test_pa_mpjpe:6.1f}mm")
    print(test_msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(test_msg + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="videopose3d",
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model to train')
    args = parser.parse_args()
    train_model(args.model)
