# 16_train_animal_poseformer.py
# 增强版动物3D姿态估计训练 - 兼容 AnimalPoseFormer 模型

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 添加路径
sys.path.append('./common')

# 导入本地模块
try:
    from common.loss import mpjpe, compute_bone_loss, compute_symmetry_loss
    from common.animal_poseformer import AnimalPoseFormer
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# ========== 配置 ==========

SKELETON_EDGES = [
    (0, 4), (4, 3), (3, 1), (3, 2), (4, 5), (5, 6), (6, 7),
    (4, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13), 
    (0, 14), (14, 15), (15, 16)
]

# ========== 辅助函数 ==========

def calculate_species_scales(data_dict, species_to_id):
    """计算每种动物的平均骨骼长度作为尺度因子"""
    species_scales = {}

    for animal_name, animal_id in species_to_id.items():
        if animal_name in data_dict:
            bone_lengths = []
            animal_data = data_dict[animal_name]
            for action in animal_data.keys():
                positions = animal_data[action]
                if len(positions) > 0:
                    for edge in SKELETON_EDGES:
                        if edge[0] < positions.shape[1] and edge[1] < positions.shape[1]:
                            bone_vec = positions[:, edge[1]] - positions[:, edge[0]]
                            bone_len = np.linalg.norm(bone_vec, axis=-1)
                            bone_lengths.extend(bone_len)

            if bone_lengths:
                species_scales[animal_id] = np.mean(bone_lengths)
            else:
                species_scales[animal_id] = 1.0
        else:
            species_scales[animal_id] = 1.0

    return species_scales

def normalize_2d(pose_2d):
    """
    将2D姿态归一化到 [-1, 1]
    pose_2d: (T, J, 2)
    """
    max_vals = np.max(np.abs(pose_2d), axis=(1, 2), keepdims=True)
    max_vals[max_vals < 1e-5] = 1.0
    return pose_2d / max_vals

def batch_compute_similarity_transform_torch(S1, S2):
    """
    计算从 S1 到 S2 的刚体变换 (Procrustes Analysis)
    """
    trans1 = S1.mean(dim=1, keepdim=True)
    trans2 = S2.mean(dim=1, keepdim=True)
    S1 = S1 - trans1
    S2 = S2 - trans2

    H = torch.matmul(S1.transpose(1, 2), S2)
    U, S, V = torch.svd(H)
    R = torch.matmul(V, U.transpose(1, 2))
    
    det = torch.det(R)
    diag = torch.ones(S1.shape[0], 3, device=S1.device)
    diag[:, 2] = torch.sign(det)
    diag_mat = torch.diag_embed(diag)
    R = torch.matmul(torch.matmul(V, diag_mat), U.transpose(1, 2))
        
    S1_hat = torch.matmul(S1, R.transpose(1, 2))
    S1_hat = S1_hat + trans2
    
    return S1_hat

# ========== 数据集 ==========

class SyntheticAnimalDataset(Dataset):
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
            print(f"📥 准备{mode}索引...")
            for list_idx, (animal, action) in enumerate(self.data_source):
                for view_idx in range(4):
                    self.val_samples.append((list_idx, view_idx))

    def __len__(self):
        if self.mode == 'train':
            return len(self.data_source) * 2
        else:
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
            pos_3d = pos_3d_raw[start : start + self.seq_len]
        else:
            pad_len = self.seq_len - len(pos_3d_raw)
            pos_3d = np.pad(pos_3d_raw, ((0, pad_len), (0, 0), (0, 0)), mode='edge')

        pos_3d = pos_3d - pos_3d[:, 0:1, :]

        if self.mode == 'train':
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-0.1, 0.1)
        else:
            angles = [0, np.pi/2, np.pi, 3*np.pi/2]
            theta = angles[view_idx]
            phi = 0.0

        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        pos_3d_rotated = np.matmul(pos_3d, Rz)
        pos_2d = pos_3d_rotated[..., [0, 2]]

        pos_2d_norm = normalize_2d(pos_2d)

        if self.mode == 'train' and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, pos_2d_norm.shape).astype(np.float32)
            pos_2d_norm += noise

        return (
            pos_2d_norm.astype(np.float32),
            pos_3d_rotated.astype(np.float32),
            species_id
        )

# ========== 训练主程序 ==========

def train():
    print("=" * 70)
    print("🚀 动物3D姿态估计 - AnimalPoseFormer 训练 (带增强位置编码)")
    print("策略: 使用预分割的训练/验证/测试集")
    print("=" * 70)

    import datetime
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log/animal_poseformer_training_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"🚀 AnimalPoseFormer Training Started at {start_time}\n")
        f.write(f"{'='*70}\n")

    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU")
    device = torch.device("cuda")

    SEQ_LEN = 27
    BATCH_SIZE = 32
    LR = 2e-4
    EPOCHS = 200

    print("📦 加载预分割数据集...")
    train_data = np.load('npz/real_npz/animals_train_3d.npz', allow_pickle=True)['positions_3d'].item()
    val_data = np.load('npz/real_npz/animals_val_3d.npz', allow_pickle=True)['positions_3d'].item()
    test_data = np.load('npz/real_npz/animals_test_3d.npz', allow_pickle=True)['positions_3d'].item()

    all_subjects = sorted(train_data.keys())
    species_to_id = {name: i for i, name in enumerate(all_subjects)}
    num_species = len(all_subjects)
    scales_dict = calculate_species_scales(train_data, species_to_id)

    print(f"物种数量: {num_species}")
    print(f"训练动作数: {sum(len(v) for v in train_data.values())}")
    print(f"验证动作数: {sum(len(v) for v in val_data.values())}")
    print(f"测试动作数: {sum(len(v) for v in test_data.values())}")

    train_dataset = SyntheticAnimalDataset(
        train_data, seq_len=SEQ_LEN, noise_std=0.005, mode='train'
    )
    val_dataset = SyntheticAnimalDataset(
        val_data, seq_len=SEQ_LEN, noise_std=0.0, mode='val'
    )
    test_dataset = SyntheticAnimalDataset(
        test_data, seq_len=SEQ_LEN, noise_std=0.0, mode='test'
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AnimalPoseFormer(
        num_frame=SEQ_LEN,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=32,
        depth=4,
        num_heads=8,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        use_lme=True,
        num_frame_kept=SEQ_LEN,
        num_coeff_kept=SEQ_LEN
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    center_idx = SEQ_LEN // 2

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_loss_mm = 0

        for batch_2d, batch_3d, batch_species in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", ncols=80):
            batch_2d = batch_2d.to(device)
            batch_3d = batch_3d.to(device)
            batch_species = batch_species.to(device)

            batch_scales = torch.tensor([scales_dict[s.item()] for s in batch_species], device=device).view(-1,1,1,1)

            target_norm = batch_3d / batch_scales

            pred_norm = model(batch_2d)

            target_norm_center = target_norm[:, center_idx : center_idx+1, :, :]
            batch_3d_center = batch_3d[:, center_idx : center_idx+1, :, :]

            loss_mpjpe = mpjpe(pred_norm, target_norm_center)

            pred_3d = pred_norm * batch_scales
            loss_bone = compute_bone_loss(pred_3d, batch_3d_center, SKELETON_EDGES)
            loss_sym = compute_symmetry_loss(pred_3d)

            loss = loss_mpjpe + 0.5 * loss_bone + 0.1 * loss_sym

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                pred_mm = pred_norm * batch_scales
                loss_mm = mpjpe(pred_mm, batch_3d_center)
                train_loss_mm += loss_mm.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_mm = (train_loss_mm / len(train_loader)) * 1000

        model.eval()
        val_pa_mpjpe_mm = 0
        val_mpjpe_mm = 0

        with torch.no_grad():
            for batch_2d, batch_3d, batch_species in tqdm(val_loader, desc=f"Epoch {epoch+1} Val", ncols=80):
                batch_2d = batch_2d.to(device)
                batch_3d = batch_3d.to(device)
                batch_species = batch_species.to(device)

                batch_scales = torch.tensor([scales_dict[s.item()] for s in batch_species], device=device).view(-1,1,1,1)

                pred_norm = model(batch_2d)
                pred_3d = pred_norm * batch_scales

                batch_3d_center = batch_3d[:, center_idx : center_idx+1, :, :]

                raw_loss = mpjpe(pred_3d, batch_3d_center)
                val_mpjpe_mm += raw_loss.item()

                pred_3d_aligned = batch_compute_similarity_transform_torch(
                    pred_3d.view(pred_3d.shape[0], -1, 3),
                    batch_3d_center.view(batch_3d_center.shape[0], -1, 3)
                ).view_as(pred_3d)

                pa_loss = mpjpe(pred_3d_aligned, batch_3d_center)
                val_pa_mpjpe_mm += pa_loss.item()

        avg_val_mpjpe = (val_mpjpe_mm / len(val_loader)) * 1000
        avg_val_pa_mpjpe = (val_pa_mpjpe_mm / len(val_loader)) * 1000

        log_msg = (f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} ({avg_train_loss_mm:.1f}mm) | "
                   f"Val MPJPE: {avg_val_mpjpe:6.1f}mm | "
                   f"Val PA-MPJPE: {avg_val_pa_mpjpe:6.1f}mm")

        print(log_msg)

        with open("log/animal_poseformer_training_log.txt", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

        scheduler.step(avg_val_pa_mpjpe)

        if avg_val_pa_mpjpe < best_val_loss:
            best_val_loss = avg_val_pa_mpjpe
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/animal_poseformer_best_model.pt')
            print(f"💾 Saved Best Model ({best_val_loss:.2f}mm)")
            with open("log/animal_poseformer_training_log.txt", "a", encoding="utf-8") as f:
                f.write(f"💾 Saved Best Model ({best_val_loss:.2f}mm)\n")

    print("\n" + "=" * 70)
    print("📊 测试集评估")
    print("=" * 70)

    model.eval()
    test_pa_mpjpe_mm = 0
    test_mpjpe_mm = 0

    with torch.no_grad():
        for batch_2d, batch_3d, batch_species in tqdm(test_loader, desc="Test", ncols=80):
            batch_2d = batch_2d.to(device)
            batch_3d = batch_3d.to(device)
            batch_species = batch_species.to(device)

            batch_scales = torch.tensor([scales_dict[s.item()] for s in batch_species], device=device).view(-1,1,1,1)

            pred_norm = model(batch_2d)
            pred_3d = pred_norm * batch_scales

            batch_3d_center = batch_3d[:, center_idx : center_idx+1, :, :]

            raw_loss = mpjpe(pred_3d, batch_3d_center)
            test_mpjpe_mm += raw_loss.item()

            pred_3d_aligned = batch_compute_similarity_transform_torch(
                pred_3d.view(pred_3d.shape[0], -1, 3),
                batch_3d_center.view(batch_3d_center.shape[0], -1, 3)
            ).view_as(pred_3d)

            pa_loss = mpjpe(pred_3d_aligned, batch_3d_center)
            test_pa_mpjpe_mm += pa_loss.item()

    avg_test_mpjpe = (test_mpjpe_mm / len(test_loader)) * 1000
    avg_test_pa_mpjpe = (test_pa_mpjpe_mm / len(test_loader)) * 1000

    test_msg = (f"Test  MPJPE:    {avg_test_mpjpe:6.1f}mm | "
                f"Test  PA-MPJPE: {avg_test_pa_mpjpe:6.1f}mm")

    print(test_msg)
    with open("log/animal_poseformer_training_log.txt", "a", encoding="utf-8") as f:
        f.write(test_msg + "\n")

if __name__ == '__main__':
    train()
