# 03c_train_enhanced.py
# 增强版动物3D姿态估计训练 - 完全合成数据版
# 策略: 
#   训练: 实时随机旋转 3D -> 2D (无限数据)
#   验证: 实时固定旋转 3D -> 2D (一致性评估)
#   移除对外部 2D npz 文件的依赖

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
    from common.animals_dataset import AnimalsDataset
    from common.loss import mpjpe, compute_bone_loss, compute_symmetry_loss
    from common.transformer_model import AnimalPoseTransformer
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

def calculate_species_scales(dataset, species_to_id):
    """计算每种动物的平均骨骼长度作为尺度因子"""
    species_scales = {}
    available_animals = dataset.subjects()
    
    for animal_name, animal_id in species_to_id.items():
        if animal_name in available_animals:
            bone_lengths = []
            animal_data = dataset[animal_name]
            for action in animal_data.keys():
                positions = animal_data[action]['positions']
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
    # 逐帧归一化
    # 找到每帧的最大绝对值
    max_vals = np.max(np.abs(pose_2d), axis=(1, 2), keepdims=True) # (T, 1, 1)
    max_vals[max_vals < 1e-5] = 1.0 # 避免除零
    
    return pose_2d / max_vals

def batch_compute_similarity_transform_torch(S1, S2):
    """
    计算从 S1 到 S2 的刚体变换 (Procrustes Analysis)
    S1, S2: (B, N, 3)
    返回: S1_hat (对齐后的 S1)
    """
    trans1 = S1.mean(dim=1, keepdim=True)
    trans2 = S2.mean(dim=1, keepdim=True)
    S1 = S1 - trans1
    S2 = S2 - trans2

    H = torch.matmul(S1.transpose(1, 2), S2)
    U, S, V = torch.svd(H) # SVD returns U, S, V. R = V * U^T
    R = torch.matmul(V, U.transpose(1, 2))
    
    # 修正反射 (Per-sample check)
    det = torch.det(R) # (B,)
    
    # 构建对角矩阵: [1, 1, sign(det)]
    diag = torch.ones(S1.shape[0], 3, device=S1.device)
    diag[:, 2] = torch.sign(det)
    diag_mat = torch.diag_embed(diag) # (B, 3, 3)
    
    # R = V * diag * U^T
    R = torch.matmul(torch.matmul(V, diag_mat), U.transpose(1, 2))
        
    S1_hat = torch.matmul(S1, R.transpose(1, 2))
    
    # 5. 为了计算误差，需要把 S1_hat 移回 S2 的位置 (或者对比 Centered S2)
    # 这里我们选择移回 S2 的位置，这样可以直接和原始 S2 比较
    S1_hat = S1_hat + trans2
    
    return S1_hat

# ========== 数据集 ==========

class SyntheticAnimalDataset(Dataset):
    """
    合成动物数据集
    mode='train': 随机旋转 (数据增强)
    mode='val': 固定旋转 (0, 90, 180, 270) (确定性评估)
    """
    def __init__(self, data_source, dataset, species_to_id, seq_len=27, noise_std=0.005, mode='train'):
        self.data_source = data_source
        self.dataset = dataset
        self.species_to_id = species_to_id
        self.seq_len = seq_len
        self.noise_std = noise_std
        self.mode = mode
        
        # 预计算验证集索引以支持多视角
        self.val_samples = []
        if self.mode == 'val':
            print("📥 准备验证索引...")
            for list_idx, (animal, action) in enumerate(self.data_source):
                # 为每个序列创建 4 个视角的样本
                # (animal, action, view_angle_idx)
                for view_idx in range(4): # 0, 90, 180, 270
                    self.val_samples.append((list_idx, view_idx))
    
    def __len__(self):
        if self.mode == 'train':
            return len(self.data_source)
        else:
            return len(self.val_samples)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            list_idx = idx
            view_idx = -1 # Random
        else:
            list_idx, view_idx = self.val_samples[idx]
            
        animal, action = self.data_source[list_idx]
        species_id = self.species_to_id[animal]
        
        # 获取原始3D数据 (相对Root)
        pos_3d_raw = self.dataset[animal][action]['positions']
        
        # 截断或填充
        if len(pos_3d_raw) >= self.seq_len:
            if self.mode == 'train':
                start = np.random.randint(0, len(pos_3d_raw) - self.seq_len + 1)
            else:
                start = (len(pos_3d_raw) - self.seq_len) // 2 # Center crop for val
            pos_3d = pos_3d_raw[start : start + self.seq_len]
        else:
            pad_len = self.seq_len - len(pos_3d_raw)
            pos_3d = np.pad(pos_3d_raw, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
            
        # Root-Relative
        pos_3d = pos_3d - pos_3d[:, 0:1, :]
        
        # 旋转逻辑
        if self.mode == 'train':
            # 随机旋转 0-360
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(-0.1, 0.1) # 微微俯仰
        else:
            # 固定旋转 0, 90, 180, 270 度
            angles = [0, np.pi/2, np.pi, 3*np.pi/2]
            theta = angles[view_idx]
            phi = 0.0

        c, s = np.cos(theta), np.sin(theta)
        # Y轴旋转矩阵
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        
        pos_3d_rotated = np.matmul(pos_3d, Ry.T)
        
        # 投影到 2D (正交投影: 取 X, Z)
        # 假设 Y 是深度
        pos_2d = pos_3d_rotated[..., [0, 2]] 
        
        # 归一化 2D
        pos_2d_norm = normalize_2d(pos_2d)
        
        # 添加噪声 (仅训练)
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
    print("🚀 动物3D姿态估计 - 完全合成流程")
    print("策略: 移除外部npz依赖，全部从3D实时生成")
    print("=" * 70)
    
    # 初始化日志文件
    import datetime
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("training_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n{'='*70}\n")
        f.write(f"🚀 Training Started at {start_time}\n")
        f.write(f"Strategy: Full Synthetic (Infinite Rotation) + Fixed Val\n")
        f.write(f"{'='*70}\n")
    
    # 1. 设置
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU")
    device = torch.device("cuda")
    
    SEQ_LEN = 27
    BATCH_SIZE = 32
    LR = 2e-4
    EPOCHS = 200
    EMBED_DIM = 256
    DEPTH = 4
    HEADS = 8
    
    # 2. 加载3D数据源
    try:
        dataset = AnimalsDataset('npz/real_npz/data_3d_animals.npz')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    all_animals = dataset.subjects()
    species_to_id = {name: i for i, name in enumerate(sorted(all_animals))}
    num_species = len(all_animals)
    scales_dict = calculate_species_scales(dataset, species_to_id)
    
    print(f"物种数量: {num_species}")
    
    # 3. 划分数据集
    all_sequences = []
    for animal in all_animals:
        for action in dataset[animal].keys():
            all_sequences.append((animal, action))
            
    import random
    random.seed(42)
    random.shuffle(all_sequences)
    split = int(len(all_sequences) * 0.8)
    train_source = all_sequences[:split]
    val_source = all_sequences[split:]
    
    print(f"训练序列数: {len(train_source)}")
    print(f"验证序列数: {len(val_source)} (x4 视角 = {len(val_source)*4} 样本)")
    
    # 4. 构建 Dataset
    train_dataset = SyntheticAnimalDataset(
        train_source, dataset, species_to_id, 
        seq_len=SEQ_LEN, noise_std=0.005, mode='train'
    )
    
    val_dataset = SyntheticAnimalDataset(
        val_source, dataset, species_to_id, 
        seq_len=SEQ_LEN, noise_std=0.0, mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 5. 模型
    model = AnimalPoseTransformer(
        num_joints=17, embed_dim=EMBED_DIM, depth=DEPTH, 
        num_heads=HEADS, seq_len=SEQ_LEN
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
    
    # 6. 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_loss_mm = 0
        
        for batch_2d, batch_3d, batch_species in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", ncols=80):
            batch_2d = batch_2d.to(device)
            batch_3d = batch_3d.to(device)
            batch_species = batch_species.to(device)
            
            batch_scales = torch.tensor([scales_dict[s.item()] for s in batch_species], device=device).view(-1,1,1,1)
            
            # Target Normalization
            target_norm = batch_3d / batch_scales
            
            pred_norm = model(batch_2d)
            
            # 计算多种损失
            loss_mpjpe = mpjpe(pred_norm, target_norm)
            
            # 新增解剖学约束损失
            pred_3d = pred_norm * batch_scales
            loss_bone = compute_bone_loss(pred_3d, batch_3d, SKELETON_EDGES)
            loss_sym = compute_symmetry_loss(pred_3d)
            
            # 组合损失（根据您的建议调整权重）
            loss = loss_mpjpe + 0.5 * loss_bone + 0.1 * loss_sym
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # 为了给用户看直观的单位，顺便计算一下还原后的 mm 误差
            with torch.no_grad():
                pred_mm = pred_norm * batch_scales
                loss_mm = mpjpe(pred_mm, batch_3d)
                train_loss_mm += loss_mm.item()
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_mm = (train_loss_mm / len(train_loader)) * 1000 # m -> mm
        
        # 验证
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
                
                # Raw MPJPE
                raw_loss = mpjpe(pred_3d, batch_3d)
                val_mpjpe_mm += raw_loss.item()
                
                # PA-MPJPE
                pred_3d_aligned = batch_compute_similarity_transform_torch(
                    pred_3d.view(pred_3d.shape[0], -1, 3), 
                    batch_3d.view(batch_3d.shape[0], -1, 3)
                ).view_as(pred_3d)
                
                pa_loss = mpjpe(pred_3d_aligned, batch_3d)
                val_pa_mpjpe_mm += pa_loss.item()
        
        avg_val_mpjpe = (val_mpjpe_mm / len(val_loader)) * 1000
        avg_val_pa_mpjpe = (val_pa_mpjpe_mm / len(val_loader)) * 1000
        
        log_msg = (f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.4f} ({avg_train_loss_mm:.1f}mm) | "
                   f"Val MPJPE: {avg_val_mpjpe:6.1f}mm | "
                   f"Val PA-MPJPE: {avg_val_pa_mpjpe:6.1f}mm")
        
        print(log_msg)
        
        # 写入日志文件
        with open("training_log.txt", "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")
        
        scheduler.step(avg_val_pa_mpjpe)
        
        if avg_val_pa_mpjpe < best_val_loss:
            best_val_loss = avg_val_pa_mpjpe
            torch.save(model.state_dict(), 'checkpoints/best_synth_model.pt')
            print(f"💾 Saved Best Model ({best_val_loss:.2f}mm)")
            with open("training_log.txt", "a", encoding="utf-8") as f:
                f.write(f"💾 Saved Best Model ({best_val_loss:.2f}mm)\n")

if __name__ == '__main__':
    train()
