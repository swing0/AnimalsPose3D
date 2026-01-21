# 03c_validate_model.py
# 增强版模型验证可视化工 - 适配合成数据训练模型
# 功能：
# 1. 加载 best_synth_model.pt
# 2. 从 SyntheticAnimalDataset (Mode=Validation) 获取样本
# 3. 可视化对比: 输入 2D, 预测 3D, 真值 3D

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import sys
import os

# 添加路径
sys.path.append('./common')

try:
    from common.animals_dataset import AnimalsDataset
    from common.transformer_model import AnimalPoseTransformer
    from common.loss import mpjpe
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# 复用训练脚本中的 Dataset 定义 (为了保证数据处理一致)
# 这里我们简单复制 SyntheticAnimalDataset 的验证逻辑
# 或者直接导入 (如果训练脚本是作为模块)。
# 为了独立性，这里重写简化的验证数据加载逻辑。

SKELETON_EDGES = [
    (0, 4), (4, 3), (3, 1), (3, 2), (4, 5), (5, 6), (6, 7),
    (4, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13), 
    (0, 14), (14, 15), (15, 16)
]

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
    max_vals = np.max(np.abs(pose_2d), axis=(1, 2), keepdims=True)
    max_vals[max_vals < 1e-5] = 1.0
    return pose_2d / max_vals

def batch_compute_similarity_transform_torch(S1, S2):
    """
    计算从 S1 到 S2 的刚体变换 (Procrustes Analysis)
    S1, S2: (B, N, 3)
    返回: S1_hat (对齐后的 S1)
    """
    # 1. 移除质心
    trans1 = S1.mean(dim=1, keepdim=True)
    trans2 = S2.mean(dim=1, keepdim=True)
    S1 = S1 - trans1
    S2 = S2 - trans2

    # 3. 计算旋转
    # H = S1^T * S2
    H = torch.matmul(S1.transpose(1, 2), S2) # (B, 3, 3)
    
    U, S, V = torch.svd(H)
    
    # R = V * U^T
    R = torch.matmul(V, U.transpose(1, 2))
    
    # 修正反射 (Per-sample check)
    det = torch.det(R) # (B,)
    
    # 构建对角矩阵: [1, 1, sign(det)]
    diag = torch.ones(S1.shape[0], 3, device=S1.device)
    diag[:, 2] = torch.sign(det)
    diag_mat = torch.diag_embed(diag) # (B, 3, 3)
    
    # R = V * diag * U^T
    # 只有当 det < 0 时才需要反转
    # 但svd返回的矩阵可能包含反射。
    # 我们这里使用通用的 R = V * diag * U^T
    R = torch.matmul(torch.matmul(V, diag_mat), U.transpose(1, 2))
        
    # 4. 应用变换
    S1_hat = torch.matmul(S1, R.transpose(1, 2))
    
    # 5.为了计算误差，需要把 S1_hat 移回 S2 的位置
    S1_hat = S1_hat + trans2
    
    return S1_hat

class ValidationVisualizer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self.load_model(model_path)
        self.dataset, self.species_to_id, self.scales = self.load_data()
        self.val_samples = self.prepare_val_samples()
        
        self.current_idx = 0
        self.fig = None
        
    def load_model(self, path):
        print(f"📦 加载模型: {path}")
        model = AnimalPoseTransformer(
            num_joints=17, embed_dim=256, depth=4, num_heads=8, seq_len=27, num_species=20
        ).to(self.device)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        return model
        
    def load_data(self):
        print("📂 加载 3D 数据...")
        dataset = AnimalsDataset('npz/real_npz/data_3d_animals.npz')
        all_animals = sorted(dataset.subjects())
        species_to_id = {name: i for i, name in enumerate(all_animals)}
        scales = calculate_species_scales(dataset, species_to_id)
        return dataset, species_to_id, scales
        
    def prepare_val_samples(self):
        print("📥 准备验证样本...")
        samples = []
        # 只取前 5 个动物的前 2 个动作，每个取 4 个视角
        # 避免太多，但这只是为了随机漫游
        all_seqs = []
        for animal in self.dataset.subjects():
            for action in self.dataset[animal].keys():
                all_seqs.append((animal, action))
                
        # 随机取 20 个序列用于验证
        import random
        random.seed(42)
        random.shuffle(all_seqs)
        selected_seqs = all_seqs[:20]
        
        for animal, action in selected_seqs:
            for view_angle in [0, 90, 180, 270]:
                samples.append({
                    'animal': animal,
                    'action': action,
                    'view': view_angle
                })
        return samples

    def get_sample_data(self, idx):
        s = self.val_samples[idx]
        animal, action, view_deg = s['animal'], s['action'], s['view']
        
        # 获取 3D
        pos_3d_raw = self.dataset[animal][action]['positions']
        # Center crop to 27
        seq_len = 27
        if len(pos_3d_raw) >= seq_len:
            start = (len(pos_3d_raw) - seq_len) // 2
            pos_3d = pos_3d_raw[start:start+seq_len]
        else:
            pad = seq_len - len(pos_3d_raw)
            pos_3d = np.pad(pos_3d_raw, ((0, pad), (0,0), (0,0)), mode='edge')
            
        pos_3d = pos_3d - pos_3d[:, 0:1, :] # Root rel
        
        # 旋转 (生成 Input)
        theta = np.deg2rad(view_deg)
        c, s_sin = np.cos(theta), np.sin(theta)
        Ry = np.array([[c, 0, s_sin], [0, 1, 0], [-s_sin, 0, c]], dtype=np.float32)
        pos_3d_rot = np.matmul(pos_3d, Ry.T)
        
        # 投影 2D
        pos_2d = pos_3d_rot[..., [0, 2]] # X, Z
        pos_2d_norm = normalize_2d(pos_2d)
        
        return {
            'input_2d': torch.tensor(pos_2d_norm, dtype=torch.float32).unsqueeze(0).to(self.device),
            'gt_3d': torch.tensor(pos_3d, dtype=torch.float32).unsqueeze(0).to(self.device), # Canonical
            'gt_3d_rot': torch.tensor(pos_3d_rot, dtype=torch.float32).to(self.device), # Rotated (Camera Space)
            'species_id': torch.tensor([self.species_to_id[animal]], device=self.device),
            'meta': s
        }

    def visualize(self):
        import matplotlib
        matplotlib.use('TkAgg')
        
        self.fig = plt.figure(figsize=(18, 8))
        self.ax1 = self.fig.add_subplot(131, title="Input 2D View")
        self.ax2 = self.fig.add_subplot(132, projection='3d', title="Prediction (Aligned)")
        self.ax3 = self.fig.add_subplot(133, projection='3d', title="Ground Truth (Canonical)")
        
        plt.subplots_adjust(bottom=0.2)
        
        self.btn_prev = Button(plt.axes([0.3, 0.05, 0.1, 0.075]), 'Previous')
        self.btn_next = Button(plt.axes([0.6, 0.05, 0.1, 0.075]), 'Next')
        
        self.btn_prev.on_clicked(self.prev_sample)
        self.btn_next.on_clicked(self.next_sample)
        
        self.update_plot()
        plt.show()
        
    def prev_sample(self, event):
        self.current_idx = (self.current_idx - 1) % len(self.val_samples)
        self.update_plot()
        
    def next_sample(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.val_samples)
        self.update_plot()
        
    def update_plot(self):
        data = self.get_sample_data(self.current_idx)
        
        # Inference
        with torch.no_grad():
            pred_norm = self.model(data['input_2d'], data['species_id'])
            scale = self.scales[data['species_id'].item()]
            pred_3d = pred_norm * scale
            
        # Post-process for visu (Take 1st frame or middle frame)
        frame_idx = 13 # Middle
        
        p2 = data['input_2d'][0, frame_idx].cpu().numpy()
        p3_pred = pred_3d[0, frame_idx].cpu().numpy()
        p3_gt = data['gt_3d'][0, frame_idx].cpu().numpy()
        p3_gt_rot = data['gt_3d_rot'][frame_idx].cpu().numpy()
        
        # Alignment (PA-MPJPE logic for viz)
        # Align Pred to GT (Canonical) for fair visual comparison
        # 注意: 模型输出的是 Camera Space (Rotated)，GT 是 Canonical。
        # 如果我们直接画 Pred，它是歪的（对于 90度视角）。
        # 为了验证 "Pose" 对不对，我们把 Pred 对齐到 GT。
        p3_pred_aligned = batch_compute_similarity_transform_torch(
            torch.tensor(p3_pred).unsqueeze(0), 
            torch.tensor(p3_gt).unsqueeze(0)
        ).squeeze(0).numpy()
        
        # 1. 2D Input
        self.ax1.clear()
        self.ax1.set_title(f"Input 2D (View {data['meta']['view']}°)")
        self.draw_2d_skeleton(self.ax1, p2)
        self.ax1.set_aspect('equal')
        self.ax1.invert_yaxis() # Image coord
        
        # 2. Prediction (White) vs GT (Green) [Aligned]
        self.ax2.clear()
        self.ax2.set_title(f"Pred (Aligned) vs GT\nAnimal: {data['meta']['animal']}")
        self.draw_3d_skeleton(self.ax2, p3_gt, 'green', 'GT')
        self.draw_3d_skeleton(self.ax2, p3_pred_aligned, 'red', 'Pred')
        self.set_3d_axes(self.ax2, p3_gt)
        self.ax2.legend()
        
        # 3. Raw Prediction (Camera Space)
        self.ax3.clear()
        self.ax3.set_title(f"Raw Output (Camera Space)\nShould match View Angle")
        self.draw_3d_skeleton(self.ax3, p3_pred, 'blue', 'RawPred')
        # 画一个参照的 Ground Plane 或 Axis 指示方向
        self.ax3.quiver(0,0,0, 100,0,0, color='r', arrow_length_ratio=0.1) # X
        self.ax3.quiver(0,0,0, 0,100,0, color='g', arrow_length_ratio=0.1) # Y
        self.ax3.quiver(0,0,0, 0,0,100, color='b', arrow_length_ratio=0.1) # Z
        self.set_3d_axes(self.ax3, p3_pred)
        
        self.fig.canvas.draw_idle()

    def draw_2d_skeleton(self, ax, pose):
        for s, e in SKELETON_EDGES:
            ax.plot([pose[s,0], pose[e,0]], [pose[s,1], pose[e,1]], 'b-')
        ax.scatter(pose[:,0], pose[:,1], c='r', s=10)

    def draw_3d_skeleton(self, ax, pose, color, label):
        first = True
        for s, e in SKELETON_EDGES:
            ax.plot([pose[s,0], pose[e,0]], [pose[s,1], pose[e,1]], [pose[s,2], pose[e,2]], color=color, label=label if first else "")
            first = False
        ax.scatter(pose[:,0], pose[:,1], pose[:,2], c=color, s=10)
        
    def set_3d_axes(self, ax, pose):
        limit = np.max(np.abs(pose)) * 1.5
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

if __name__ == '__main__':
    checkpoint = 'checkpoints/best_synth_model.pt'
    if not os.path.exists(checkpoint):
        print(f"❌ 找不到模型: {checkpoint}")
    else:
        viz = ValidationVisualizer(checkpoint)
        viz.visualize()