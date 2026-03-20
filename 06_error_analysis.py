import os
import sys
import numpy as np
import torch
import pandas as pd
from collections import defaultdict

sys.path.append('./common')
from common.animals_dataset import AnimalsDataset
from common.loss import mpjpe
from common.transformer_model import AnimalPoseTransformer

def batch_compute_similarity_transform_torch(S1, S2):
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

def normalize_2d(pose_2d):
    max_vals = np.max(np.abs(pose_2d), axis=(1, 2), keepdims=True)
    max_vals[max_vals < 1e-5] = 1.0
    return pose_2d / max_vals

def calculate_species_scales(dataset, species_to_id):
    SKELETON_EDGES = [
        (0, 4), (4, 3), (3, 1), (3, 2), (4, 5), (5, 6), (6, 7),
        (4, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13), 
        (0, 14), (14, 15), (15, 16)
    ]
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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading data...")
    dataset = AnimalsDataset('npz/real_npz/data_3d_animals.npz')
    all_animals = sorted(dataset.subjects())
    species_to_id = {name: i for i, name in enumerate(all_animals)}
    scales_dict = calculate_species_scales(dataset, species_to_id)
    
    print("Loading model...")
    checkpoint_path = 'checkpoints/best_synth_model.pt'
    
    # 动态获取模型参数 (之前为了移除 species_embed，移除了 num_species 的输入)
    # let's try with new constructor based on transformer_model
    model = AnimalPoseTransformer(
        num_joints=17, embed_dim=256, depth=4, num_heads=8, seq_len=27
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    results = []
    seq_len = 27
    views = [0, 90, 180, 270]
    
    print("Evaluating...")
    with torch.no_grad():
        for animal in all_animals:
            species_id = species_to_id[animal]
            scale = scales_dict[species_id]
            for action in dataset[animal].keys():
                pos_3d_raw = dataset[animal][action]['positions']
                
                if len(pos_3d_raw) >= seq_len:
                    start = (len(pos_3d_raw) - seq_len) // 2
                    pos_3d = pos_3d_raw[start : start + seq_len]
                else:
                    pad_len = seq_len - len(pos_3d_raw)
                    pos_3d = np.pad(pos_3d_raw, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
                    
                pos_3d = pos_3d - pos_3d[:, 0:1, :]
                
                for view_angle in views:
                    theta = np.deg2rad(view_angle)
                    c, s = np.cos(theta), np.sin(theta)
                    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
                    pos_3d_rotated = np.matmul(pos_3d, Ry.T)
                    
                    pos_2d = pos_3d_rotated[..., [0, 2]]
                    pos_2d_norm = normalize_2d(pos_2d)
                    
                    input_2d = torch.tensor(pos_2d_norm, dtype=torch.float32).unsqueeze(0).contiguous().to(device)
                    gt_3d = torch.tensor(pos_3d_rotated, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    pred_norm = model(input_2d)
                    pred_3d = pred_norm * scale
                    
                    pred_3d_aligned = batch_compute_similarity_transform_torch(
                        pred_3d.view(1, -1, 3), 
                        gt_3d.view(1, -1, 3)
                    ).squeeze(0).view_as(pred_3d)
                    
                    pa_err = mpjpe(pred_3d_aligned, gt_3d).item() * 1000
                    norm_pa_err = (pa_err / (scale * 1000)) * 100 if scale > 0 else 0
                    
                    results.append({
                        'Species': animal,
                        'Action': action,
                        'View': view_angle,
                        'PA-MPJPE (mm)': pa_err,
                        'Normalized PA-MPJPE (%)': norm_pa_err
                    })

    df = pd.DataFrame(results)
    
    species_err = df.groupby('Species')['PA-MPJPE (mm)'].mean().sort_values()
    print("\n--- Per-Species Error (mm) ---")
    print(species_err)
    
    species_norm_err = df.groupby('Species')['Normalized PA-MPJPE (%)'].mean().sort_values()
    print("\n--- Per-Species Normalized Error (%) ---")
    print(species_norm_err)
    
    action_err = df.groupby('Action')['PA-MPJPE (mm)'].mean().sort_values(ascending=False)
    print("\n--- Action-wise Error (mm) ---")
    print(action_err)
    
    action_norm_err = df.groupby('Action')['Normalized PA-MPJPE (%)'].mean().sort_values(ascending=False)
    print("\n--- Action-wise Normalized Error (%) ---")
    print(action_norm_err)
    
    with open('error_analysis_data.txt', 'w', encoding='utf-8') as f:
        f.write("Species (Absolute Error mm):\n" + str(species_err) + "\n\n")
        f.write("Species (Normalized Error %):\n" + str(species_norm_err) + "\n\n")
        f.write("Actions (Absolute Error mm):\n" + str(action_err) + "\n\n")
        f.write("Actions (Normalized Error %):\n" + str(action_norm_err))

if __name__ == '__main__':
    main()
