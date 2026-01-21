# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def compute_bone_loss(predicted, target, skeleton_edges):
    """
    骨骼长度损失：强制预测的骨骼长度与真值一致
    predicted, target: (B, T, J, 3)
    skeleton_edges: 骨架连接边列表
    """
    assert predicted.shape == target.shape
    
    loss = 0
    valid_edges = 0
    
    for edge in skeleton_edges:
        p1, p2 = edge
        if p2 >= predicted.shape[2]: 
            continue
        
        pred_bone = torch.norm(predicted[:, :, p1] - predicted[:, :, p2], dim=-1)
        gt_bone = torch.norm(target[:, :, p1] - target[:, :, p2], dim=-1)
        
        # 使用MSE损失，对异常值更敏感
        loss += torch.nn.functional.mse_loss(pred_bone, gt_bone)
        valid_edges += 1
    
    return loss / valid_edges if valid_edges > 0 else torch.tensor(0.0)

def compute_symmetry_loss(predicted):
    """
    对称性约束：防止左右肢体塌陷到同一侧
    假设：左侧关节索引为 L, 右侧为 R
    惩罚左右关节在3D空间中靠得太近
    """
    # AP10K关键点定义：左侧关节 [1, 2, 5, 6, 7, 11, 12, 13]
    # 右侧关节 [3, 4, 8, 9, 10, 14, 15, 16]
    left_joints = [1, 2, 5, 6, 7, 11, 12, 13]
    right_joints = [3, 4, 8, 9, 10, 14, 15, 16]
    
    # 检查索引有效性
    max_joint = predicted.shape[2]
    valid_left = [j for j in left_joints if j < max_joint]
    valid_right = [j for j in right_joints if j < max_joint]
    
    if not valid_left or not valid_right:
        return torch.tensor(0.0)
    
    # 惩罚左右对称点在3D空间中靠得太近
    left_pos = predicted[:, :, valid_left]
    right_pos = predicted[:, :, valid_right]
    
    # 计算所有左右关节对之间的最小距离
    batch_size, seq_len, num_left, _ = left_pos.shape
    _, _, num_right, _ = right_pos.shape
    
    # 重塑为 (B*T, num_left, 3) 和 (B*T, num_right, 3)
    left_flat = left_pos.reshape(-1, num_left, 3)
    right_flat = right_pos.reshape(-1, num_right, 3)
    
    # 计算所有左右关节对之间的距离矩阵
    dist_matrix = torch.cdist(left_flat, right_flat)  # (B*T, num_left, num_right)
    
    # 取每对左右关节的最小距离
    min_dist = torch.min(dist_matrix, dim=2)[0]  # (B*T, num_left)
    min_dist = min_dist.reshape(batch_size, seq_len, num_left)
    
    # 如果距离小于阈值（0.1米），则产生惩罚
    threshold = 0.1
    penalty = torch.nn.functional.relu(threshold - min_dist)
    
    return torch.mean(penalty)

def compute_anatomy_loss(predicted, target, skeleton_edges):
    """
    综合解剖学约束损失
    结合骨骼长度损失和对称性损失
    """
    bone_loss = compute_bone_loss(predicted, target, skeleton_edges)
    sym_loss = compute_symmetry_loss(predicted)
    
    return bone_loss, sym_loss