import os
import torch
import torch.optim as optim
import numpy as np
from common.transformer_model import AnimalPoseTransformer
from common.loss import mpjpe
from common.generators import ChunkedGenerator
from common.animals_dataset import AnimalsDataset
from common.arguments import parse_args

# 定义骨架连线（用于骨骼长度损失）
SKELETON_EDGES = [
    (0, 4), (4, 3), (3, 1), (3, 2), (4, 5), (5, 6), (6, 7),
    (4, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13), (0, 14), (14, 15), (15, 16)
]


def bone_length_loss(pred, gt):
    p1 = pred[:, :, [e[0] for e in SKELETON_EDGES], :]
    p2 = pred[:, :, [e[1] for e in SKELETON_EDGES], :]
    g1 = gt[:, :, [e[0] for e in SKELETON_EDGES], :]
    g2 = gt[:, :, [e[1] for e in SKELETON_EDGES], :]
    return torch.mean(torch.abs(torch.norm(p1 - p2, dim=-1) - torch.norm(g1 - g2, dim=-1)))


def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 配置 ---
    SEQ_LEN = 27
    LR = 1e-3
    WAPMUP_EPOCHS = 5

    # --- 数据加载 (同你之前的逻辑) ---
    dataset = AnimalsDataset(f'npz/real_npz/data_3d_animals.npz')
    keypoints_data = np.load(f'npz/real_npz/data_2d_animals_gt.npz', allow_pickle=True)
    keypoints_2d = keypoints_data['positions_2d'].item()

    # 自动处理对称与副本
    kps_left = list(dataset.skeleton().joints_left())
    kps_right = list(dataset.skeleton().joints_right())
    for sub in dataset.subjects():
        for act in dataset[sub].keys():
            num_views = len(keypoints_2d[sub][act])
            dataset[sub][act]['positions_3d'] = [dataset[sub][act]['positions'].copy() for _ in range(num_views)]

    # 划分与提取
    subs = dataset.subjects()
    # train_subs, test_subs = subs[:int(len(subs) * 0.8)], subs[int(len(subs) * 0.8):]

    # Create (subject, action) pairs for training and testing
    all_data_source = []
    for s in subs:
        for a in dataset[s].keys():
            all_data_source.append((s, a))

    # Split into train and test
    split_idx = int(len(all_data_source) * 0.8)
    train_data_source = all_data_source[:split_idx]
    test_data_source = all_data_source[split_idx:]

    print(f"  Train Source Size: {len(train_data_source)}")
    print(f"  Test Source Size: {len(test_data_source)}")

    def fetch(data_source, label=""):
        o3, o2 = [], []
        skipped = 0
        for (s, a) in data_source:
            if s in keypoints_2d and a in keypoints_2d[s]: # 确保2D数据存在
                for v in range(len(keypoints_2d[s][a])):
                    o2.append(keypoints_2d[s][a][v])
                    # 注意：dataset[s][a]['positions_3d'] 已经在上面被处理成列表了
                    o3.append(dataset[s][a]['positions_3d'][v])
            else:
                skipped += 1
                if skipped <= 5: print(f"    [WARN] Key missing in 2D: {s}/{a}")
        
        if skipped > 0: print(f"    Total skipped in {label}: {skipped}")
        return o3, o2

    t3, t2 = fetch(train_data_source, "Train")
    v3, v2 = fetch(test_data_source, "Validation") # 验证集

    print(f"  由 {len(t2)} 个序列组成训练集")
    print(f"  由 {len(v2)} 个序列组成验证集")
    if len(t2) == 0:
        raise ValueError("训练集为空! 请检查 failing keys match (2D vs 3D) 或 split logic.")
    if len(v2) == 0:
        raise ValueError("验证集为空! 请检查数据划分逻辑.")
    
    # --- 模型与优化 ---
    model = AnimalPoseTransformer(num_joints=t2[0].shape[-2], embed_dim=256, seq_len=SEQ_LEN).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 数据生成器 ---
    train_gen = ChunkedGenerator(args.batch_size, None, t3, t2, chunk_length=SEQ_LEN,
                                 shuffle=True, augment=True, kps_left=kps_left, kps_right=kps_right,
                                 joints_left=kps_left, joints_right=kps_right)
    
    val_gen = ChunkedGenerator(args.batch_size, None, v3, v2, chunk_length=SEQ_LEN,
                               shuffle=False, augment=False, kps_left=kps_left, kps_right=kps_right,
                               joints_left=kps_left, joints_right=kps_right)

    print(f"🚀 启动加强版训练 | SEQ_LEN: {SEQ_LEN}")
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20  # 早停耐心值
    early_stop = False

    for epoch in range(args.epochs):
        # 1. 线性预热
        if epoch < WAPMUP_EPOCHS:
            curr_lr = LR * (epoch + 1) / WAPMUP_EPOCHS
            for param_group in optimizer.param_groups: param_group['lr'] = curr_lr

        model.train()
        epoch_loss = 0
        num_batches = 0

        # 训练阶段
        for _, batch_3d, batch_2d in train_gen.next_epoch():
            in_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)
            gt_3d = torch.from_numpy(batch_3d.astype('float32')).to(device)  
            gt_3d -= gt_3d[:, :, 0:1, :]  # Root-relative

            optimizer.zero_grad()
            pred_3d = model(in_2d)

            # 组合损失函数
            loss_mpjpe = mpjpe(pred_3d, gt_3d)
            loss_bone = bone_length_loss(pred_3d, gt_3d)
            total_loss = loss_mpjpe + 0.1 * loss_bone

            total_loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_mpjpe.item()  # 只记录物理含义明确的 MPJPE
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_batches = 0
         
        with torch.no_grad():
            for _, batch_3d, batch_2d in val_gen.next_epoch():
                in_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)
                gt_3d = torch.from_numpy(batch_3d.astype('float32')).to(device)  # 保持米制单位
                gt_3d -= gt_3d[:, :, 0:1, :]  # Root-relative
                
                pred_3d = model(in_2d)
                loss_mpjpe = mpjpe(pred_3d, gt_3d)
                val_loss += loss_mpjpe.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            os.makedirs(args.checkpoint, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'best_val_loss': best_val_loss
            }, os.path.join(args.checkpoint, 'best_transformer_v2.pt'))
            print(f"💾 保存最佳模型 | 验证损失: {avg_val_loss * 1000:.2f}mm")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                early_stop = True
                print(f"🛑 早停触发 | 连续 {max_patience} 个epoch验证损失未改善")
        
        # 打印训练日志
        print(f"Epoch {epoch + 1:03d} | Train: {avg_train_loss * 1000:.2f}mm | Val: {avg_val_loss * 1000:.2f}mm | LR: {optimizer.param_groups[0]['lr']:.6f} | Patience: {patience_counter}/{max_patience}")
        
        # 检查早停
        if early_stop:
            print(f"🎯 训练完成! 最佳验证损失: {best_val_loss * 1000:.2f}mm")
            break


if __name__ == '__main__':
    train()