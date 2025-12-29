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
    train_subs, test_subs = subs[:int(len(subs) * 0.8)], subs[int(len(subs) * 0.8):]

    def fetch(subjects):
        o3, o2 = [], []
        for s in subjects:
            for a in dataset[s].keys():
                for v in range(len(keypoints_2d[s][a])):
                    o2.append(keypoints_2d[s][a][v]);
                    o3.append(dataset[s][a]['positions_3d'][v])
        return o3, o2

    t3, t2 = fetch(train_subs)

    # --- 模型与优化 ---
    model = AnimalPoseTransformer(num_joints=t2[0].shape[-2], embed_dim=256, seq_len=SEQ_LEN).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    train_gen = ChunkedGenerator(args.batch_size, None, t3, t2, chunk_length=SEQ_LEN,
                                 shuffle=True, augment=True, kps_left=kps_left, kps_right=kps_right,
                                 joints_left=kps_left, joints_right=kps_right)

    print(f"🚀 启动加强版训练 | SEQ_LEN: {SEQ_LEN}")
    best_loss = float('inf')

    for epoch in range(args.epochs):
        # 1. 线性预热
        if epoch < WAPMUP_EPOCHS:
            curr_lr = LR * (epoch + 1) / WAPMUP_EPOCHS
            for param_group in optimizer.param_groups: param_group['lr'] = curr_lr

        model.train()
        epoch_loss = 0
        num_batches = 0

        for _, batch_3d, batch_2d in train_gen.next_epoch():
            in_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)
            gt_3d = torch.from_numpy(batch_3d.astype('float32')).to(device) * 1000  # 转毫米
            gt_3d -= gt_3d[:, :, 0:1, :]  # Root-relative

            optimizer.zero_grad()
            pred_3d = model(in_2d)

            # 组合损失函数
            loss_mpjpe = mpjpe(pred_3d, gt_3d)
            loss_bone = bone_length_loss(pred_3d, gt_3d)
            total_loss = loss_mpjpe + 0.1 * loss_bone

            total_loss.backward()
            optimizer.step()

            epoch_loss += loss_mpjpe.item()  # 只记录物理含义明确的 MPJPE
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        scheduler.step(avg_loss)

        print(f"Epoch {epoch + 1:03d} | MPJPE: {avg_loss:.2f}mm | LR: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.checkpoint, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.checkpoint, 'best_transformer_v2.pt'))


if __name__ == '__main__':
    train()