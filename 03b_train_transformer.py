import os
import sys
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
from tqdm import tqdm

# 添加路径
sys.path.append('./common')

# 导入本地模块
try:
    from common.animals_dataset import AnimalsDataset
    from common.loss import mpjpe
    from common.generators import ChunkedGenerator
    from common.arguments import parse_args
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保common目录存在且包含所需模块")
    sys.exit(1)

# 导入轻量模型
try:
    from common.transformer_model import UltraLightAnimalPoseTransformer
except ImportError:
    print("❌ 无法导入ultra_light_transformer")
    print("请确保ultra_light_transformer.py在同一目录下")
    sys.exit(1)

# ========== 配置和辅助函数 ==========

# 骨架连线
SKELETON_EDGES = [
    (0, 4), (4, 3), (3, 1), (3, 2), (4, 5), (5, 6), (6, 7),
    (4, 8), (8, 9), (9, 10), (0, 11), (11, 12), (12, 13), 
    (0, 14), (14, 15), (15, 16)
]

def bone_length_loss(pred, gt):
    """骨骼长度一致性损失"""
    p1 = pred[:, :, [e[0] for e in SKELETON_EDGES], :]
    p2 = pred[:, :, [e[1] for e in SKELETON_EDGES], :]
    g1 = gt[:, :, [e[0] for e in SKELETON_EDGES], :]
    g2 = gt[:, :, [e[1] for e in SKELETON_EDGES], :]
    
    pred_bones = torch.norm(p1 - p2, dim=-1)
    gt_bones = torch.norm(g1 - g2, dim=-1)
    
    return torch.mean(torch.abs(pred_bones - gt_bones))

def create_balanced_split(all_data_source, train_ratio=0.8, seed=42):
    """平衡数据划分 - 确保每种动物都有训练和验证样本"""
    np.random.seed(seed)
    
    # 按动物分组
    animal_groups = defaultdict(list)
    for animal, action in all_data_source:
        animal_groups[animal].append((animal, action))
    
    train_sources, val_sources = [], []
    
    print("📊 平衡数据划分:")
    for animal, sequences in animal_groups.items():
        # 随机打乱
        np.random.shuffle(sequences)
        
        # 按比例划分
        split_idx = int(len(sequences) * train_ratio)
        
        train_sources.extend(sequences[:split_idx])
        val_sources.extend(sequences[split_idx:])
        
        train_count = len(sequences[:split_idx])
        val_count = len(sequences[split_idx:])
        print(f"  {animal}: {train_count}训练 + {val_count}验证")
    
    print(f"总计: {len(train_sources)}训练, {len(val_sources)}验证")
    return train_sources, val_sources

def truncate_sequence(data, target_length=16):
    """截断或填充序列到目标长度"""
    current_length = len(data)
    
    if current_length >= target_length:
        # 从中间截取
        start = (current_length - target_length) // 2
        return data[start:start + target_length]
    else:
        # 填充
        pad_length = target_length - current_length
        # 边缘填充模式
        return np.pad(data, ((0, pad_length), (0, 0), (0, 0)), mode='edge')

def fetch_data_with_truncation(data_source, keypoints_2d, dataset, target_length=16, label=""):
    """获取数据并截断到统一长度"""
    o3, o2 = [], []
    
    print(f"📥 获取{label}数据 (截断到{target_length}帧)...")
    
    for animal, action in tqdm(data_source, desc=f"处理{label}", ncols=80):
        if animal in keypoints_2d and action in keypoints_2d[animal]:
            num_views = len(keypoints_2d[animal][action])
            for view_idx in range(num_views):
                # 获取原始数据
                seq_2d = keypoints_2d[animal][action][view_idx]
                seq_3d = dataset[animal][action]['positions_3d'][view_idx]
                
                # 截断到统一长度
                seq_2d = truncate_sequence(seq_2d, target_length)
                seq_3d = truncate_sequence(seq_3d, target_length)
                
                o2.append(seq_2d)
                o3.append(seq_3d)
    
    return o3, o2

def check_data_shapes(train_2d, train_3d, val_2d, val_3d):
    """检查数据形状"""
    print("\n📐 数据形状检查:")
    print(f"训练2D: {len(train_2d)}序列, 形状: {train_2d[0].shape}")
    print(f"训练3D: {len(train_3d)}序列, 形状: {train_3d[0].shape}")
    print(f"验证2D: {len(val_2d)}序列, 形状: {val_2d[0].shape}")
    print(f"验证3D: {len(val_3d)}序列, 形状: {val_3d[0].shape}")
    
    # 检查一致性
    assert train_2d[0].shape == val_2d[0].shape, "训练和验证2D形状不一致"
    assert train_3d[0].shape == val_3d[0].shape, "训练和验证3D形状不一致"
    assert train_2d[0].shape[:2] == train_3d[0].shape[:2], "2D和3D时间/关节维度不一致"
    
    print("✅ 数据形状检查通过")

def setup_training_environment():
    """设置训练环境"""
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 没有可用的GPU，退出")
        sys.exit(1)
    
    device = torch.device("cuda")
    
    # 显示GPU信息
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🖥️ GPU: {gpu_name}")
    print(f"💾 显存: {gpu_memory:.1f} GB")
    
    # 设置cuDNN基准
    torch.backends.cudnn.benchmark = True
    
    return device

def create_data_generators(train_3d, train_2d, val_3d, val_2d, 
                          batch_size, seq_len, kps_left, kps_right):
    """创建数据生成器 - 使用兼容的参数"""
    
    try:
        # 尝试使用标准参数
        train_gen = ChunkedGenerator(
            batch_size, 
            None,  # cameras
            train_3d, 
            train_2d,
            chunk_length=seq_len,
            shuffle=True,
            augment=True,  # 只设置augment=True，让内部使用默认参数
            kps_left=kps_left,
            kps_right=kps_right,
            joints_left=kps_left,
            joints_right=kps_right
        )
        
        val_gen = ChunkedGenerator(
            batch_size,
            None,
            val_3d,
            val_2d,
            chunk_length=seq_len,
            shuffle=False,
            augment=False,
            kps_left=kps_left,
            kps_right=kps_right,
            joints_left=kps_left,
            joints_right=kps_right
        )
        
        print("✅ 数据生成器创建成功")
        return train_gen, val_gen
        
    except TypeError as e:
        print(f"⚠️ 标准参数失败: {e}")
        print("尝试使用最简参数...")
        
        # 使用最简参数
        train_gen = ChunkedGenerator(
            batch_size, 
            None, 
            train_3d, 
            train_2d,
            chunk_length=seq_len,
            shuffle=True
        )
        
        val_gen = ChunkedGenerator(
            batch_size,
            None,
            val_3d,
            val_2d,
            chunk_length=seq_len,
            shuffle=False
        )
        
        print("✅ 最简参数数据生成器创建成功")
        return train_gen, val_gen

# ========== 主训练函数 ==========

def train_lightweight():
    """主训练函数 - 专为8GB显存优化"""
    
    # 解析参数
    args = parse_args()
    
    # ========== 超参数配置 ==========
    SEQ_LEN = 16          # 序列长度 (必须小!)
    BATCH_SIZE = 8        # batch大小
    GRAD_ACCUM_STEPS = 4  # 梯度累积步数 (等效batch_size=32)
    LR = 3e-4            # 学习率
    WARMUP_EPOCHS = 5     # 预热轮数
    MAX_EPOCHS = 100      # 最大轮数
    
    # 模型参数
    EMBED_DIM = 96        # 嵌入维度
    DEPTH = 2             # Transformer层数
    HEADS = 4             # 注意力头数
    
    print("=" * 70)
    print("🚀 动物3D姿态估计 - 轻量版训练")
    print("=" * 70)
    print(f"📋 配置:")
    print(f"  序列长度: {SEQ_LEN}")
    print(f"  Batch大小: {BATCH_SIZE} (累积{GRAD_ACCUM_STEPS}步)")
    print(f"  学习率: {LR}")
    print(f"  嵌入维度: {EMBED_DIM}")
    print(f"  Transformer层: {DEPTH}")
    print(f"  注意力头: {HEADS}")
    print("=" * 70)
    
    # 设置环境
    device = setup_training_environment()
    
    # ========== 数据加载 ==========
    print("\n📂 加载数据...")
    
    try:
        dataset = AnimalsDataset('npz/real_npz/data_3d_animals.npz')
        keypoints_data = np.load('npz/real_npz/data_2d_animals_gt.npz', allow_pickle=True)
        keypoints_2d = keypoints_data['positions_2d'].item()
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        sys.exit(1)
    
    # 获取所有动物
    all_animals = dataset.subjects()
    print(f"🦓 动物类型数: {len(all_animals)}")
    print(f"🐾 动物列表: {sorted(all_animals)[:5]}..." if len(all_animals) > 5 else f"🐾 动物列表: {sorted(all_animals)}")
    
    # 获取左右关节
    kps_left = list(dataset.skeleton().joints_left())
    kps_right = list(dataset.skeleton().joints_right())
    
    print("🔄 准备多视角数据...")
    for animal in tqdm(all_animals, desc="处理动物", ncols=80):
        for action in dataset[animal].keys():
            if animal in keypoints_2d and action in keypoints_2d[animal]:
                num_views = len(keypoints_2d[animal][action])
                # 为每个视角创建3D数据副本
                dataset[animal][action]['positions_3d'] = [
                    dataset[animal][action]['positions'].copy() 
                    for _ in range(num_views)
                ]
    
    # ========== 数据划分 ==========
    print("\n📊 数据划分...")
    
    # 收集所有(动物, 动作)对
    all_data_source = []
    for animal in all_animals:
        for action in dataset[animal].keys():
            all_data_source.append((animal, action))
    
    print(f"总动作序列数: {len(all_data_source)}")
    
    # 平衡划分
    train_data_source, val_data_source = create_balanced_split(
        all_data_source, 
        train_ratio=0.8,
        seed=42
    )
    
    # ========== 获取并截断数据 ==========
    train_3d, train_2d = fetch_data_with_truncation(
        train_data_source, keypoints_2d, dataset, SEQ_LEN, "训练"
    )
    
    val_3d, val_2d = fetch_data_with_truncation(
        val_data_source, keypoints_2d, dataset, SEQ_LEN, "验证"
    )
    
    # 检查数据
    check_data_shapes(train_2d, train_3d, val_2d, val_3d)
    
    # ========== 创建模型 ==========
    print("\n🏗️ 创建模型...")
    
    num_joints = train_2d[0].shape[1]
    
    model = UltraLightAnimalPoseTransformer(
        num_joints=num_joints,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=HEADS,
        seq_len=SEQ_LEN,
        dropout=0.1
    ).to(device)
    
    # ========== 优化器和调度器 ==========
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=8,
        verbose=True,
        min_lr=1e-6
    )
    
    # ========== 数据生成器 ==========
    print("\n🔄 创建数据生成器...")
    
    train_gen, val_gen = create_data_generators(
        train_3d, train_2d, val_3d, val_2d,
        BATCH_SIZE, SEQ_LEN, kps_left, kps_right
    )
    
    # ========== 训练循环 ==========
    print("\n🎯 开始训练...")
    print("-" * 90)
    
    # 训练状态
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 25
    early_stop = False
    
    # 创建检查点目录
    checkpoint_dir = args.checkpoint if hasattr(args, 'checkpoint') else 'checkpoints_light'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练历史
    train_history = []
    val_history = []
    
    for epoch in range(MAX_EPOCHS):
        # 1. 学习率预热
        if epoch < WARMUP_EPOCHS:
            warmup_lr = LR * (epoch + 1) / WARMUP_EPOCHS
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
        
        # 2. 训练阶段
        model.train()
        epoch_train_loss = 0.0
        epoch_bone_loss = 0.0
        num_batches = 0
        accum_steps = 0
        
        optimizer.zero_grad()
        
        # 使用tqdm进度条
        pbar = tqdm(train_gen.next_epoch(), 
                   desc=f"Epoch {epoch+1:03d} 训练",
                   total=len(train_gen.pairs) // BATCH_SIZE + 1,
                   ncols=80)
        
        for batch_idx, (_, batch_3d, batch_2d) in enumerate(pbar):
            # 转换为Tensor
            batch_2d_tensor = torch.from_numpy(batch_2d.astype('float32')).to(device)
            batch_3d_tensor = torch.from_numpy(batch_3d.astype('float32')).to(device)
            
            # Root-relative
            batch_3d_tensor = batch_3d_tensor - batch_3d_tensor[:, :, 0:1, :]
            
            # 前向传播
            pred_3d = model(batch_2d_tensor)
            
            # 计算损失
            loss_mpjpe = mpjpe(pred_3d, batch_3d_tensor)
            loss_bone = bone_length_loss(pred_3d, batch_3d_tensor)
            total_loss = (loss_mpjpe + 0.1 * loss_bone) / GRAD_ACCUM_STEPS
            
            # 反向传播
            total_loss.backward()
            
            # 累积损失
            epoch_train_loss += loss_mpjpe.item() * GRAD_ACCUM_STEPS
            epoch_bone_loss += loss_bone.item() * GRAD_ACCUM_STEPS
            accum_steps += 1
            
            # 梯度累积
            if accum_steps % GRAD_ACCUM_STEPS == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                optimizer.zero_grad()
                
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': f"{loss_mpjpe.item()*1000:.1f}mm",
                    'bone': f"{loss_bone.item()*1000:.1f}mm"
                })
        
        # 处理剩余的梯度
        if accum_steps % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            num_batches += 1
        
        # 计算平均损失
        avg_train_loss = epoch_train_loss / accum_steps if accum_steps > 0 else 0
        avg_bone_loss = epoch_bone_loss / accum_steps if accum_steps > 0 else 0
        train_history.append(avg_train_loss)
        
        # 3. 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for _, batch_3d, batch_2d in tqdm(val_gen.next_epoch(),
                                           desc=f"Epoch {epoch+1:03d} 验证",
                                           total=len(val_gen.pairs) // BATCH_SIZE + 1,
                                           ncols=80):
                # 转换为Tensor
                batch_2d_tensor = torch.from_numpy(batch_2d.astype('float32')).to(device)
                batch_3d_tensor = torch.from_numpy(batch_3d.astype('float32')).to(device)
                
                # Root-relative
                batch_3d_tensor = batch_3d_tensor - batch_3d_tensor[:, :, 0:1, :]
                
                # 前向传播
                pred_3d = model(batch_2d_tensor)
                
                # 计算损失
                loss_mpjpe = mpjpe(pred_3d, batch_3d_tensor)
                epoch_val_loss += loss_mpjpe.item()
                val_batches += 1
        
        avg_val_loss = epoch_val_loss / val_batches if val_batches > 0 else float('inf')
        val_history.append(avg_val_loss)
        
        # 4. 学习率调度
        scheduler.step(avg_val_loss)
        
        # 5. 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存完整模型
            model_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_path)
            
            print(f"💾 保存最佳模型 | 验证损失: {avg_val_loss * 1000:.2f}mm")
        else:
            patience_counter += 1
        
        # 6. 早停检查
        if patience_counter >= max_patience:
            early_stop = True
            print(f"🛑 早停触发 | 连续 {max_patience} 个epoch验证损失未改善")
        
        # 7. 打印日志
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1:03d} | "
              f"Train: {avg_train_loss * 1000:6.1f}mm | "
              f"Val: {avg_val_loss * 1000:6.1f}mm | "
              f"Bone: {avg_bone_loss * 1000:6.1f}mm | "
              f"LR: {current_lr:.2e} | "
              f"Patience: {patience_counter:2d}/{max_patience}")
        
        print("-" * 90)
        
        # 8. 检查早停
        if early_stop:
            print(f"\n🎯 训练完成!")
            print(f"最佳验证损失: {best_val_loss * 1000:.2f}mm")
            break
    
    # 训练完成
    if not early_stop:
        print(f"\n🎯 达到最大轮数 {MAX_EPOCHS}")
        print(f"最佳验证损失: {best_val_loss * 1000:.2f}mm")
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    
    # 打印总结
    print(f"\n📈 训练总结:")
    print(f"  最佳验证损失: {best_val_loss * 1000:.2f}mm")
    print(f"  训练轮数: {len(train_history)}")
    print(f"  检查点保存到: {checkpoint_dir}")
    
    return best_val_loss


def main():
    """主函数"""
    try:
        best_loss = train_lightweight()
        print(f"\n✅ 训练成功完成! 最佳损失: {best_loss * 1000:.2f}mm")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()