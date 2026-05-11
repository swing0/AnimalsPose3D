import os
import numpy as np
import random

def main():
    # 路径配置
    input_path = r'npz\real_npz\data_3d_animals.npz'
    output_dir = r'npz\real_npz'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"📦 正在加载原始 3D 数据: {input_path}")
    try:
        raw_data = np.load(input_path, allow_pickle=True)
        # 兼容多种可能的 key 名
        key = 'positions_3d' if 'positions_3d' in raw_data else 'positions'
        data = raw_data[key].item()
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return

    subjects = sorted(data.keys())
    train_3d = {}
    val_3d = {}
    test_3d = {}

    random.seed(42)
    
    print("✂️ 正在按动作序列划分数据集 (70% 训练, 10% 验证, 20% 测试)...")

    for sub in subjects:
        actions = sorted(list(data[sub].keys()))
        random.shuffle(actions)
        
        n = len(actions)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)
        train_actions = actions[:train_end]
        val_actions = actions[train_end:val_end]
        test_actions = actions[val_end:]
        
        train_3d[sub] = {}
        val_3d[sub] = {}
        test_3d[sub] = {}
        
        for act in actions:
            pos_3d_raw = data[sub][act]
            
            pos_3d_rel = pos_3d_raw - pos_3d_raw[:, 0:1, :]
            
            if act in train_actions:
                train_3d[sub][act] = pos_3d_rel.astype('float32')
            elif act in val_actions:
                val_3d[sub][act] = pos_3d_rel.astype('float32')
            else:
                test_3d[sub][act] = pos_3d_rel.astype('float32')

    train_out = os.path.join(output_dir, 'animals_train_3d.npz')
    val_out = os.path.join(output_dir, 'animals_val_3d.npz')
    test_out = os.path.join(output_dir, 'animals_test_3d.npz')

    np.savez_compressed(train_out, positions_3d=train_3d)
    np.savez_compressed(val_out, positions_3d=val_3d)
    np.savez_compressed(test_out, positions_3d=test_3d)

    print(f"✅ 处理完成！")
    print(f"   [训练集]: {train_out} ({sum(len(v) for v in train_3d.values())} 条动作)")
    print(f"   [验证集]: {val_out} ({sum(len(v) for v in val_3d.values())} 条动作)")
    print(f"   [测试集]: {test_out} ({sum(len(v) for v in test_3d.values())} 条动作)")
    print(f"💡 提示：2D 投影逻辑现已交给 Dataset 类在训练时实时计算。")

if __name__ == '__main__':
    main()