import torch
import torch.nn as nn
import torch.nn.functional as F


class AnimalPoseTransformer(nn.Module):
    """
    增强版动物姿态Transformer - 专为8GB显存优化但性能更强
    关键改进：
    1. 增加模型容量 (embed_dim=256, depth=4)
    2. 从2D关键点自动提取形态特征 (无需物种ID)
    3. 支持更长的序列 (seq_len=27)
    4. 优化损失函数
    """
    
    def __init__(self, num_joints=17, in_dim=2, embed_dim=256, 
                 depth=4, num_heads=8, seq_len=27, dropout=0.1):
        """
        参数：
            num_joints: 关节数量 (默认17)
            in_dim: 输入维度 (2D坐标=2)
            embed_dim: 嵌入维度 (增加到256)
            depth: Transformer层数 (增加到4)
            num_heads: 注意力头数 (增加到8)
            seq_len: 序列长度 (增加到27)
            dropout: Dropout率
        """
        super().__init__()
        
        # 保存参数
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.in_dim = in_dim
        
        # 1. 从2D关键点提取全局形态特征
        # 这将自动学习物种特定的形态特征（体型比例、肢体长度等）
        self.pose_feature_extractor = nn.Sequential(
            nn.Linear(num_joints * in_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 2. 关节特征嵌入
        self.joint_embed = nn.Linear(in_dim, embed_dim)
        
        # 3. 位置编码 (学习式)
        self.time_pos_embed = nn.Parameter(torch.randn(1, seq_len, 1, embed_dim) * 0.02)
        
        # 增强位置编码：为左右关节注入不同的偏置值
        joint_pos_embed = torch.randn(1, 1, num_joints, embed_dim) * 0.02
        
        # AP10K关键点定义：左侧关节 [1, 2, 5, 6, 7, 11, 12, 13]
        # 右侧关节 [3, 4, 8, 9, 10, 14, 15, 16]
        left_joints = [1, 2, 5, 6, 7, 11, 12, 13]
        right_joints = [3, 4, 8, 9, 10, 14, 15, 16]
        
        # 为左侧关节添加正偏置，右侧关节添加负偏置
        for j in left_joints:
            if j < num_joints:
                joint_pos_embed[0, 0, j, :] += 0.1  # 左侧偏置  消融2：no Asym. PE
        
        for j in right_joints:
            if j < num_joints:
                joint_pos_embed[0, 0, j, :] -= 0.1  # 右侧偏置  消融2：no Asym. PE
        
        self.joint_pos_embed = nn.Parameter(joint_pos_embed)
        
        # 4. 增强Transformer编码器 (4层)
        self.transformer_layers = nn.ModuleList([
            self._create_transformer_layer(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        
        # 5. 输出层
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, 3)  # 直接输出3D坐标
        

        # 6. 初始化
        self._init_weights()
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔧 增强版模型创建完成:")
        print(f"  参数量: {total_params:,}")
        print(f"  序列长度: {seq_len}")
        print(f"  嵌入维度: {embed_dim}")
        print(f"  层数: {depth}")
        print(f"  注意力头: {num_heads}")
        print(f"  特征提取: 从2D关键点自动学习")
    
    def _create_transformer_layer(self, embed_dim, num_heads, dropout):
        """创建轻量Transformer层"""
        return nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,  # 小FFN
            dropout=dropout,
            batch_first=True,
            activation='relu',  # 使用ReLU节省显存
            norm_first=True  # 先归一化更稳定
        )
    
    def _init_weights(self):
        """初始化权重"""
        # 线性层使用Xavier初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 位置编码已随机初始化
    
    def forward(self, x):
        """
        前向传播
        输入: 
            x: (B, T, J, 2) - 2D关节点坐标
        输出: (B, T, J, 3) - 3D关节点坐标
        """
        batch_size, seq_len, num_joints, _ = x.shape
        
        # 检查输入序列长度
        if seq_len > self.seq_len:
            raise ValueError(f"输入序列长度{seq_len}超过模型最大长度{self.seq_len}")
        
        # 保存原始输入用于特征提取
        x_input = x.clone()
        
        # 1. 关节特征嵌入
        x = self.joint_embed(x)  # (B, T, J, D)
        
        # 2. 从2D关键点提取全局形态特征
        # 对时间维度取平均以获得稳定的姿态表示
        x_flat = x_input.view(batch_size, seq_len, -1)  # (B, T, J*2)
        pose_feat = self.pose_feature_extractor(x_flat.mean(dim=1))  # (B, D)
        pose_feat = pose_feat.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, D)
        x = x + pose_feat  # 添加到关节嵌入中  消融1： no LME 模块 (Morphology Awareness)
        
        # 3. 添加位置编码 (广播)
        x = x + self.time_pos_embed[:, :seq_len, :, :]
        x = x + self.joint_pos_embed[:, :, :num_joints, :]
        
        # 4. 重塑为序列: (B, T*J, D)
        x = x.reshape(batch_size, seq_len * num_joints, self.embed_dim)
        
        # 5. 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 6. 层归一化
        x = self.norm(x)
        
        # 7. 恢复原始形状
        x = x.reshape(batch_size, seq_len, num_joints, self.embed_dim)
        
        # 8. 输出3D坐标
        x = self.output_proj(x)
        
        return x



def test_model_memory():
    """测试模型显存占用"""
    import time
    
    print("\n🧪 测试模型显存占用...")
    
    # 测试配置
    batch_sizes = [2, 4, 8]
    seq_lens = [8, 16]
    
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                # 创建模型
                model = AnimalPoseTransformer(
                    num_joints=17,
                    embed_dim=96,
                    seq_len=seq_len
                ).cuda()
                
                # 创建测试数据
                test_input = torch.randn(batch_size, seq_len, 17, 2).cuda()
                
                # 前向传播
                output = model(test_input)
                
                # 创建虚拟损失并反向传播
                loss = output.mean()
                loss.backward()
                
                # 获取显存使用
                peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
                allocated = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                
                print(f"  batch={batch_size}, seq={seq_len}: "
                      f"峰值{peak_memory:.1f}MB, "
                      f"已分配{allocated:.1f}MB, "
                      f"保留{reserved:.1f}MB")
                
                del model, test_input, output, loss
                torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError:
                print(f"  ❌ batch={batch_size}, seq={seq_len}: OOM错误")
            
            time.sleep(0.5)
    
    print("测试完成!\n")


if __name__ == '__main__':
    # 运行显存测试
    if torch.cuda.is_available():
        test_model_memory()
    else:
        print("没有可用的GPU，跳过显存测试")