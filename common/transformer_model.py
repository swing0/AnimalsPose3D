import torch
import torch.nn as nn
import torch.nn.functional as F


class UltraLightAnimalPoseTransformer(nn.Module):
    """
    超轻量动物姿态Transformer - 专为8GB显存优化
    关键优化：
    1. 极小的模型尺寸
    2. 减少序列长度和注意力计算
    3. 混合精度训练友好
    """
    
    def __init__(self, num_joints=17, in_dim=2, embed_dim=96, 
                 depth=2, num_heads=4, seq_len=16, dropout=0.1):
        """
        参数：
            num_joints: 关节数量 (默认17)
            in_dim: 输入维度 (2D坐标=2)
            embed_dim: 嵌入维度 (大幅减少)
            depth: Transformer层数 (大幅减少)
            num_heads: 注意力头数 (减少)
            seq_len: 序列长度 (必须<=16)
            dropout: Dropout率
        """
        super().__init__()
        
        # 保存参数
        self.num_joints = num_joints
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        # 1. 关节特征嵌入 (极简)
        self.joint_embed = nn.Linear(in_dim, embed_dim)
        
        # 2. 位置编码 (学习式)
        self.time_pos_embed = nn.Parameter(torch.randn(1, seq_len, 1, embed_dim) * 0.02)
        self.joint_pos_embed = nn.Parameter(torch.randn(1, 1, num_joints, embed_dim) * 0.02)
        
        # 3. 极简Transformer编码器 (2层)
        self.transformer_layers = nn.ModuleList([
            self._create_transformer_layer(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        
        # 4. 输出层 (极简)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, 3)  # 直接输出3D坐标
        
        # 初始化
        self._init_weights()
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.parameters())
        print(f"🔧 超轻量模型创建完成:")
        print(f"  参数量: {total_params:,}")
        print(f"  序列长度: {seq_len}")
        print(f"  嵌入维度: {embed_dim}")
        print(f"  层数: {depth}")
        print(f"  注意力头: {num_heads}")
    
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
        输入: (B, T, J, 2) - 2D关节点坐标
        输出: (B, T, J, 3) - 3D关节点坐标
        """
        batch_size, seq_len, num_joints, _ = x.shape
        
        # 检查输入序列长度
        if seq_len > self.seq_len:
            raise ValueError(f"输入序列长度{seq_len}超过模型最大长度{self.seq_len}")
        
        # 1. 关节特征嵌入
        x = self.joint_embed(x)  # (B, T, J, D)
        
        # 2. 添加位置编码 (广播)
        x = x + self.time_pos_embed[:, :seq_len, :, :]
        x = x + self.joint_pos_embed[:, :, :num_joints, :]
        
        # 3. 重塑为序列: (B, T*J, D)
        x = x.reshape(batch_size, seq_len * num_joints, self.embed_dim)
        
        # 4. 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 5. 层归一化
        x = self.norm(x)
        
        # 6. 恢复原始形状
        x = x.reshape(batch_size, seq_len, num_joints, self.embed_dim)
        
        # 7. 输出3D坐标
        x = self.output_proj(x)
        
        return x


class TinyAnimalPoseTransformer(nn.Module):
    """
    超小模型 - 如果上面的模型还是太大
    使用卷积+注意力混合架构
    """
    def __init__(self, num_joints=17, seq_len=16, hidden_dim=64):
        super().__init__()
        
        # 1. 卷积编码器 (节省显存)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=(3, 3), padding=1),  # (B, 32, T, J)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((seq_len, num_joints))
        )
        
        # 2. 轻量注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=0.1,
            batch_first=True
        )
        
        # 3. 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
        
        print(f"🔧 超小模型创建完成: {sum(p.numel() for p in self.parameters()):,}参数")
    
    def forward(self, x):
        # x: (B, T, J, 2)
        b, t, j, _ = x.shape
        
        # 转置为卷积格式
        x = x.permute(0, 3, 1, 2)  # (B, 2, T, J)
        x = self.conv_encoder(x)
        x = x.permute(0, 2, 3, 1)  # (B, T, J, 64)
        
        # 重塑并应用注意力
        x = x.reshape(b, t * j, -1)
        x, _ = self.attention(x, x, x)
        
        # 恢复形状并输出
        x = x.reshape(b, t, j, -1)
        x = self.output(x)
        
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
                model = UltraLightAnimalPoseTransformer(
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