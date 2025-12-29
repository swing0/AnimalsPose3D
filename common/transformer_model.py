import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


class AnimalPoseTransformer(nn.Module):
    def __init__(self, num_joints, in_dim=2, embed_dim=256, depth=6, num_heads=8, seq_len=27, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints

        # 1. 关节特征嵌入
        self.joint_embed = nn.Linear(in_dim, embed_dim)

        # 2. 空间编码层
        self.spatial_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(depth // 2)
        ])

        # 3. 时间编码层 + 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        self.temporal_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(depth // 2)
        ])

        # 4. 修正回归头设计
        self.regression_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_joints * 3)  # 输出所有关节的3D坐标
        )

    def forward(self, x):
        b, t, j, c = x.shape

        # 空间建模
        x = x.view(b * t, j, c)
        x = self.joint_embed(x)
        for block in self.spatial_blocks:
            x = block(x)

        # 聚合空间信息
        x = torch.mean(x, dim=1)  # (B*T, Embed)
        x = x.view(b, t, -1)      # (B, T, Embed)

        # 时间建模
        x = x + self.pos_embed
        for block in self.temporal_blocks:
            x = block(x)

        # 回归 3D 坐标
        # 重塑: (B*T, Embed) -> (B*T, J*3) -> (B, T, J, 3)
        x = x.view(b * t, -1)
        x = self.regression_head(x)
        x = x.view(b, t, j, 3)
        
        return x