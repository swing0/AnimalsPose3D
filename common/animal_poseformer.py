import math
import torch
import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FreqMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        b, f, _ = x.shape
        x = dct.dct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = dct.idct(x.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MixedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim)
        self.mlp2 = FreqMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        b, f, c = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :f//2] + self.drop_path(self.mlp1(self.norm2(x[:, :f//2])))
        x2 = x[:, f//2:] + self.drop_path(self.mlp2(self.norm3(x[:, f//2:])))
        return torch.cat((x1, x2), dim=1)

class AnimalPoseFormer(nn.Module):
    def __init__(self, num_frame=27, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 use_lme=True, num_frame_kept=27, num_coeff_kept=27):
        """
        AnimalPoseFormer 升级为基于 PoseTransformerV2 的内核结构，
        它混合了 DCT 频域分析以及局部形态学提取调制 (LME Module)。
        """
        super().__init__()

        self.use_lme = use_lme
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints   
        out_dim = num_joints * 3 

        self.num_frame_kept = num_frame_kept
        self.num_coeff_kept = num_coeff_kept

        # --- 1. 空间与频率嵌入层 ---
        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans * num_joints, embed_dim)

        # --- 2. 形态学调制模块 (LME + FiLM) ---
        if self.use_lme:
            # 提取器：从 2D 序列中学习全局比例和尺度先验
            self.morphology_extractor = nn.Sequential(
                nn.Linear(num_joints * in_chans, embed_dim_ratio * 2),
                nn.LayerNorm(embed_dim_ratio * 2),
                nn.GELU(),
                nn.Linear(embed_dim_ratio * 2, embed_dim_ratio),
                nn.Tanh() # 将调制信号限制在 [-1, 1] 防止梯度爆炸
            )
            # FiLM 调制：生成缩放 gamma 和 平移 beta
            self.film_gamma = nn.Linear(embed_dim_ratio, embed_dim_ratio)
            self.film_beta = nn.Linear(embed_dim_ratio, embed_dim_ratio)
        
        # --- 3. 位置编码 ---
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # --- 4. 空间与时间 Transformer 块 ---
        self.Spatial_blocks = nn.ModuleList([
            Block(dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks = nn.ModuleList([
            MixedBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)
        
        self.weighted_mean = nn.Conv1d(in_channels=self.num_coeff_kept, out_channels=1, kernel_size=1)
        self.weighted_mean_ = nn.Conv1d(in_channels=self.num_frame_kept, out_channels=1, kernel_size=1)
        
        self.head = nn.Sequential(nn.LayerNorm(embed_dim*2), nn.Linear(embed_dim*2, out_dim))

        trunc_normal_(self.Spatial_pos_embed, std=.02)
        trunc_normal_(self.Temporal_pos_embed, std=.02)
        trunc_normal_(self.Temporal_pos_embed_, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def Spatial_forward_features(self, x, gamma, beta, b, f, p, c):
        num_frame_kept = self.num_frame_kept

        index = torch.arange((f-1)//2-num_frame_kept//2, (f-1)//2+num_frame_kept//2+1, device=x.device)

        x_spatial = self.Joint_embedding(x[:, index].view(b*num_frame_kept, p, -1))
        
        # 注入位置编码
        x_spatial += self.Spatial_pos_embed
        
        # 应用形态学调制 (FiLM)
        if self.use_lme and gamma is not None and beta is not None:
            # 将 [B, 1, D] 重复到 [B*F(kept), P, D]
            g = gamma.repeat_interleave(num_frame_kept, dim=0)
            b_ = beta.repeat_interleave(num_frame_kept, dim=0)
            # 调制：通过全局形态对局部特征进行约束
            x_spatial = x_spatial * (1 + g) + b_

        x_spatial = self.pos_drop(x_spatial)
        for blk in self.Spatial_blocks:
            x_spatial = blk(x_spatial)
            
        x_spatial = self.Spatial_norm(x_spatial)
        x_spatial = rearrange(x_spatial, '(b f) p c -> b f (p c)', f=num_frame_kept)
        return x_spatial

    def forward_features(self, x, Spatial_feature):
        b, f, p, _ = x.shape
        num_coeff_kept = self.num_coeff_kept

        x_freq = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        x_freq = x_freq.permute(0, 3, 1, 2).contiguous().view(b, num_coeff_kept, -1)
        x_freq = self.Freq_embedding(x_freq) 
        
        Spatial_feature = Spatial_feature + self.Temporal_pos_embed
        x_freq = x_freq + self.Temporal_pos_embed_
        x_combined = torch.cat((x_freq, Spatial_feature), dim=1)

        for blk in self.blocks:
            x_combined = blk(x_combined)

        x_combined = self.Temporal_norm(x_combined)
        return x_combined

    def forward(self, x):
        # 统一规范输入格式 [B, F, J, 2]
        if x.shape[-1] != 2:
            if x.shape[1] == 2:
                x = x.permute(0, 2, 3, 1).clone()  # [B, 2, F, J] -> [B, F, J, 2]
            
        x_input = x.clone()
        b, f, p, c = x_input.shape
        
        # --- 形态特征提取 ---
        gamma, beta = None, None
        if self.use_lme:
            avg_pose = x_input.mean(dim=1).view(b, -1) # [B, J*2]
            morph_feat = self.morphology_extractor(avg_pose) # [B, D]
            gamma = self.film_gamma(morph_feat).unsqueeze(1) # [B, 1, D]
            beta = self.film_beta(morph_feat).unsqueeze(1)   # [B, 1, D]

        # --- 空间处理 (Spatial Transformer) ---
        Spatial_feature = self.Spatial_forward_features(x_input, gamma, beta, b, f, p, c)
        
        # --- 混合时间处理 (Mixed Temporal & Freq Transformer) ---
        x_out = self.forward_features(x_input, Spatial_feature)
        
        # --- 双分支预测输出合并 ---
        x_out = torch.cat((self.weighted_mean(x_out[:, :self.num_coeff_kept]), 
                           self.weighted_mean_(x_out[:, self.num_coeff_kept:])), dim=-1)

        x_out = self.head(x_out).view(b, 1, p, 3) # [B, 1, J, 3]
        return x_out
