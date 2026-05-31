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


class HyperHead(nn.Module):
    """
    超网络解码头：用形态向量生成物种特异的线性解码权重
    每个物种得到自己的解码矩阵 W 和偏置 b，实现"共享特征 + 特异输出"
    """
    def __init__(self, in_dim, out_dim, morph_dim=64):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.base_weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.base_bias = nn.Parameter(torch.empty(out_dim))
        trunc_normal_(self.base_weight, std=.02)
        nn.init.zeros_(self.base_bias)

        self.weight_delta = nn.Linear(morph_dim, in_dim * out_dim)
        self.bias_delta = nn.Linear(morph_dim, out_dim)
        nn.init.zeros_(self.weight_delta.weight)
        nn.init.zeros_(self.weight_delta.bias)
        nn.init.zeros_(self.bias_delta.weight)
        nn.init.zeros_(self.bias_delta.bias)

    def forward(self, x, morph):
        B = x.shape[0]
        W = self.base_weight + self.weight_delta(morph).view(B, self.in_dim, self.out_dim)
        b = self.base_bias + self.bias_delta(morph).view(B, self.out_dim)
        return torch.bmm(x.unsqueeze(1), W).squeeze(1) + b


class MorphCrossAttention(nn.Module):
    """
    形态交叉注意力: morph → Query, 特征 → Key/Value
    作用: 让模型"知道"当前物种，据此调整每个 token 的注意力分布
    """
    def __init__(self, dim, morph_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(morph_dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        nn.init.zeros_(self.to_q.weight)
        nn.init.zeros_(self.to_q.bias)

    def forward(self, x, morph):
        B, N, C = x.shape

        q = self.to_q(morph).unsqueeze(1)
        k, v = self.to_kv(x).chunk(2, dim=-1)

        q = q.view(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = v.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)

        x_out = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_out = self.proj(x_out)

        return x + x_out.expand(-1, N, -1)


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

    def forward(self, x, morph=None):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x1 = x[:, :x.shape[1] // 2] + self.drop_path(self.mlp1(self.norm2(x[:, :x.shape[1] // 2])))
        x2 = x[:, x.shape[1] // 2:] + self.drop_path(self.mlp2(self.norm3(x[:, x.shape[1] // 2:])))
        return torch.cat((x1, x2), dim=1)


class QuadVideo3D(nn.Module):
    MORPH_EDGES = [
        (0, 4), (4, 3), (3, 1), (3, 2),
        (4, 5), (5, 6), (6, 7),
        (4, 8), (8, 9), (9, 10),
        (0, 11), (11, 12), (12, 13),
        (0, 14), (14, 15), (15, 16)
    ]
    NUM_BONES = len(MORPH_EDGES)

    def __init__(self, num_frame=27, num_joints=17, in_chans=2, embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None,
                 use_hyper_head=False, use_morph_cross_attn=False, morph_dim=64,
                 num_frame_kept=27, num_coeff_kept=27):
        super().__init__()

        self.use_hyper_head = use_hyper_head
        self.use_morph_cross_attn = use_morph_cross_attn
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio * num_joints
        out_dim = num_joints * 3

        self.num_frame_kept = num_frame_kept
        self.num_coeff_kept = num_coeff_kept

        self.Joint_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Freq_embedding = nn.Linear(in_chans * num_joints, embed_dim)

        self.morph_dim = morph_dim
        self.morph_encoder = nn.Sequential(
            nn.Linear(self.NUM_BONES, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, morph_dim),
            nn.Tanh()
        )

        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, embed_dim))
        self.Temporal_pos_embed_ = nn.Parameter(torch.zeros(1, self.num_coeff_kept, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

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

        if use_morph_cross_attn:
            self.morph_cross_attn = MorphCrossAttention(embed_dim, morph_dim, num_heads=4)

        if use_hyper_head:
            self.head_norm = nn.LayerNorm(embed_dim * 2)
            self.head = HyperHead(embed_dim * 2, out_dim, morph_dim)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(embed_dim * 2, out_dim)
            )

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

    def _compute_bone_lengths_2d(self, x):
        b, f, j, c = x.shape
        device = x.device
        edges = torch.tensor(self.MORPH_EDGES, device=device)

        start = x[:, :, edges[:, 0]]
        end = x[:, :, edges[:, 1]]
        bone_vec = end - start
        bone_len = torch.norm(bone_vec, dim=-1)

        bone_len_mean = bone_len.mean(dim=1)

        torso_len = bone_len_mean[:, 0:1] + 1e-8
        bone_ratios = bone_len_mean / torso_len

        return bone_ratios

    def encode_morphology(self, x):
        bone_ratios = self._compute_bone_lengths_2d(x)
        morph = self.morph_encoder(bone_ratios)
        return morph

    def forward(self, x):
        if x.shape[-1] != 2:
            if x.shape[1] == 2: x = x.permute(0, 2, 3, 1).clone()

        x_input = x.clone()
        b, f, p, c = x_input.shape
        num_frame_kept = self.num_frame_kept
        num_coeff_kept = self.num_coeff_kept

        morph = None
        if self.use_hyper_head or self.use_morph_cross_attn:
            morph = self.encode_morphology(x_input)

        index = torch.arange((f - 1) // 2 - num_frame_kept // 2, (f - 1) // 2 + num_frame_kept // 2 + 1,
                             device=x.device)
        Spatial_feature = self.Joint_embedding(x_input[:, index].view(b * num_frame_kept, p, -1))
        Spatial_feature += self.Spatial_pos_embed
        Spatial_feature = self.pos_drop(Spatial_feature)

        for blk in self.Spatial_blocks:
            Spatial_feature = blk(Spatial_feature)

        Spatial_feature = self.Spatial_norm(Spatial_feature)
        Spatial_feature = rearrange(Spatial_feature, '(b f) p c -> b f (p c)', f=num_frame_kept)

        x_freq = dct.dct(x_input.permute(0, 2, 3, 1))[:, :, :, :num_coeff_kept]
        x_freq = x_freq.permute(0, 3, 1, 2).contiguous().view(b, num_coeff_kept, -1)
        x_freq = self.Freq_embedding(x_freq)

        Spatial_feature = Spatial_feature + self.Temporal_pos_embed
        x_freq = x_freq + self.Temporal_pos_embed_
        x_combined = torch.cat((x_freq, Spatial_feature), dim=1)

        for blk in self.blocks:
            x_combined = blk(x_combined, morph)

        x_combined = self.Temporal_norm(x_combined)

        if self.use_morph_cross_attn and morph is not None:
            x_combined = self.morph_cross_attn(x_combined, morph)

        x_out = torch.cat((self.weighted_mean(x_combined[:, :num_coeff_kept]),
                           self.weighted_mean_(x_combined[:, num_coeff_kept:])), dim=-1)

        if self.use_hyper_head and morph is not None:
            x_out = self.head_norm(x_out.squeeze(1))
            x_out = self.head(x_out, morph)
        else:
            x_out = self.head(x_out.squeeze(1))

        x_out = x_out.view(b, 1, p, 3)
        return x_out
