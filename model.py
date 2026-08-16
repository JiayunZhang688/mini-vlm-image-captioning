"""
Mini-VLM: 一个从零实现的小型视觉-语言模型（图像描述生成 / Image Captioning）。

设计目的：不依赖 transformers/CLIP 等封装库，手写 Multi-Head Attention、
Self-Attention、Cross-Attention 与 ViT Patch Embedding，用来在简历/面试中证明
"理解 Transformer 内部机制" 而不只是会调 API。

结构：
  ImageEncoder (mini-ViT)：把图像切成 patch，过若干层 Self-Attention 编码器，
      输出一组 patch embedding（就是后面 cross-attention 的 K/V 来源）。
  TextDecoder：自回归 Transformer 解码器，每层包含
      1) 对已生成文本的 causal self-attention
      2) 对图像 patch embedding 的 cross-attention（视觉-语言对齐的核心）
      3) FFN
  MiniVLM：把两者拼起来，训练目标是 teacher-forcing 的下一个词预测（类似 image captioning）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- 基础组件 -----------------------------

class MultiHeadAttention(nn.Module):
    """手写的多头注意力，Q/K/V 可以来自不同序列（用于 cross-attention）。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (B, T, d_model) -> (B, n_heads, T, d_head)
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        """
        query: (B, Tq, d_model)   来自 decoder 自身（self-attn）或文本侧（cross-attn）
        key/value: (B, Tk, d_model)  self-attn 时和 query 相同来源；cross-attn 时来自图像 encoder
        attn_mask: (Tq, Tk) 的 bool/float mask，True/​-inf 表示禁止看到该位置（用于因果掩码）
        """
        B = query.size(0)
        Q = self._split_heads(self.q_proj(query))
        K = self._split_heads(self.k_proj(key))
        V = self._split_heads(self.v_proj(value))

        # Dot-Product Attention: softmax(QK^T / sqrt(d_head)) V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores + attn_mask  # 加性 mask，禁止位置为 -inf

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, n_heads, Tq, d_head)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, -1, self.d_model)
        out = self.out_proj(out)
        return (out, attn) if need_weights else (out, None)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------- 图像编码器 (mini-ViT) -----------------------------

class PatchEmbedding(nn.Module):
    """把 (B, C, H, W) 图像切成不重叠的 patch，并线性投影到 d_model。"""

    def __init__(self, img_size=32, patch_size=8, in_chans=3, d_model=128):
        super().__init__()
        assert img_size % patch_size == 0
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

    def forward(self, x):
        x = self.proj(x)                       # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)        # (B, n_patches, d_model)
        return x + self.pos_embed


class ViTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h, _ = self.attn(x, x, x)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x


class ImageEncoder(nn.Module):
    def __init__(self, img_size=32, patch_size=8, d_model=128, n_heads=4, n_layers=3, d_ff=256):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, d_model)
        self.layers = nn.ModuleList([
            ViTEncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

    def forward(self, images):
        x = self.patch_embed(images)
        for layer in self.layers:
            x = layer(x)
        return x  # (B, n_patches, d_model) —— 后面作为 cross-attention 的 K/V


# ----------------------------- 文本解码器 -----------------------------

class DecoderLayer(nn.Module):
    """标准 Transformer decoder block：causal self-attn -> cross-attn(图像) -> FFN"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, causal_mask, return_cross_attn=False):
        h, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = self.norm1(x + h)

        h, cross_w = self.cross_attn(x, memory, memory, need_weights=return_cross_attn)
        x = self.norm2(x + h)

        x = self.norm3(x + self.ff(x))
        return x, cross_w


class TextDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=3, d_ff=256,
                 max_len=32, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.out_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def causal_mask(T, device):
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf：不能看未来
        return mask

    def forward(self, tokens, memory, return_cross_attn=False):
        B, T = tokens.shape
        x = self.tok_embed(tokens) + self.pos_embed[:, :T, :]
        mask = self.causal_mask(T, tokens.device)

        attn_maps = []
        for layer in self.layers:
            x, cross_w = layer(x, memory, mask, return_cross_attn=return_cross_attn)
            if return_cross_attn:
                attn_maps.append(cross_w)

        logits = self.out_proj(x)
        return (logits, attn_maps) if return_cross_attn else (logits, None)


# ----------------------------- 整体模型 -----------------------------

class MiniVLM(nn.Module):
    def __init__(self, vocab_size, img_size=32, patch_size=8, d_model=128,
                 n_heads=4, n_layers=3, d_ff=256, max_len=32, pad_id=0):
        super().__init__()
        self.encoder = ImageEncoder(img_size, patch_size, d_model, n_heads, n_layers, d_ff)
        self.decoder = TextDecoder(vocab_size, d_model, n_heads, n_layers, d_ff, max_len, pad_id)
        self.max_len = max_len
        self.pad_id = pad_id

    def forward(self, images, input_tokens, return_cross_attn=False):
        memory = self.encoder(images)
        logits, attn_maps = self.decoder(input_tokens, memory, return_cross_attn)
        return logits, attn_maps

    @torch.no_grad()
    def generate(self, images, bos_id, eos_id, return_cross_attn=False):
        """贪心解码，逐词生成，可选返回每步的 cross-attention（用于可视化"模型在看图像哪个区域"）。"""
        self.eval()
        B = images.size(0)
        device = images.device
        memory = self.encoder(images)

        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        all_attn = []
        for _ in range(self.max_len - 1):
            logits, attn_maps = self.decoder(tokens, memory, return_cross_attn=return_cross_attn)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if return_cross_attn:
                # 只存最后一层、最后一个位置对图像 patch 的注意力，用于可视化
                all_attn.append(attn_maps[-1][:, :, -1, :].mean(dim=1))  # (B, n_patches)
            if (next_token == eos_id).all():
                break
        return tokens, all_attn
