# ============================================================
# featurework.py - Kronos 时序模型核心实现
# 合并自 Kronos 官方 kronos.py + module.py，逻辑未修改
# 包含: BSQuantizer 量化器、KronosTokenizer 编解码器、
#       Kronos Predictor 自回归预测器、辅助采样函数
# ============================================================

import math

from einops import rearrange, reduce
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import pandas as pd
from huggingface_hub import PyTorchModelHubMixin
from tqdm import trange


class DifferentiableEntropyFunction(Function):
    """可微分码本熵计算（前向算熵，反向传梯度）"""
    @staticmethod
    def forward(ctx, zq, basis, K, eps):
        zb = (zq + 1) / 2
        zi = ((zb * basis).sum(-1)).to(torch.int64)
        cnt = torch.scatter_reduce(torch.zeros(2 ** K, device=zq.device, dtype=zq.dtype),
                                   0,
                                   zi.flatten(),
                                   torch.ones_like(zi.flatten()).to(zq.dtype),
                                   'sum')
        prob = (cnt + eps) / (cnt + eps).sum()
        H = -(prob * torch.log(prob)).sum()
        ctx.save_for_backward(zq, zi, prob)
        ctx.K = K
        return H

    @staticmethod
    def backward(ctx, grad_output):
        zq, zi, prob = ctx.saved_tensors
        grad_array = -grad_output * (torch.log(prob) + 1) / zi.numel() / ctx.K
        reord_grad = grad_array[zi.flatten()].reshape(zi.shape)
        grad_input = reord_grad.unsqueeze(-1) * zq
        return grad_input, None, None, None, None


def codebook_entropy(zq, basis, K, eps=1e-4):
    return DifferentiableEntropyFunction.apply(zq, basis, K, eps)


class BinarySphericalQuantizer(nn.Module):
    """二值球面量化器：将连续向量量化为 ±1 的二值码，支持软熵训练"""
    def __init__(self, embed_dim, beta, gamma0, gamma, zeta,
                 input_format='bchw',
                 soft_entropy=True, group_size=9,
                 persample_entropy_compute='analytical',
                 cb_entropy_compute='group',
                 l2_norm=True,
                 inv_temperature=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.input_format = input_format
        assert self.embed_dim % group_size == 0, "embed_dim must be divisible by group_size"
        self.num_groups = self.embed_dim // group_size
        self.group_size = group_size
        assert persample_entropy_compute in ['group', 'analytical'], "persample_entropy_compute must be either 'group' or 'analytical'"
        assert cb_entropy_compute in ['group', 'nce'], "cb_entropy_compute must be either 'group' or 'nce'"
        self.persample_entropy_compute = persample_entropy_compute
        self.cb_entropy_compute = cb_entropy_compute
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature

        self.register_buffer('basis', 2 ** torch.arange(embed_dim - 1, -1, -1))
        self.register_buffer('group_basis', 2 ** torch.arange(group_size - 1, -1, -1))

        self.num_dimensions = 2 ** embed_dim
        self.bits_per_index = embed_dim

        group_codes = torch.arange(2 ** self.group_size)
        group_codebook = self.indexes_to_codes(group_codes).float()[:, -group_size:]
        self.register_buffer('group_codebook', group_codebook, persistent=False)

        self.soft_entropy = soft_entropy

    def quantize(self, z):
        assert z.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {z.shape[-1]}"
        zhat = torch.where(z > 0,
                           torch.tensor(1, dtype=z.dtype, device=z.device),
                           torch.tensor(-1, dtype=z.dtype, device=z.device))
        return z + (zhat - z).detach()

    def forward(self, z, collect_metrics=True):
        zq = self.quantize(z)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        zq = zq * q_scale

        if not collect_metrics:
            return zq, zq.new_zeros(()), {}

        indices = self.codes_to_indexes(zq.detach())
        group_indices = self.codes_to_group_indexes(zq.detach())
        if not self.training:
            used_codes = torch.unique(indices, return_counts=False)
        else:
            used_codes = None

        if self.soft_entropy:
            persample_entropy, cb_entropy, avg_prob = self.soft_entropy_loss(z)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy
        else:
            zb_by_sample = ((zq + 1) / 2).reshape(z.shape[0], -1, z.shape[-1]).to(torch.float32)
            persample_entropy = self.get_hard_per_sample_entropy(zb_by_sample)
            cb_entropy = codebook_entropy(zq, self.basis, self.embed_dim)
            entropy_penalty = self.gamma0 * persample_entropy - self.gamma * cb_entropy

        commit_loss = self.beta * torch.mean(((zq.detach() - z) ** 2).sum(dim=-1))

        return (
            zq,
            commit_loss + self.zeta * entropy_penalty / self.inv_temperature,
            {"H": cb_entropy, "used_codes": used_codes, "indices": indices, "group_indices": group_indices,
             "avg_prob": avg_prob}
        )

    def soft_entropy_loss(self, z):
        group_code_book = self.group_codebook / (self.embed_dim ** 0.5 if self.l2_norm else 1)
        divided_z = rearrange(z, '... (g c) -> ... g c', c=self.group_size)
        distance = - 2 * torch.einsum('... g c, d c ->... g d', divided_z, group_code_book)
        prob = (-distance * self.inv_temperature).softmax(dim=-1)
        if self.persample_entropy_compute == 'analytical':
            if self.l2_norm:
                p = torch.sigmoid(-4 * z / (self.embed_dim ** 0.5) * self.inv_temperature)
            else:
                p = torch.sigmoid(-4 * z * self.inv_temperature)
            prob = torch.stack([p, 1 - p], dim=-1)
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()
        else:
            per_sample_entropy = self.get_entropy(prob, dim=-1, normalize=False).sum(dim=-1).mean()

        avg_prob = reduce(prob, '... g d ->g d', 'mean')
        codebook_entropy = self.get_entropy(avg_prob, dim=-1, normalize=False)
        return per_sample_entropy, codebook_entropy.sum(), avg_prob

    def get_hard_per_sample_entropy(self, zb_by_sample):
        probs_per_dim = zb_by_sample.sum(1) / zb_by_sample.shape[1]
        persample_entropy = - probs_per_dim * torch.log(probs_per_dim + 1e-8) - (1 - probs_per_dim) * torch.log(1 - probs_per_dim + 1e-8)
        persample_entropy = persample_entropy.sum(-1)
        return persample_entropy.mean()

    def codes_to_indexes(self, zhat):
        assert zhat.shape[-1] == self.embed_dim, f"Expected {self.embed_dim} dimensions, got {zhat.shape[-1]}"
        return ((zhat + 1) / 2 * self.basis).sum(axis=-1).to(torch.int64)

    def codes_to_group_indexes(self, zhat):
        zhat_in_group = rearrange(zhat, 'b ... (g c) -> b ... g c', c=self.group_size)
        return ((zhat_in_group + 1) / 2 * self.group_basis).sum(axis=-1).to(torch.int64)

    def indexes_to_codes(self, indices):
        indices = indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(indices, self.basis), 2
        )
        return codes_non_centered * 2 - 1

    def group_indexes_to_codes(self, group_indices):
        group_indices = group_indices.unsqueeze(-1)
        codes_non_centered = torch.remainder(
            torch.floor_divide(group_indices, self.group_basis), 2
        )
        codes_non_centered = rearrange(codes_non_centered, 'b ... g c -> b ... (g c)')
        return codes_non_centered * 2 - 1

    def get_entropy(self, count, dim=-1, eps=1e-4, normalize=True):
        if normalize:
            probs = (count + eps) / (count + eps).sum(dim=dim, keepdim=True)
        else:
            probs = count
        H = -(probs * torch.log(probs + 1e-8)).sum(dim=dim)
        return H

    def get_group_codebook_entry(self, group_indices):
        z_q = self.group_indexes_to_codes(group_indices)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        z_q = z_q * q_scale
        if self.input_format == 'bchw':
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q

    def get_codebook_entry(self, indices):
        z_q = self.indexes_to_codes(indices)
        q_scale = 1. / (self.embed_dim ** 0.5) if self.l2_norm else 1.
        z_q = z_q * q_scale
        if self.input_format == 'bchw':
            h, w = int(z_q.shape[1] ** 0.5)
            assert h * w == z_q.shape[1], 'Invalid sequence length'
            z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h)
        return z_q


class BSQuantizer(nn.Module):
    """双层级 BSQ 量化器：将编码向量分为 s1/s2 两段独立量化，输出 token 索引"""
    def __init__(self, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.codebook_dim = s1_bits + s2_bits
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.bsq = BinarySphericalQuantizer(self.codebook_dim, beta, gamma0, gamma, zeta, group_size=group_size)

    def bits_to_indices(self, bits):
        bits = (bits >= 0).to(torch.long)
        indices = 2 ** torch.arange(
            0,
            bits.shape[-1],
            1,
            dtype=torch.long,
            device=bits.device,
        )
        return (bits * indices).sum(-1)

    def forward(self, z, half=False, collect_metrics=True):
        z = F.normalize(z, dim=-1)
        quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
        if half:
            q_pre = quantized[:, :, :self.s1_bits]
            q_post = quantized[:, :, self.s1_bits:]
            z_indices = [self.bits_to_indices(q_pre), self.bits_to_indices(q_post)]
        else:
            z_indices = self.bits_to_indices(quantized)
        return bsq_loss, quantized, z_indices


class RMSNorm(torch.nn.Module):
    """RMS 归一化（LayerNorm 的无偏置变体）"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    """SwiGLU 前馈网络（门控线性单元变体）"""
    def __init__(self, d_model, ff_dim, ffn_dropout_p=0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim, bias=False)
        self.w3 = nn.Linear(d_model, ff_dim, bias=False)
        self.w2 = nn.Linear(ff_dim, d_model, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 (RoPE)：通过旋转矩阵为 q/k 注入位置信息"""
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_cache(self, x, seq_len):
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached, self.sin_cached

    def forward(self, q, k):
        cos, sin = self._update_cos_sin_cache(q, q.shape[-2])
        return (
            (q * cos) + (self._rotate_half(q) * sin),
            (k * cos) + (self._rotate_half(k) * sin),
        )

    def _rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttentionWithRoPE(nn.Module):
    """自注意力层 + RoPE，使用 SDPA 加速，支持因果掩码"""
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, seq_len, -1)
        else:
            attn_mask = None

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=True
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class MultiHeadCrossAttentionWithRoPE(nn.Module):
    """交叉注意力层 + RoPE，用于 s2 预测时 attend 到 s1 的上下文"""
    def __init__(self, d_model, n_heads, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary = RotaryPositionalEmbedding(self.head_dim)
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        batch_size, q_len, _ = query.shape
        _, seq_len, _ = key.shape

        q = self.q_proj(query).view(batch_size, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k = self.rotary(q, k)

        if key_padding_mask is not None:
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_mask = attn_mask.expand(-1, self.n_heads, q_len, -1)
        else:
            attn_mask = None

        is_causal_flag = self.training

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal_flag
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.resid_dropout(self.out_proj(attn_output))


class HierarchicalEmbedding(nn.Module):
    """层级 Embedding：将 s1/s2 两段 token 索引映射为融合表示"""
    def __init__(self, s1_bits, s2_bits, d_model=256):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits

        vocab_s1 = 2 ** s1_bits
        vocab_s2 = 2 ** s2_bits

        self.emb_s1 = nn.Embedding(vocab_s1, d_model)
        self.emb_s2 = nn.Embedding(vocab_s2, d_model)
        self.d_model = d_model
        self.fusion_proj = nn.Linear(d_model * 2, d_model)

        nn.init.normal_(self.emb_s1.weight, mean=0, std=d_model ** -0.5)
        nn.init.normal_(self.emb_s2.weight, mean=0, std=d_model ** -0.5)

    def split_token(self, token_ids: torch.Tensor, s2_bits: int):
        assert isinstance(s2_bits, int) and s2_bits > 0, "s2_bits must be a positive integer"
        t = token_ids.long()
        mask = (1 << s2_bits) - 1
        s2_ids = t & mask
        s1_ids = t >> s2_bits
        return s1_ids, s2_ids

    def forward(self, token_ids):
        if isinstance(token_ids, tuple) or isinstance(token_ids, list):
            s1_ids, s2_ids = token_ids
        else:
            s1_ids, s2_ids = self.split_token(token_ids, self.s2_bits)
        s1_emb = self.emb_s1(s1_ids) * math.sqrt(self.d_model)
        s2_emb = self.emb_s2(s2_ids) * math.sqrt(self.d_model)
        return self.fusion_proj(torch.cat([s1_emb, s2_emb], dim=-1))


class DependencyAwareLayer(nn.Module):
    """依赖感知层：通过交叉注意力让 s2 token 条件依赖 s1 token"""
    def __init__(self, d_model, n_heads=4, attn_dropout_p=0.0, resid_dropout=0.0):
        super().__init__()
        self.cross_attn = MultiHeadCrossAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states, sibling_embed, key_padding_mask=None):
        attn_out = self.cross_attn(
            query=sibling_embed,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask
        )
        return self.norm(hidden_states + attn_out)


class TransformerBlock(nn.Module):
    """标准 Transformer 残差块：自注意力 + SwiGLU 前馈"""
    def __init__(self, d_model, n_heads, ff_dim=1024, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttentionWithRoPE(d_model, n_heads, attn_dropout_p, resid_dropout_p)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, ffn_dropout_p)

    def forward(self, x, key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        attn_out = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + attn_out

        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = residual + ffn_out
        return x


class DualHead(nn.Module):
    """双预测头：分别预测 s1/s2 两级 token 的 logits，支持交叉熵损失计算"""
    def __init__(self, s1_bits, s2_bits, d_model):
        super().__init__()
        self.vocab_s1 = 2 ** s1_bits
        self.vocab_s2 = 2 ** s2_bits
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)

    def compute_loss(self, s1_logits, s2_logits, s1_targets, s2_targets, padding_mask=None):
        if padding_mask is not None:
            valid_mask = (padding_mask == 0)
            s1_logits = s1_logits[valid_mask]
            s2_logits = s2_logits[valid_mask]
            s1_targets = s1_targets[valid_mask]
            s2_targets = s2_targets[valid_mask]
            ce_s1 = F.cross_entropy(s1_logits, s1_targets)
            ce_s2 = F.cross_entropy(s2_logits, s2_targets)
        else:
            ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
            ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
        ce_loss = (ce_s1 + ce_s2) / 2
        return ce_loss, ce_s1, ce_s2

    def forward(self, x):
        return self.proj_s1(x)

    def cond_forward(self, x2):
        return self.proj_s2(x2)


class FixedEmbedding(nn.Module):
    """固定正弦位置 Embedding（不可学习）"""
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """时间特征 Embedding：分钟/小时/星期/日/月，支持可学习或固定"""
    def __init__(self, d_model, learn_pe):
        super(TemporalEmbedding, self).__init__()
        minute_size = 60
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        Embed = FixedEmbedding if not learn_pe else nn.Embedding
        self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 0])
        hour_x = self.hour_embed(x[:, :, 1])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 3])
        month_x = self.month_embed(x[:, :, 4])
        return hour_x + weekday_x + day_x + month_x + minute_x


class KronosTokenizer(nn.Module, PyTorchModelHubMixin):
    """Kronos Tokenizer：编码器-解码器结构，将连续 K 线数据量化为离散 token

    编码流程: 输入 -> 线性映射 -> Transformer 编码器 -> BSQ 量化 -> token 索引
    解码流程: token 索引 -> 还原为量化向量 -> Transformer 解码器 -> 线性映射 -> 原始数据
    """
    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers, ffn_dropout_p, attn_dropout_p, resid_dropout_p, s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.enc_layers = n_enc_layers
        self.dec_layers = n_dec_layers
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p

        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        self.embed = nn.Linear(self.d_in, self.d_model)
        self.head = nn.Linear(self.d_model, self.d_in)

        self.encoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.enc_layers - 1)
        ])
        self.decoder = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.dec_layers - 1)
        ])
        self.quant_embed = nn.Linear(in_features=self.d_model, out_features=self.codebook_dim)
        self.post_quant_embed_pre = nn.Linear(in_features=self.s1_bits, out_features=self.d_model)
        self.post_quant_embed = nn.Linear(in_features=self.codebook_dim, out_features=self.d_model)
        self.tokenizer = BSQuantizer(self.s1_bits, self.s2_bits, beta, gamma0, gamma, zeta, group_size)

    def forward(self, x):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z)
        quantized_pre = quantized[:, :, :self.s1_bits]
        z_pre = self.post_quant_embed_pre(quantized_pre)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z_pre = layer(z_pre)
        z_pre = self.head(z_pre)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return (z_pre, z), bsq_loss, quantized, z_indices

    def indices_to_bits(self, x, half=False):
        if half:
            x1 = x[0]
            x2 = x[1]
            mask = 2 ** torch.arange(self.codebook_dim//2, device=x1.device, dtype=torch.long)
            x1 = (x1.unsqueeze(-1) & mask) != 0
            x2 = (x2.unsqueeze(-1) & mask) != 0
            x = torch.cat([x1, x2], dim=-1)
        else:
            mask = 2 ** torch.arange(self.codebook_dim, device=x.device, dtype=torch.long)
            x = (x.unsqueeze(-1) & mask) != 0
        x = x.float() * 2 - 1
        q_scale = 1. / (self.codebook_dim ** 0.5)
        x = x * q_scale
        return x

    def encode(self, x, half=False):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        z = self.quant_embed(z)
        bsq_loss, quantized, z_indices = self.tokenizer(z, half=half, collect_metrics=False)
        return z_indices

    def decode(self, x, half=False):
        quantized = self.indices_to_bits(x, half)
        z = self.post_quant_embed(quantized)
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z


class Kronos(nn.Module, PyTorchModelHubMixin):
    """Kronos Predictor：Transformer Decoder 自回归预测未来 token

    结构: 层级 Embedding + 时间 Embedding -> N 层 Transformer -> RMSNorm
         -> DualHead 预测 s1 -> DependencyAwareLayer 条件预测 s2
    """
    def __init__(self, s1_bits, s2_bits, n_layers, d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p, token_dropout_p, learn_te):
        super().__init__()
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.learn_te = learn_te
        self.ff_dim = ff_dim
        self.ffn_dropout_p = ffn_dropout_p
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.token_dropout_p = token_dropout_p

        self.s1_vocab_size = 2 ** self.s1_bits
        self.token_drop = nn.Dropout(self.token_dropout_p)
        self.embedding = HierarchicalEmbedding(self.s1_bits, self.s2_bits, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model, self.learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.ff_dim, self.ffn_dropout_p, self.attn_dropout_p, self.resid_dropout_p)
            for _ in range(self.n_layers)
        ])
        self.norm = RMSNorm(self.d_model)
        self.dep_layer = DependencyAwareLayer(self.d_model)
        self.head = DualHead(self.s1_bits, self.s2_bits, self.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=self.embedding.d_model ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, s1_ids, s2_ids, stamp=None, padding_mask=None, use_teacher_forcing=False, s1_targets=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)

        x = self.norm(x)
        s1_logits = self.head(x)

        if use_teacher_forcing:
            sibling_embed = self.embedding.emb_s1(s1_targets)
        else:
            s1_probs = F.softmax(s1_logits.detach(), dim=-1)
            sample_s1_ids = torch.multinomial(s1_probs.view(-1, self.s1_vocab_size), 1).view(s1_ids.shape)
            sibling_embed = self.embedding.emb_s1(sample_s1_ids)

        x2 = self.dep_layer(x, sibling_embed, key_padding_mask=padding_mask)
        s2_logits = self.head.cond_forward(x2)
        return s1_logits, s2_logits

    def decode_s1(self, s1_ids, s2_ids, stamp=None, padding_mask=None):
        x = self.embedding([s1_ids, s2_ids])
        if stamp is not None:
            time_embedding = self.time_emb(stamp)
            x = x + time_embedding
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        x = self.norm(x)
        s1_logits = self.head(x)
        return s1_logits, x

    def decode_s2(self, context, s1_ids, padding_mask=None):
        sibling_embed = self.embedding.emb_s1(s1_ids)
        x2 = self.dep_layer(context, sibling_embed, key_padding_mask=padding_mask)
        return self.head.cond_forward(x2)


def top_k_top_p_filtering(
        logits,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
):
    """Top-k + Top-p 联合过滤：保留概率最高的若干 token，截断低概率尾部"""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        return logits

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        return logits


def sample_from_logits(logits, temperature=1.0, top_k=None, top_p=None, sample_logits=True):
    """从 logits 采样：温度缩放 -> top-k/top-p 过滤 -> 多项采样或 argmax"""
    logits = logits / temperature
    if top_k is not None or top_p is not None:
        if top_k > 0 or top_p < 1.0:
            logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if not sample_logits:
        _, x = top_k(probs, k=1, dim=-1)
    else:
        x = torch.multinomial(probs, num_samples=1)
    return x


def auto_regressive_inference(tokenizer, model, x, x_stamp, y_stamp, max_context, pred_len, clip=5, T=1.0, top_k=0, top_p=0.99, sample_count=5, verbose=False):
    """自回归推理：逐 token 编码历史 -> 采样预测 -> 解码为价格

    使用滑动窗口缓冲区管理上下文，超出 max_context 时滚动淘汰旧 token。
    支持 sample_count 次采样后取均值，提高预测稳定性。
    """
    with torch.no_grad():
        x = torch.clip(x, -clip, clip)
        device = x.device
        x = x.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x.size(1), x.size(2)).to(device)
        x_stamp = x_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, x_stamp.size(1), x_stamp.size(2)).to(device)
        y_stamp = y_stamp.unsqueeze(1).repeat(1, sample_count, 1, 1).reshape(-1, y_stamp.size(1), y_stamp.size(2)).to(device)

        x_token = tokenizer.encode(x, half=True)
        initial_seq_len = x.size(1)
        batch_size = x_token[0].size(0)
        total_seq_len = initial_seq_len + pred_len
        full_stamp = torch.cat([x_stamp, y_stamp], dim=1)

        generated_pre = x_token[0].new_empty(batch_size, pred_len)
        generated_post = x_token[1].new_empty(batch_size, pred_len)

        pre_buffer = x_token[0].new_zeros(batch_size, max_context)
        post_buffer = x_token[1].new_zeros(batch_size, max_context)
        buffer_len = min(initial_seq_len, max_context)
        if buffer_len > 0:
            start_idx = max(0, initial_seq_len - max_context)
            pre_buffer[:, :buffer_len] = x_token[0][:, start_idx:start_idx + buffer_len]
            post_buffer[:, :buffer_len] = x_token[1][:, start_idx:start_idx + buffer_len]

        if verbose:
            ran = trange
        else:
            ran = range
        for i in ran(pred_len):
            current_seq_len = initial_seq_len + i
            window_len = min(current_seq_len, max_context)

            if current_seq_len <= max_context:
                input_tokens = [
                    pre_buffer[:, :window_len],
                    post_buffer[:, :window_len]
                ]
            else:
                input_tokens = [pre_buffer, post_buffer]

            context_end = current_seq_len
            context_start = max(0, context_end - max_context)
            current_stamp = full_stamp[:, context_start:context_end, :].contiguous()

            s1_logits, context = model.decode_s1(input_tokens[0], input_tokens[1], current_stamp)
            s1_logits = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            s2_logits = model.decode_s2(context, sample_pre)
            s2_logits = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits, temperature=T, top_k=top_k, top_p=top_p, sample_logits=True)

            generated_pre[:, i] = sample_pre.squeeze(-1)
            generated_post[:, i] = sample_post.squeeze(-1)

            if current_seq_len < max_context:
                pre_buffer[:, current_seq_len] = sample_pre.squeeze(-1)
                post_buffer[:, current_seq_len] = sample_post.squeeze(-1)
            else:
                pre_buffer.copy_(torch.roll(pre_buffer, shifts=-1, dims=1))
                post_buffer.copy_(torch.roll(post_buffer, shifts=-1, dims=1))
                pre_buffer[:, -1] = sample_pre.squeeze(-1)
                post_buffer[:, -1] = sample_post.squeeze(-1)

        full_pre = torch.cat([x_token[0], generated_pre], dim=1)
        full_post = torch.cat([x_token[1], generated_post], dim=1)

        context_start = max(0, total_seq_len - max_context)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous()
        ]
        z = tokenizer.decode(input_tokens, half=True)
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)
        return preds


def calc_time_stamps(x_timestamp):
    """从时间戳序列提取分钟/小时/星期/日/月五维时间特征"""
    time_df = pd.DataFrame()
    time_df['minute'] = x_timestamp.dt.minute
    time_df['hour'] = x_timestamp.dt.hour
    time_df['weekday'] = x_timestamp.dt.weekday
    time_df['day'] = x_timestamp.dt.day
    time_df['month'] = x_timestamp.dt.month
    return time_df


class KronosPredictor:
    """Kronos 推理封装：标准化 -> 自回归采样 -> 反标准化 -> 输出预测 DataFrame"""
    def __init__(self, model, tokenizer, device=None, max_context=512, clip=5):
        self.tokenizer = tokenizer
        self.model = model
        self.max_context = max_context
        self.clip = clip
        self.price_cols = ['open', 'high', 'low', 'close']
        self.vol_col = 'volume'
        self.amt_vol = 'amount'
        self.time_cols = ['minute', 'hour', 'weekday', 'day', 'month']

        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.tokenizer = self.tokenizer.to(self.device)
        self.model = self.model.to(self.device)

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
        x_tensor = torch.from_numpy(np.array(x).astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(np.array(x_stamp).astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(np.array(y_stamp).astype(np.float32)).to(self.device)

        preds = auto_regressive_inference(self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor, self.max_context, pred_len,
                                          self.clip, T, top_k, top_p, sample_count, verbose)
        preds = preds[:, -pred_len:, :]
        return preds

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_vol] = 0.0
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        x_time_df = calc_time_stamps(x_timestamp)
        y_time_df = calc_time_stamps(y_timestamp)

        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_stamp = x_time_df.values.astype(np.float32)
        y_stamp = y_time_df.values.astype(np.float32)

        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.clip, self.clip)

        x = x[np.newaxis, :]
        x_stamp = x_stamp[np.newaxis, :]
        y_stamp = y_stamp[np.newaxis, :]

        preds = self.generate(x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose)

        preds = preds.squeeze(0)
        preds = preds * (x_std + 1e-5) + x_mean

        pred_df = pd.DataFrame(preds, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp)
        return pred_df

    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, T=1.0, top_k=0, top_p=0.9, sample_count=1, verbose=True):
        if not isinstance(df_list, (list, tuple)) or not isinstance(x_timestamp_list, (list, tuple)) or not isinstance(y_timestamp_list, (list, tuple)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must be list or tuple types.")
        if not (len(df_list) == len(x_timestamp_list) == len(y_timestamp_list)):
            raise ValueError("df_list, x_timestamp_list, y_timestamp_list must have consistent lengths.")

        num_series = len(df_list)
        x_list = []
        x_stamp_list = []
        y_stamp_list = []
        means = []
        stds = []
        seq_lens = []
        y_lens = []

        for i in range(num_series):
            df = df_list[i]
            if not isinstance(df, pd.DataFrame):
                raise ValueError(f"Input at index {i} is not a pandas DataFrame.")
            if not all(col in df.columns for col in self.price_cols):
                raise ValueError(f"DataFrame at index {i} is missing price columns {self.price_cols}.")

            df = df.copy()
            if self.vol_col not in df.columns:
                df[self.vol_col] = 0.0
                df[self.amt_vol] = 0.0
            if self.amt_vol not in df.columns and self.vol_col in df.columns:
                df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)

            if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
                raise ValueError(f"DataFrame at index {i} contains NaN values in price or volume columns.")

            x_timestamp = x_timestamp_list[i]
            y_timestamp = y_timestamp_list[i]

            x_time_df = calc_time_stamps(x_timestamp)
            y_time_df = calc_time_stamps(y_timestamp)

            x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
            x_stamp = x_time_df.values.astype(np.float32)
            y_stamp = y_time_df.values.astype(np.float32)

            if x.shape[0] != x_stamp.shape[0]:
                raise ValueError(f"Inconsistent lengths at index {i}: x has {x.shape[0]} vs x_stamp has {x_stamp.shape[0]}.")
            if y_stamp.shape[0] != pred_len:
                raise ValueError(f"y_timestamp length at index {i} should equal pred_len={pred_len}, got {y_stamp.shape[0]}.")

            x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
            x_norm = (x - x_mean) / (x_std + 1e-5)
            x_norm = np.clip(x_norm, -self.clip, self.clip)

            x_list.append(x_norm)
            x_stamp_list.append(x_stamp)
            y_stamp_list.append(y_stamp)
            means.append(x_mean)
            stds.append(x_std)
            seq_lens.append(x_norm.shape[0])
            y_lens.append(y_stamp.shape[0])

        if len(set(seq_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent historical lengths, got: {seq_lens}")
        if len(set(y_lens)) != 1:
            raise ValueError(f"Parallel prediction requires all series to have consistent prediction lengths, got: {y_lens}")

        x_batch = np.stack(x_list, axis=0).astype(np.float32)
        x_stamp_batch = np.stack(x_stamp_list, axis=0).astype(np.float32)
        y_stamp_batch = np.stack(y_stamp_list, axis=0).astype(np.float32)

        preds = self.generate(x_batch, x_stamp_batch, y_stamp_batch, pred_len, T, top_k, top_p, sample_count, verbose)

        pred_dfs = []
        for i in range(num_series):
            preds_i = preds[i] * (stds[i] + 1e-5) + means[i]
            pred_df = pd.DataFrame(preds_i, columns=self.price_cols + [self.vol_col, self.amt_vol], index=y_timestamp_list[i])
            pred_dfs.append(pred_df)

        return pred_dfs
