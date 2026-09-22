# ============================================================
# train.py - Vanilla Transformer 对照模型训练
# 任务：根据过去 lookback 日 K 线（6 特征），预测未来 5 日 K 线
# 后续 test 脚本用 T+1/T+5 开盘价计算预期收益率完成选股
# 结构：线性嵌入 + 可学习位置编码 + 因果掩码的 Transformer 编码器
#       + 末端全连接回归头（decoder-only 风格的 Vanilla Transformer）
# 训练口径与 kronos/train.py 保持一致（种子/确定性/归一化），
# 保证对比实验公平
# ============================================================
# ========== 所有环境变量必须在 import torch 之前设置 ==========
import os

# 确定性相关：cuBLAS 矩阵乘法、Python 哈希种子
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '42')

import sys
import time
import json
import math
import torch
import torch.nn as nn

# 将项目 code/ 目录加入系统路径，导入共用模块（统一配置/数据/日志）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import SEED, NUM_WORKERS, USE_TF32, configure_determinism, get_device
from common.paths import model_dir
from common.data_utils import (
    generate_samples, KLineDataset, collate_fn,
    seed_worker, make_generator, STOCK_CSV, HS300_CSV, CALENDAR_JSON,
)
from common.featurework import INPUT_DIM
from common.logger_utils import (
    timestamp, setup_logger, save_config_snapshot, MetricsWriter,
)

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== import torch 之后执行统一确定性配置 ==========
configure_determinism()


# ==================== 配置 ====================
class Config:
    """训练配置类，统一管理所有超参数"""
    # 历史窗口大小（输入给模型的 K 线天数）
    lookback = 60
    # 预测窗口大小（预测未来 K 线天数）
    pred_window = 5
    # 滑动窗口步长
    step = 5
    # 输入特征维度（6 原始 + 7 衍生 = 13，见 code/common/featurework.py）
    input_dim = INPUT_DIM
    # 输出特征维度
    output_dim = 6
    # Transformer 模型维度
    d_model = 128
    # 注意力头数
    n_heads = 8
    # 编码器层数
    n_layers = 3
    # 前馈网络维度
    ff_dim = 512
    # dropout 概率
    dropout = 0.1
    # 批次大小
    batch_size = 256
    # 训练轮数
    epochs = 30
    # 学习率
    lr = 1e-4
    # 权重衰减（L2 正则化系数）
    weight_decay = 1e-4
    # 梯度裁剪阈值
    max_grad_norm = 1.0
    # 早停耐心值（训练损失连续 N 轮不下降则停止）
    patience = 8
    # 随机种子
    seed = SEED
    # TF32 开关（USE_TF32=1 时启用）
    use_tf32 = USE_TF32
    # 模型保存目录（环境变量 MODEL_DIR 可覆盖）
    save_dir = model_dir('transformer')
    # 训练设备（GPU 或 CPU）
    device = get_device()
    # DataLoader 工作进程数（环境变量 NUM_WORKERS 可覆盖）
    num_workers = NUM_WORKERS


cfg = Config()


# ==================== Vanilla Transformer 模型定义 ====================
class VanillaTransformer(nn.Module):
    """
    原生（Vanilla）Transformer 序列回归模型：
    1. 输入 (B, lookback, 6) 归一化 K 线 -> 线性投影到 d_model 维
    2. 加可学习位置编码
    3. 经因果掩码（下三角）的多层 Transformer 编码器，只允许看到历史
    4. 取最后时刻的隐状态，经全连接层直接回归未来 5 日 (pred_window, 6) 序列
    """
    def __init__(self, input_dim=6, d_model=128, n_heads=8, n_layers=3,
                 ff_dim=512, output_dim=6, pred_len=5, max_len=512, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.output_dim = output_dim
        self.d_model = d_model

        # 输入线性嵌入
        self.input_proj = nn.Linear(input_dim, d_model)
        # 可学习位置编码
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        # 因果掩码（下三角，禁止看到未来）
        self.register_buffer('causal_mask', torch.triu(
            torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1))
        # 标准 Transformer 编码器层（Pre-Norm 结构，训练更稳定）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        # 回归头：预测 pred_len * output_dim
        self.head = nn.Linear(d_model, pred_len * output_dim)
        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x: (B, lookback, input_dim)
        B, L, _ = x.shape
        h = self.input_proj(x) * math.sqrt(self.d_model)      # (B, L, d_model)
        pos = torch.arange(L, device=x.device).unsqueeze(0)  # (1, L)
        h = self.drop(h + self.pos_embedding(pos))
        # 因果掩码 + padding 掩码（全 1，序列定长）
        mask = self.causal_mask[:L, :L]
        pad_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)
        h = self.encoder(h, mask=mask, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        out = self.head(h[:, -1, :])                          # 取最后时刻 (B, d_model)
        return out.reshape(-1, self.pred_len, self.output_dim)


# ==================== 模型配置写盘 ====================
def write_model_config(best_loss=None):
    """
    写模型结构配置到 save_dir/config.json
    训练开始时即写一次（中途中断 test.py 也能正常加载），训练结束时补写 best_loss
    """
    info = {
        'model': 'VanillaTransformer',
        'lookback': cfg.lookback,
        'pred_window': cfg.pred_window,
        'input_dim': cfg.input_dim,
        'output_dim': cfg.output_dim,
        'd_model': cfg.d_model,
        'n_heads': cfg.n_heads,
        'n_layers': cfg.n_layers,
        'ff_dim': cfg.ff_dim,
        'dropout': cfg.dropout,
        'seed': cfg.seed,
        'batch_size': cfg.batch_size,
        'epochs': cfg.epochs,
        'lr': cfg.lr,
        'num_workers': cfg.num_workers,
        'use_tf32': cfg.use_tf32,
        'train_start': os.environ.get('TRAIN_START_DATE', ''),
        'train_end': os.environ.get('TRAIN_END_DATE', ''),
    }
    if best_loss is not None:
        info['best_loss'] = best_loss
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# ==================== 训练 ====================
def train(data_file: str, logger=None, ts: str = None) -> str:
    """训练 Vanilla Transformer 模型，保存最佳权重与配置到 save_dir"""
    log = logger.info if logger else print
    metrics = MetricsWriter('transformer', ts) if ts else None
    # 构建模型
    model = VanillaTransformer(
        input_dim=cfg.input_dim,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        output_dim=cfg.output_dim,
        pred_len=cfg.pred_window,
        max_len=max(cfg.lookback, 512),
        dropout=cfg.dropout,
    ).to(cfg.device)

    # 构建 DataLoader（固定种子保证可复现）
    ds = KLineDataset(data_file)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn,
                    num_workers=cfg.num_workers, worker_init_fn=seed_worker,
                    generator=make_generator(), pin_memory=True, drop_last=True)
    log(f"训练样本数: {len(ds)}, 设备: {cfg.device}")
    # 优化器与余弦退火学习率调度
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.epochs * len(dl))
    loss_fn = nn.MSELoss()

    os.makedirs(cfg.save_dir, exist_ok=True)
    # 训练开始即写 config.json：中途中断也能被 test.py 正常加载
    write_model_config()
    best, patience = float('inf'), 0

    for ep in range(cfg.epochs):
        model.train()
        t0 = time.time()
        tl = 0
        pbar = tqdm(dl, desc=f"Transformer E{ep + 1}/{cfg.epochs}")
        for x, y in pbar:
            x, y = x.to(cfg.device), y.to(cfg.device)
            if torch.isnan(x).any() or torch.isnan(y).any():
                continue
            pred = model(x)
            loss = loss_fn(pred, y)
            if torch.isnan(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            sch.step()
            tl += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = tl / max(len(dl), 1)
        dt = time.time() - t0
        log(f"[Transformer] Epoch {ep + 1}/{cfg.epochs} 平均损失: {avg:.6f} "
            f"(lr={sch.get_last_lr()[0]:.2e}, 耗时 {dt:.1f}s)")
        if metrics:
            metrics.write(ep + 1, avg, sch.get_last_lr()[0], dt)

        # 保存最佳模型（训练集上使用全部数据，不划分验证集，与 Kronos 口径一致）
        if avg < best:
            best = avg
            patience = 0
            torch.save(model.state_dict(), os.path.join(cfg.save_dir, 'best_model.pt'))
            log(f"保存最佳模型 (loss={best:.6f})")
        else:
            patience += 1
            if patience >= cfg.patience:
                log(f"早停：连续 {cfg.patience} 轮训练损失不下降，"
                    f"实际训练 {ep + 1} 轮")
                break

    # 训练结束补写 config.json（记录最佳损失）
    write_model_config(best_loss=best)
    return cfg.save_dir


# ==================== 主流程 ====================
def main():
    """主函数：生成训练数据 -> 训练 Vanilla Transformer"""
    ts = timestamp()
    logger = setup_logger('transformer', ts)
    logger.info("Vanilla Transformer 训练流程开始")

    # 第一步：生成训练样本（含缓存，与 LSTM 共用同一份数据）
    data_file = generate_samples(
        lookback=cfg.lookback,
        pred_window=cfg.pred_window,
        step=cfg.step,
    )

    # 配置快照：超参数 + 时间划分 + 代码/数据版本（MD5），保证结果可溯源
    snapshot_config = {k: v for k, v in vars(Config).items() if not k.startswith('_')}
    snapshot_config['train_start'] = os.environ.get('TRAIN_START_DATE', '')
    snapshot_config['train_end'] = os.environ.get('TRAIN_END_DATE', '')
    cfg_file = save_config_snapshot(
        'transformer', ts,
        config=snapshot_config,
        code_files=[os.path.abspath(__file__)],
        data_files=[STOCK_CSV, HS300_CSV, CALENDAR_JSON, data_file],
    )
    logger.info(f"配置快照已保存: {cfg_file}")

    # 第二步：训练
    train(data_file, logger, ts)
    logger.info("Vanilla Transformer 训练流程结束")


if __name__ == "__main__":
    main()
