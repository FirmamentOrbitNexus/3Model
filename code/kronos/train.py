# ============================================================
# train.py - Kronos v6 微调训练【精简可复现版】
# 流程: 生成训练数据 -> 微调 Tokenizer -> 微调 Predictor
# ============================================================
# ========== 所有环境变量必须在 import torch 之前设置 ==========

import os

# 确定性相关：cuBLAS 矩阵乘法、Python 哈希种子
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '42')
# 离线模式：禁止 Hugging Face / Transformers 联网
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

# 导入系统相关模块
import sys
# 导入时间模块，用于统计每轮训练耗时
import time
# 导入 pickle 用于序列化数据
import pickle
# 导入 json 用于读取配置文件
import json
# 导入 random 用于设置随机种子
import random
# 导入 numpy 用于数值计算
import numpy as np
# 导入 pandas 用于数据处理
import pandas as pd
# 导入 torch 深度学习框架
import torch

# 导入神经网络功能模块
import torch.nn.functional as F
# 导入数据集和数据加载器
from torch.utils.data import Dataset, DataLoader
# 导入 AdamW 优化器
from torch.optim import AdamW
# 导入余弦退火学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR
# 导入 SDPA 内核选择器，用于指定注意力计算的后端
from torch.nn.attention import sdpa_kernel, SDPBackend
# 导入进度条显示工具
from tqdm import tqdm
# 导入警告模块
import warnings
# 忽略所有警告信息
warnings.filterwarnings('ignore')

# 将当前文件所在目录加入系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 导入自定义的 Kronos 模型、Tokenizer 和时间戳计算函数
from featurework import Kronos, KronosTokenizer, calc_time_stamps
# 将项目 code/ 目录加入系统路径，导入共用模块（统一配置 / 路径 / 日志）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import SEED, configure_determinism, get_device
from common.paths import DATA_DIR, STOCK_CSV, HS300_CSV, CALENDAR_JSON, model_dir
from common.logger_utils import timestamp, setup_logger, save_config_snapshot, MetricsWriter

# ========== 统一确定性配置（种子 / cuDNN / TF32）==========
configure_determinism()


# ==================== 数据生成 ====================
def generate_finetune_data():
    """生成微调训练数据，如果已存在则直接返回路径"""
    # 训练数据时间划分（与 LSTM/Transformer 口径一致，默认不限制即保持原有行为）
    train_start = os.environ.get('TRAIN_START_DATE', '')
    train_end = os.environ.get('TRAIN_END_DATE', '')
    split_tag = (f"_s{train_start.replace('-', '')}" if train_start else '') + \
                (f"_e{train_end.replace('-', '')}" if train_end else '')
    # 输出文件的保存路径（统一走 paths，Docker 内解析为 /app/data）
    output_file = os.path.join(DATA_DIR, f"finetune_samples_v6{split_tag}.pkl")
    # 如果文件已存在，直接返回路径，跳过生成
    if os.path.exists(output_file):
        return output_file

    # 原始股票数据文件路径（统一走 paths）
    DATA_FILE = STOCK_CSV
    # 历史窗口大小（输入给模型的 K 线数量）
    LOOKBACK = 512
    # 预测窗口大小（需要预测的未来 K 线数量）
    PRED_WINDOW = 5
    # 滑动窗口的步长
    STEP = 10
    # 每只股票至少需要的数据量
    MIN_REQUIRED = LOOKBACK + PRED_WINDOW + 10
    # 需要使用的列名
    MODEL_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'amount']

    # 读取原始数据
    df = pd.read_csv(DATA_FILE)
    # 重命名列为统一格式
    df = df.rename(columns={
        '股票代码': 'code', '日期': 'timestamps',
        '开盘': 'open', '收盘': 'close', '最高': 'high',
        '最低': 'low', '成交量': 'volume', '成交额': 'amount'
    })
    # 转换日期列格式
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    # 填充缺失值
    df['volume'] = df['volume'].fillna(0)
    df['amount'] = df['amount'].fillna(0)

    # 读取沪深300成分股列表，过滤非成分股（统一走 paths）
    hs300 = pd.read_csv(HS300_CSV, encoding='utf-8-sig')
    # 统一股票代码格式（6位数字字符串）
    hs300['code'] = hs300['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    valid_codes = set(hs300['code'].unique())
    df['code'] = df['code'].astype(str).str.zfill(6)
    # 只保留沪深300成分股的数据
    df = df[df['code'].isin(valid_codes)]

    # 时间划分：仅当显式设置 TRAIN_START_DATE / TRAIN_END_DATE 时生效（论文实验防数据泄露）
    if train_start:
        df = df[df['timestamps'] >= pd.to_datetime(train_start)]
    if train_end:
        df = df[df['timestamps'] <= pd.to_datetime(train_end)]
    if df.empty:
        raise ValueError(f"时间划分后无训练数据: [{train_start or '最早'}, {train_end or '最新'}]")
    print(f"训练数据区间: {df['timestamps'].min().date()} ~ {df['timestamps'].max().date()}")

    # 获取所有股票代码并排序
    stock_codes = sorted(df['code'].unique())

    # 加载交易日历（统一走 paths）
    calendar_file = CALENDAR_JSON
    if os.path.exists(calendar_file):
        # 如果本地存在缓存，直接读取
        with open(calendar_file) as f:
            trade_dates = pd.to_datetime(json.load(f))

    # 生成训练样本
    samples = []
    valid_stocks = 0
    for code in tqdm(stock_codes, desc="生成样本"):
        # 取单只股票数据
        stock = df[df['code'] == code].copy().set_index('timestamps')
        stock = stock[~stock.index.duplicated(keep='last')]
        # 对齐交易日历，补全停牌日
        sc = trade_dates[(trade_dates >= stock.index.min()) & (trade_dates <= stock.index.max())]
        if len(sc.difference(stock.index)) > 0:
            stock = stock.reindex(sc)
            for col in ['open','high','low','close']:
                if col in stock.columns: stock[col] = stock[col].ffill()
            for col in ['volume','amount']:
                if col in stock.columns: stock[col] = stock[col].fillna(0)
            stock = stock.bfill()
        stock = stock.reset_index().rename(columns={'index': 'timestamps'})
        # 数据量不足则跳过
        if len(stock) < MIN_REQUIRED:
            continue
        valid_stocks += 1
        # 滑动窗口切分样本
        for start_idx in range(0, len(stock) - LOOKBACK - PRED_WINDOW + 1, STEP):
            x_df = stock[MODEL_COLUMNS].iloc[start_idx:start_idx+LOOKBACK].reset_index(drop=True)
            x_ts = stock['timestamps'].iloc[start_idx:start_idx+LOOKBACK].reset_index(drop=True)
            y_ts = stock['timestamps'].iloc[start_idx+LOOKBACK:start_idx+LOOKBACK+PRED_WINDOW].reset_index(drop=True)
            # 跳过包含空值的样本
            if x_df.isnull().values.any():
                continue
            samples.append(dict(code=code, x_df=x_df, x_timestamp=x_ts, y_timestamp=y_ts))

    # 保存样本到本地 pkl 文件
    with open(output_file, 'wb') as f:
        pickle.dump(samples, f)
    return output_file


# ==================== 配置 ====================
class Config:
    """训练配置类，统一管理所有超参数"""
    # 预训练权重路径（默认 Docker 内相对路径；本地可用环境变量指向任意位置）
    tokenizer_pretrained = os.environ.get('KRONOS_TOKENIZER_PATH', "./pretrained/Kronos-Tokenizer-base")
    # 预训练 Predictor 路径
    model_pretrained = os.environ.get('KRONOS_PREDICTOR_PATH', "./pretrained/Kronos-small")
    # 批次大小
    batch_size = 12
    # Tokenizer 梯度裁剪阈值
    max_grad_norm = 1.0
    # Predictor 梯度裁剪阈值
    max_grad_norm_pred = 0.5
    # 权重衰减（L2正则化系数）
    weight_decay = 0.01
    # Tokenizer 学习率
    lr_tokenizer = 5e-6
    # Tokenizer 训练轮数
    epochs_tokenizer = 12
    # Predictor 学习率
    lr_predictor = 5e-6
    # Predictor 训练轮数
    epochs_predictor = 25
    # 消融开关：跳过 Tokenizer 微调（环境变量 KRONOS_SKIP_TOKENIZER_TRAIN=1 生效）
    # 三组实验对比: 完整微调(默认) / 只调 Predictor(此开关) / zero-shot(test.py 指向 pretrained/)
    skip_tokenizer_train = os.environ.get('KRONOS_SKIP_TOKENIZER_TRAIN', '0') == '1'
    # 随机种子
    seed = SEED
    # 模型保存目录（Kronos 微调权重；统一走 paths，环境变量 MODEL_DIR 可覆盖）
    save_dir = model_dir('kronos')
    # 训练设备（GPU 或 CPU）
    device = get_device()
    # DataLoader 工作进程数（默认 0：Kronos 样本为 DataFrame 结构，
    # 多进程在 Windows 下会重复反序列化整个数据集，显存/内存代价高）
    num_workers = int(os.environ.get('KRONOS_NUM_WORKERS', '0'))
    # 训练数据截止日（环境变量 TRAIN_END_DATE 可覆盖）
    # 与 test.py 的首个预测基准日保持一致，避免测试期数据泄露进训练集
    train_end_date = '2026-08-14'


cfg = Config()


# ==================== 数据集 ====================
class FinetuneDataset(Dataset):
    """微调数据集类"""
    def __init__(self, pkl_path):
        """从 pkl 文件加载训练样本"""
        with open(pkl_path, 'rb') as f:
            self.samples = pickle.load(f)

    def __len__(self):
        """返回样本总数"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取单个样本，包含标准化后的 K 线数据和时间戳特征"""
        s = self.samples[idx]
        # 提取特征值
        xv = s['x_df'].values.astype(np.float32)
        # 计算均值和标准差进行标准化
        xm, xs = np.mean(xv, axis=0), np.std(xv, axis=0)
        xs = np.where(xs < 1e-4, 1.0, xs)
        xn = np.clip(np.nan_to_num((xv - xm) / xs, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0)
        # 计算时间戳特征
        xt = calc_time_stamps(pd.to_datetime(s['x_timestamp']))
        yt = calc_time_stamps(pd.to_datetime(s['y_timestamp']))
        return {
            'x': torch.from_numpy(xn),
            'x_stamp': torch.from_numpy(np.nan_to_num(xt.values.astype(np.float32), nan=0.0)),
            'y_stamp': torch.from_numpy(np.nan_to_num(yt.values.astype(np.float32), nan=0.0)),
        }


def collate_fn(b):
    """自定义批处理函数，将多个样本堆叠成一个批次"""
    return torch.stack([i['x'] for i in b]), torch.stack([i['x_stamp'] for i in b]), torch.stack([i['y_stamp'] for i in b])


def seed_worker(worker_id):
    """DataLoader 工作进程的随机种子设置函数，确保多进程数据加载可复现"""
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


# 创建全局随机数生成器，用于 DataLoader 的 shuffle
g = torch.Generator()
g.manual_seed(SEED)


# ==================== Tokenizer 训练 ====================
def train_tokenizer(data_file, logger=None, ts=None):
    """微调 Tokenizer：将 K 线数据编码为离散 token"""
    log = logger.info if logger else print
    metrics = MetricsWriter('kronos_tokenizer', ts) if ts else None
    # 加载预训练 Tokenizer
    tokenizer = KronosTokenizer.from_pretrained(cfg.tokenizer_pretrained).to(cfg.device)
    tokenizer.train()

    # 构建 DataLoader
    ds = FinetuneDataset(data_file)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn,
                    num_workers=cfg.num_workers, worker_init_fn=seed_worker, generator=g,
                    pin_memory=True, drop_last=True)
    # 优化器和学习率调度器
    opt = AdamW(tokenizer.parameters(), lr=cfg.lr_tokenizer, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.epochs_tokenizer * len(dl))
    best, sp = float('inf'), os.path.join(cfg.save_dir, "tokenizer")
    os.makedirs(sp, exist_ok=True)

    # 使用 MATH 后端进行注意力计算，确保确定性
    with sdpa_kernel(SDPBackend.MATH):
        for ep in range(cfg.epochs_tokenizer):
            t0 = time.time()
            tl = 0
            pbar = tqdm(dl, desc=f"Tokenizer E{ep+1}/{cfg.epochs_tokenizer}")
            opt.zero_grad()
            for _, (x, _, _) in enumerate(pbar):
                x = x.to(cfg.device)
                if torch.isnan(x).any(): continue
                # 前向传播：编码 -> 量化 -> 解码
                (_, z), bsq, _, _ = tokenizer(x)
                # 重建损失
                rl = F.mse_loss(z, x)
                if torch.isnan(rl) or torch.isnan(bsq): continue
                # 总损失 = 重建损失 + 量化损失
                (rl + bsq).backward()
                torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), cfg.max_grad_norm)
                opt.step(); opt.zero_grad(); sch.step()
                tl += rl.item() + bsq.item()
            avg = tl / max(len(dl), 1)
            dt = time.time() - t0
            log(f"[Tokenizer] Epoch {ep+1}/{cfg.epochs_tokenizer} 平均损失: {avg:.6f} "
                f"(lr={sch.get_last_lr()[0]:.2e}, 耗时 {dt:.1f}s)")
            if metrics:
                metrics.write(ep + 1, avg, sch.get_last_lr()[0], dt)
            # 保存最佳模型
            if avg < best:
                best = avg
                tokenizer.save_pretrained(sp)
                log(f"保存最佳 Tokenizer (loss={best:.6f})")
    return sp


# ==================== Predictor 训练 ====================
def train_predictor(tokenizer_path, data_file, logger=None, ts=None):
    """微调 Predictor：基于 token 预测未来价格"""
    log = logger.info if logger else print
    metrics = MetricsWriter('kronos_predictor', ts) if ts else None
    # 加载微调后的 Tokenizer（冻结参数）
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_path).to(cfg.device).eval()
    for p in tokenizer.parameters(): p.requires_grad = False
    # 加载预训练 Predictor
    model = Kronos.from_pretrained(cfg.model_pretrained).to(cfg.device).train()

    # 构建 DataLoader
    ds = FinetuneDataset(data_file)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn,
                    num_workers=cfg.num_workers, worker_init_fn=seed_worker, generator=g,
                    pin_memory=True, drop_last=True)
    # 优化器和学习率调度器
    opt = AdamW(model.parameters(), lr=cfg.lr_predictor, weight_decay=cfg.weight_decay)
    sch = CosineAnnealingLR(opt, T_max=cfg.epochs_predictor * len(dl))
    best, sp, patience = float('inf'), os.path.join(cfg.save_dir, "predictor"), 0
    os.makedirs(sp, exist_ok=True)

    # 使用 MATH 后端进行注意力计算，确保确定性
    with sdpa_kernel(SDPBackend.MATH):
        for ep in range(cfg.epochs_predictor):
            t0 = time.time()
            tl = 0
            pbar = tqdm(dl, desc=f"Predictor E{ep+1}/{cfg.epochs_predictor}")
            opt.zero_grad()
            for _, (x, xs, _) in enumerate(pbar):
                x, xs = x.to(cfg.device), xs.to(cfg.device)
                if torch.isnan(x).any(): continue
                with torch.no_grad():
                    try: s1, s2 = tokenizer.encode(x, half=True)
                    except: continue
                # 自回归预测：用前 N-1 个 token 预测第 N 个
                is1, is2 = s1[:, :-1].contiguous(), s2[:, :-1].contiguous()
                ts1, ts2 = s1[:, 1:].contiguous(), s2[:, 1:].contiguous()
                # 模型前向传播
                l1, l2 = model(is1, is2, stamp=xs[:, :-1, :].contiguous())
                # 计算交叉熵损失
                loss, _, _ = model.head.compute_loss(l1, l2, ts1, ts2)
                if torch.isnan(loss): continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm_pred)
                opt.step(); opt.zero_grad(); sch.step()
                tl += loss.item()
            avg = tl / max(len(dl), 1)
            dt = time.time() - t0
            log(f"[Predictor] Epoch {ep+1}/{cfg.epochs_predictor} 平均损失: {avg:.6f} "
                f"(lr={sch.get_last_lr()[0]:.2e}, 耗时 {dt:.1f}s)")
            if metrics:
                metrics.write(ep + 1, avg, sch.get_last_lr()[0], dt)
            if avg < best:
                best = avg
                patience = 0
                model.save_pretrained(sp)
                log(f"保存最佳 Predictor (loss={best:.6f})")
            else:
                patience += 1
                # 早停：连续 8 轮不下降则停止训练
                if patience >= 8:
                    log(f"早停：连续 {patience} 轮训练损失不下降，"
                        f"实际训练 {ep+1} 轮")
                    break
    return sp


#==================== 主流程 ====================
def main():
    """主函数：生成数据 -> 微调 Tokenizer -> 微调 Predictor"""
    ts = timestamp()
    logger = setup_logger('kronos', ts)
    logger.info("Kronos 微调训练流程开始")

    # 训练数据截止日（写入环境变量，供 generate_finetune_data 做时间划分，避免泄露测试期）
    os.environ.setdefault('TRAIN_END_DATE', cfg.train_end_date)
    if os.environ.get('TRAIN_END_DATE'):
        logger.info(f"训练数据截止日: {os.environ['TRAIN_END_DATE']}")

    # 第一步：生成训练数据
    data_file = generate_finetune_data()
    logger.info(f"训练数据就绪: {data_file}")

    # 配置快照：超参数 + 代码/数据版本（MD5），保证结果可溯源
    this_dir = os.path.dirname(os.path.abspath(__file__))
    snapshot_config = {k: v for k, v in vars(Config).items() if not k.startswith('_')}
    snapshot_config['train_start'] = os.environ.get('TRAIN_START_DATE', '')
    snapshot_config['train_end'] = os.environ.get('TRAIN_END_DATE', '')
    cfg_file = save_config_snapshot(
        'kronos', ts,
        config=snapshot_config,
        code_files=[os.path.join(this_dir, 'train.py'), os.path.join(this_dir, 'featurework.py')],
        data_files=[data_file, STOCK_CSV, HS300_CSV, CALENDAR_JSON],
    )
    logger.info(f"配置快照已保存: {cfg_file}")

    # 第二步：微调 Tokenizer（消融模式下跳过，直接用预训练 Tokenizer）
    if cfg.skip_tokenizer_train:
        logger.info("消融模式: 跳过 Tokenizer 微调，"
                    "Predictor 将基于预训练 Tokenizer 的 token 训练 (KRONOS_SKIP_TOKENIZER_TRAIN=1)")
        tp = cfg.tokenizer_pretrained
    else:
        tp = train_tokenizer(data_file, logger, ts)
        logger.info("Tokenizer 微调完成")
    # 第三步：微调 Predictor
    pp = train_predictor(tp, data_file, logger, ts)
    logger.info("Predictor 微调完成，Kronos 训练流程结束")
    return tp, pp


if __name__ == "__main__":
    main()