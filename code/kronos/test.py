# ============================================================
# test.py - Kronos 模型推理 + 沪深300选股
# ============================================================
# ========== 所有环境变量必须在 import torch 之前设置 ==========

import sys, os

# 确定性相关：cuBLAS 矩阵乘法、Python 哈希种子
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONHASHSEED', '42')
# 离线模式：禁止 Hugging Face / Transformers 联网
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 将 code/kronos 与 code/ 加入 sys.path，确保 featurework 与 common 可被导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from featurework import Kronos, KronosTokenizer, KronosPredictor
from safetensors.torch import load_file
from common.config import configure_determinism, get_device
from common.data_io import load_trade_calendar
from common.paths import STOCK_CSV, HS300_CSV, model_dir, output_dir
from common.strategy import (
    TOP_K, select_portfolio, save_full_predictions, save_portfolio,
)

# ---------- 统一随机种子 & 确定性配置 ----------
configure_determinism()

# ---------- 路径与参数 ----------
# 历史窗口大小（输入给模型的 K 线天数，需与训练时一致）
LOOKBACK = 512
# 预测窗口大小（预测未来 K 线天数）
PRED_LEN = 5

# 数据文件路径（统一走 paths，环境变量 DATA_FILE / HS300_FILE 可覆盖）
DATA_FILE = STOCK_CSV
HS300_FILE = HS300_CSV
# 模型权重目录（默认微调权重；zero-shot 评测时用环境变量指向预训练权重，如：
#   TOKENIZER_DIR=/app/pretrained/Kronos-Tokenizer-base MODEL_DIR=/app/pretrained/Kronos-small）
TOKENIZER_DIR = os.environ.get('TOKENIZER_DIR', os.path.join(model_dir('kronos'), 'tokenizer'))
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(model_dir('kronos'), 'predictor'))
# 预测结果输出目录（位于 output/kronos/ 下）
OUTPUT_DIR = output_dir('kronos')

# 预测基准日列表（环境变量 END_DATES 逗号分隔可覆盖）
# 每个基准日预测其后 5 个交易日的开盘价，具体目标日期由交易日历推算（见 next_trade_dates）
#   2026-08-14（周五）-> 08-17 ~ 08-21
#   2026-08-21（周五）-> 08-24 ~ 08-28
#   2026-08-28（周五）-> 08-31 ~ 09-04
END_DATES = [d.strip() for d in
             os.environ.get('END_DATES', '2026-08-14,2026-08-21,2026-08-28').split(',')
             if d.strip()]

'''
选股逻辑（与其余三模型统一，实现在 common/strategy.py::select_portfolio）：
1. 过滤 expected_roi >= MIN_ROI（默认 0.0，只买预期上涨的股票）
2. 按 expected_roi 降序取前 TOP_K 只（默认 5，满足"最多不超过 5 只"约束）
3. 等权 1/n 分配（n = 实际入选数，权重和恒为 1.0，尽量满仓）
另：result_full.csv / result_portfolio.csv 由 common/strategy.py 统一维护
'''

# 训练设备（GPU 或 CPU）
_device = get_device()
# 全局预测器变量，懒加载
_predictor = None

def load_model():
    """加载模型：从本地文件加载 Tokenizer 和 Predictor 的权重"""
    global _predictor
    if _predictor is None:
        # 读取 Tokenizer 配置文件并构建模型
        with open(os.path.join(TOKENIZER_DIR,"config.json")) as f: tk_cfg = json.load(f)
        with open(os.path.join(MODEL_DIR,"config.json")) as f: md_cfg = json.load(f)
        tokenizer = KronosTokenizer(
            d_in=tk_cfg['d_in'],d_model=tk_cfg['d_model'],n_heads=tk_cfg['n_heads'],
            ff_dim=tk_cfg['ff_dim'],n_enc_layers=tk_cfg['n_enc_layers'],
            n_dec_layers=tk_cfg['n_dec_layers'],ffn_dropout_p=tk_cfg['ffn_dropout_p'],
            attn_dropout_p=tk_cfg['attn_dropout_p'],resid_dropout_p=tk_cfg['resid_dropout_p'],
            s1_bits=tk_cfg['s1_bits'],s2_bits=tk_cfg['s2_bits'],
            beta=tk_cfg['beta'],gamma0=tk_cfg['gamma0'],gamma=tk_cfg['gamma'],
            zeta=tk_cfg['zeta'],group_size=tk_cfg['group_size'],
        )
        # 加载 Tokenizer 权重
        tokenizer.load_state_dict(load_file(os.path.join(TOKENIZER_DIR,"model.safetensors")), strict=False)
        tokenizer = tokenizer.to(_device).eval()

        # 构建 Predictor 模型
        model = Kronos(
            s1_bits=md_cfg['s1_bits'],s2_bits=md_cfg['s2_bits'],n_layers=md_cfg['n_layers'],
            d_model=md_cfg['d_model'],n_heads=md_cfg['n_heads'],ff_dim=md_cfg['ff_dim'],
            ffn_dropout_p=md_cfg['ffn_dropout_p'],attn_dropout_p=md_cfg['attn_dropout_p'],
            resid_dropout_p=md_cfg['resid_dropout_p'],token_dropout_p=md_cfg['token_dropout_p'],
            learn_te=md_cfg['learn_te'],
        )
        # 加载 Predictor 权重
        model.load_state_dict(load_file(os.path.join(MODEL_DIR,"model.safetensors")), strict=False)
        model = model.to(_device).eval()

        # 创建预测器
        _predictor = KronosPredictor(model, tokenizer, device=_device, max_context=LOOKBACK)
    return _predictor

def preprocess_data():
    """数据预处理：加载并清洗股票数据，过滤沪深300成分股"""
    df = pd.read_csv(DATA_FILE)
    # 重命名列为统一格式
    df = df.rename(columns={'股票代码':'code','日期':'timestamps','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume','成交额':'amount'})
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df = df[['code','timestamps','open','high','low','close','volume','amount']]
    # 填充缺失值
    df['volume'] = df['volume'].fillna(0)
    df['amount'] = df['amount'].fillna(0)

    # 读取沪深300成分股列表，过滤非成分股
    hs300 = pd.read_csv(HS300_FILE)
    hs300['code'] = hs300['code'].str.replace('sh.','').str.replace('sz.','').str.zfill(6)
    df['code'] = df['code'].astype(str).str.zfill(6)
    df = df[df['code'].isin(set(hs300['code'].unique()))]
    return df

def winsorize_x_df(x_df):
    """输入数据缩尾处理：将超过3倍标准差的极端值截断，防止异常值干扰"""
    x_np = x_df.values.astype(np.float32)
    mean = np.mean(x_np,axis=0)
    std = np.std(x_np,axis=0)
    std = np.where(std<1e-6,1.0,std)
    x_np = np.clip(x_np, mean-3*std, mean+3*std)
    return pd.DataFrame(x_np, columns=x_df.columns)

def next_trade_dates(trade_dates, end_date: str, n: int):
    """返回基准日 end_date 之后的 n 个交易日（作为 Kronos 的 y_timestamp）"""
    after = trade_dates[trade_dates > pd.to_datetime(end_date)]
    return after[:n]


def predict_cross_section(predictor, df_all: pd.DataFrame, end_date: str, pred_dates):
    """
    在某个基准日对全市场做截面预测
    returns: (result_df, 候选股票数, 异常股票数)
    """
    df = df_all[df_all['timestamps'] <= pd.to_datetime(end_date)]
    stock_codes = sorted(df['code'].unique())
    y_ts = pd.to_datetime(list(pred_dates)).to_series()
    results, errors = [], 0
    for code in tqdm(stock_codes, desc=f"预测 {end_date}"):
        try:
            stock = df[df['code'] == code].copy()
            # 数据不足则跳过
            if len(stock) < LOOKBACK:
                continue
            lb = min(LOOKBACK, len(stock))
            # 取最近 LOOKBACK 天的数据作为输入
            x_df = stock[['open','high','low','close','volume','amount']].iloc[-lb:].reset_index(drop=True)
            # 缩尾处理
            x_df = winsorize_x_df(x_df)
            x_ts = stock['timestamps'].iloc[-lb:].reset_index(drop=True)

            # 模型预测
            p = predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=PRED_LEN,
                T = 0.8,        # 采样温度，控制预测的随机性
                top_p=0.9,      # 核采样参数
                sample_count=1, # 采样次数
                verbose=False   # 不显示进度条
            )
            pred_mean = p[['open','high','low','close','volume','amount']].values
            # 保存预测结果：T+1 和 T+5 的开盘价
            results.append({'code': code, 'T+1_open': pred_mean[0, 0], 'T+5_open': pred_mean[4, 0]})
        except Exception as e:
            errors += 1
            if errors <= 3:               # 只打印前 3 条，避免刷屏
                print(f"股票 {code} 预测失败: {e}")
            continue

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df['expected_roi'] = (result_df['T+5_open'] - result_df['T+1_open']) / result_df['T+1_open']
        result_df['rank'] = result_df['expected_roi'].rank(method='first', ascending=False).astype(int)
    return result_df, len(stock_codes), errors


def main():
    """主函数：数据加载 -> 逐期预测 -> 选股 -> 输出结果（三期滚动）"""
    print(f"Kronos 推理 + 沪深300 选股流程开始（{len(END_DATES)} 期）")
    df_all = preprocess_data()
    predictor = load_model()
    trade_dates = load_trade_calendar()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for end_date in END_DATES:
        pred_dates = next_trade_dates(trade_dates, end_date, PRED_LEN)
        if len(pred_dates) < PRED_LEN:
            print(f"[{end_date}] 警告：交易日历中其后不足 {PRED_LEN} 个交易日，跳过")
            continue

        result_df, n_codes, n_err = predict_cross_section(predictor, df_all, end_date, pred_dates)
        if result_df.empty:
            print(f"[{end_date}] 无有效预测结果（候选 {n_codes} 只，异常 {n_err} 只）")
            continue
        print(f"[{end_date}] 候选 {n_codes} 只，有效预测 {len(result_df)} 只，异常 {n_err} 只")

        # 保存全量预测（累积，供统一评估）
        save_full_predictions(result_df, OUTPUT_DIR, end_date)

        # 选股（Top-K 等权，与其余三模型统一口径）
        select_rows = select_portfolio(result_df)
        if len(select_rows) < TOP_K:
            print(f"[{end_date}] 候选不足，仅选出 {len(select_rows)} 只（上限 {TOP_K}）")

        out_df = pd.DataFrame(select_rows)
        # 保存组合记录（累积，供统一评估）并输出竞赛格式结果
        save_portfolio(select_rows, OUTPUT_DIR, end_date)
        period_file = os.path.join(OUTPUT_DIR, f"result_{end_date.replace('-', '')}.csv")
        out_df.to_csv(period_file, index=False)
        print(f"[{end_date}] 结果已保存: {period_file}")
        print(f"\n===== 基准日 {end_date}"
              f"（预测 {pred_dates[0].date()} ~ {pred_dates[-1].date()}）=====")
        print(out_df.to_string(index=False) if len(out_df) else "（无满足条件的股票）")

    print("Kronos 推理 + 选股流程结束")

if __name__ == "__main__":
    main()