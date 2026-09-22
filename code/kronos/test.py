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
from common.paths import STOCK_CSV, HS300_CSV, model_dir, output_dir

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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'result.csv')
# 全量预测 / 组合累积文件（滚动评估用，按 pred_date 累积去重）
FULL_FILE = os.path.join(OUTPUT_DIR, 'result_full.csv')
PORTFOLIO_FILE = os.path.join(OUTPUT_DIR, 'result_portfolio.csv')

# 预测目标日期（可由环境变量 PRED_DATES 覆盖）
PRED_DATES = os.environ.get('PRED_DATES', '2026-08-03,2026-08-04,2026-08-05,2026-08-06,2026-08-07')
PRED_DATES = PRED_DATES.split(',')
# 数据截止日期（可由环境变量 END_DATE 覆盖）
END_DATE = os.environ.get('END_DATE', '2026-07-31')
# 本次预测基准日期（评估分组键）
PRED_DATE = END_DATE

'''
选股逻辑：
1. 按expected_roi降序排序
2. 超高收益池 roi>0.10，取距离池子均值最近1只，权重0.4
3. 高收益池 0.05<roi<0.10，过滤|roi‑池均值|≤0.01，取距离过滤后均值最近1只，权重0.3
4. 负收益池过滤：计算V=(super_roi*0.4+high_roi*0.3)/5；筛选V+roi_neg>0，取sum_val最靠近0的标的，权重0.3
总权重 0.4+0.3+0.3 =1.0
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

def save_full_predictions(result_df):
    """保存全部股票的预测结果，按 pred_date 累积去重，供 evaluate.py 统一评估"""
    full_df = result_df[['code', 'T+1_open', 'T+5_open', 'expected_roi', 'rank']].copy()
    full_df['code'] = full_df['code'].astype(str).str.zfill(6)
    full_df.insert(0, 'pred_date', PRED_DATE)
    if os.path.exists(FULL_FILE):
        old = pd.read_csv(FULL_FILE, dtype=str)
        old = old[old['pred_date'] != PRED_DATE]
        full_df = pd.concat([old, full_df], ignore_index=True)
    full_df = full_df.sort_values(['pred_date', 'rank']).reset_index(drop=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    full_df.to_csv(FULL_FILE, index=False)
    print(f"全量预测已保存: {FULL_FILE} (共 {len(full_df)} 行)")


def save_portfolio(select_rows):
    """保存组合选择结果（含 pred_date，累积），供 evaluate.py 组合回测"""
    if not select_rows:
        return
    port_df = pd.DataFrame(select_rows)
    port_df['stock_id'] = port_df['stock_id'].astype(str).str.zfill(6)
    port_df.insert(0, 'pred_date', PRED_DATE)
    if os.path.exists(PORTFOLIO_FILE):
        old = pd.read_csv(PORTFOLIO_FILE, dtype=str)
        old = old[old['pred_date'] != PRED_DATE]
        port_df = pd.concat([old, port_df], ignore_index=True)
    port_df = port_df.sort_values(['pred_date', 'weight'], ascending=[True, False]).reset_index(drop=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    port_df.to_csv(PORTFOLIO_FILE, index=False)
    print(f"组合记录已保存: {PORTFOLIO_FILE}")


def main():
    """主函数：数据加载 -> 模型预测 -> 选股 -> 输出结果"""
    # 加载并清洗数据
    df_all = preprocess_data()
    # 过滤到截止日期之前的数据
    df_all = df_all[df_all['timestamps'] <= pd.to_datetime(END_DATE)]
    # 获取所有股票代码并排序
    stock_codes = sorted(df_all['code'].unique())

    # 加载模型
    load_model()
    results = []
    # 遍历每只股票进行预测
    for code in tqdm(stock_codes, desc="预测"):
        try:
            stock = df_all[df_all['code']==code].copy()
            # 数据不足则跳过
            if len(stock) < LOOKBACK:
                continue
            lb = min(LOOKBACK, len(stock))
            # 取最近 LOOKBACK 天的数据作为输入
            x_df = stock[['open','high','low','close','volume','amount']].iloc[-lb:].reset_index(drop=True)
            # 缩尾处理
            x_df = winsorize_x_df(x_df)
            x_ts = stock['timestamps'].iloc[-lb:].reset_index(drop=True)
            y_ts = pd.to_datetime(PRED_DATES).to_series()

            # 模型预测
            p = _predictor.predict(
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
            results.append({'code':code,'T+1_open':pred_mean[0,0],'T+5_open':pred_mean[4,0]})
        except Exception:
            continue

    # 构建结果 DataFrame
    result_df = pd.DataFrame(results)
    # 计算预期收益率
    result_df['expected_roi'] = (result_df['T+5_open'] - result_df['T+1_open']) / result_df['T+1_open']
    # 按收益率排序并赋予排名
    result_df["rank"] = result_df["expected_roi"].rank(method="first", ascending=False).astype(int)

    # 保存全量预测（累积，供统一评估）
    save_full_predictions(result_df)

    # ====================== 选股开始 ======================
    # 根据预期收益率划分三个池子
    # 超高收益池：预期收益率大于 10%
    pool_super_high  = result_df[result_df['expected_roi'] > 0.10].copy()
    # 高收益池：预期收益率在 5% 到 10% 之间
    pool_high        = result_df[(result_df['expected_roi'] >0.05) & (result_df['expected_roi'] <0.10)].copy()
    # 负收益池：预期收益率在 -2% 到 0 之间（小幅亏损）
    pool_neg_small   = result_df[(result_df['expected_roi'] > -0.02) & (result_df['expected_roi'] <0)].copy()

    # 初始化最终选中的三只股票
    final_super = None
    final_high = None
    final_neg = None

    # 1. 超高收益池选股：距离池子原始均值最近1只，权重0.4
    if len(pool_super_high) > 0:
        # 计算池子的原始均值
        mean_super_raw = pool_super_high['expected_roi'].mean()
        # 计算每只股票收益率与均值的绝对距离
        pool_super_high['abs_to_mean'] = np.abs(pool_super_high['expected_roi'] - mean_super_raw)
        # 取距离均值最近的股票
        final_super = pool_super_high.loc[pool_super_high['abs_to_mean'].idxmin()]

    # 2. 高收益池选股：过滤后取距离过滤后均值最近1只，权重0.3
    if len(pool_high) >0:
        # 计算高收益池的原始均值
        mean_high_raw = pool_high['expected_roi'].mean()
        # 过滤：保留收益率与均值差距在 1% 以内的股票
        pool_high_filtered = pool_high[np.abs(pool_high['expected_roi'] - mean_high_raw) <= 0.01].copy()
        if len(pool_high_filtered) >0:
            # 计算过滤后的均值
            mean_high_filtered = pool_high_filtered['expected_roi'].mean()
            # 取距离过滤后均值最近的股票
            pool_high_filtered['abs_to_mean'] = np.abs(pool_high_filtered['expected_roi'] - mean_high_filtered)
            final_high = pool_high_filtered.loc[pool_high_filtered['abs_to_mean'].idxmin()]

    # 3. 负收益池选股：满足约束条件后取 sum_val 最接近 0 的股票，权重0.3
    if final_super is not None and final_high is not None and len(pool_neg_small)>0:
        # 获取超高收益和高收益池选出的股票的收益率
        roi_super = final_super['expected_roi']
        roi_high = final_high['expected_roi']
        # 计算组合基准值 V
        V = (roi_super * 0.4 + roi_high * 0.3) / 5.0

        pool_neg_new = pool_neg_small.copy()
        # 过滤：收益率的绝对值在 0.01 到 0.018 之间
        pool_neg_new = pool_neg_new[(np.abs(pool_neg_new['expected_roi']) >=0.01) & (np.abs(pool_neg_new['expected_roi']) <=0.018)].copy()
        # 计算 V + roi_neg
        pool_neg_new['sum_val'] = V + pool_neg_new['expected_roi']
        # 筛选 sum_val > 0 的候选股票
        candidate = pool_neg_new[pool_neg_new['sum_val'] > 0].copy()

        if len(candidate) >0:
            # 按 sum_val 升序排列，取最接近 0 的股票
            candidate = candidate.sort_values("sum_val", ascending=True).reset_index(drop=True)
            final_neg = candidate.iloc[0]

    # 组装输出结果
    select_rows = []
    if final_super is not None:
        select_rows.append({"stock_id":final_super["code"], "weight":0.4})
    if final_high is not None:
        select_rows.append({"stock_id":final_high["code"], "weight":0.3})
    if final_neg is not None:
        select_rows.append({"stock_id":final_neg["code"], "weight":0.3})

    # 兜底：当某池无满足条件样本，提示警告
    if len(select_rows)!=3:
        print("⚠️警告：部分池子没有满足筛选条件的股票，输出条目不足3条")

    out_df = pd.DataFrame(select_rows)
    # ====================== 选股逻辑结束 ======================

    # 保存组合记录（累积，供统一评估）并输出竞赛格式结果
    save_portfolio(select_rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(out_df.to_string(index=False))

if __name__ == "__main__":
    main()