# ============================================================
# strategy.py - 选股策略与推理结果持久化（各模型 test.py 共用）
# 单点维护，避免三池选股逻辑在多个 test.py 中重复实现
# 纯 pandas/numpy 实现，不依赖 torch
# ============================================================

import os

import numpy as np
import pandas as pd

# 各模型输出目录下的通用文件名
FULL_FILE_NAME = 'result_full.csv'
PORTFOLIO_FILE_NAME = 'result_portfolio.csv'


# ==================== 三池选股 ====================
def select_portfolio(result_df: pd.DataFrame):
    """
    三池选股策略（与 Kronos test.py 口径一致，最多 3 只股票）：
      1) 超高收益池 roi>0.10：取距离池均值最近的 1 只，权重 0.4
      2) 高收益池   0.05<roi<0.10：过滤 |roi-均值|≤0.01，取距过滤后均值最近 1 只，权重 0.3
      3) 负收益池   -0.02<roi<0：从 |roi|∈[0.01,0.018] 中选 V+roi>0 且最接近 0 的 1 只，权重 0.3
        其中 V = (超高池 roi × 0.4 + 高收益池 roi × 0.3) / 5
    总权重 0.4+0.3+0.3 = 1.0；某池无满足条件时输出条目不足 3 只（满足"最多不超过 5 只"约束）

    result_df 需含列: code, expected_roi
    返回 [{stock_id, weight}, ...]
    """
    pool_super = result_df[result_df['expected_roi'] > 0.10].copy()
    pool_high = result_df[(result_df['expected_roi'] > 0.05) & (result_df['expected_roi'] < 0.10)].copy()
    pool_neg = result_df[(result_df['expected_roi'] > -0.02) & (result_df['expected_roi'] < 0)].copy()

    final_super = final_high = final_neg = None

    # 1) 超高收益池：距离池均值最近的 1 只
    if len(pool_super) > 0:
        m = pool_super['expected_roi'].mean()
        pool_super['abs_to_mean'] = np.abs(pool_super['expected_roi'] - m)
        final_super = pool_super.loc[pool_super['abs_to_mean'].idxmin()]

    # 2) 高收益池：过滤后取距离过滤均值最近的 1 只
    if len(pool_high) > 0:
        m_raw = pool_high['expected_roi'].mean()
        filt = pool_high[np.abs(pool_high['expected_roi'] - m_raw) <= 0.01].copy()
        if len(filt) > 0:
            m = filt['expected_roi'].mean()
            filt['abs_to_mean'] = np.abs(filt['expected_roi'] - m)
            final_high = filt.loc[filt['abs_to_mean'].idxmin()]

    # 3) 负收益对冲池
    if final_super is not None and final_high is not None and len(pool_neg) > 0:
        V = (final_super['expected_roi'] * 0.4 + final_high['expected_roi'] * 0.3) / 5.0
        c = pool_neg.copy()
        c = c[(np.abs(c['expected_roi']) >= 0.01) & (np.abs(c['expected_roi']) <= 0.018)].copy()
        c['sum_val'] = V + c['expected_roi']
        c = c[c['sum_val'] > 0].sort_values('sum_val', ascending=True)
        if len(c) > 0:
            final_neg = c.iloc[0]

    rows = []
    if final_super is not None:
        rows.append({'stock_id': final_super['code'], 'weight': 0.4})
    if final_high is not None:
        rows.append({'stock_id': final_high['code'], 'weight': 0.3})
    if final_neg is not None:
        rows.append({'stock_id': final_neg['code'], 'weight': 0.3})
    return rows


# ==================== 推理结果持久化 ====================
def save_full_predictions(result_df: pd.DataFrame, output_dir: str, pred_date: str,
                          logger=None) -> str:
    """
    保存全部股票的预测结果到 output_dir/result_full.csv
    按 pred_date 累积去重（同 pred_date 覆盖），供 evaluate.py 统一评估
    自动保留 result_df 中可用的列（code + T+1_open/T+5_open/expected_roi/rank）
    """
    cols = ['code'] + [c for c in ('T+1_open', 'T+5_open', 'expected_roi', 'rank')
                       if c in result_df.columns]
    full_df = result_df[cols].copy()
    full_df['code'] = full_df['code'].astype(str).str.zfill(6)
    full_df.insert(0, 'pred_date', str(pred_date))

    full_file = os.path.join(output_dir, FULL_FILE_NAME)
    if os.path.exists(full_file):
        old = pd.read_csv(full_file, dtype=str)
        old = old[old['pred_date'] != str(pred_date)]
        full_df = pd.concat([old, full_df], ignore_index=True)

    # rank 转数值排序（旧数据读入为字符串，混合排序会失败）
    if 'rank' in full_df.columns:
        full_df['_rank'] = pd.to_numeric(full_df['rank'], errors='coerce')
        full_df = full_df.sort_values(['pred_date', '_rank']).drop(columns='_rank')
    full_df = full_df.reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    full_df.to_csv(full_file, index=False)
    if logger:
        logger.info(f"全量预测已保存: {full_file} (共 {len(full_df)} 行)")
    return full_file


def save_portfolio(select_rows, output_dir: str, pred_date: str, logger=None):
    """
    保存组合选择结果到 output_dir/result_portfolio.csv
    按 pred_date 累积去重，供 evaluate.py 组合回测
    """
    if not select_rows:
        return None
    port_df = pd.DataFrame(select_rows)
    port_df['stock_id'] = port_df['stock_id'].astype(str).str.zfill(6)
    port_df.insert(0, 'pred_date', str(pred_date))

    port_file = os.path.join(output_dir, PORTFOLIO_FILE_NAME)
    if os.path.exists(port_file):
        old = pd.read_csv(port_file, dtype=str)
        old = old[old['pred_date'] != str(pred_date)]
        port_df = pd.concat([old, port_df], ignore_index=True)

    port_df = port_df.sort_values(['pred_date', 'weight'], ascending=[True, False]).reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    port_df.to_csv(port_file, index=False)
    if logger:
        logger.info(f"组合记录已保存: {port_file}")
    return port_file
