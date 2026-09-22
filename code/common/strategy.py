# ============================================================
# strategy.py - 选股策略与推理结果持久化（各模型 test.py 共用）
# 单点维护，避免选股逻辑在多个 test.py 中重复实现
# 纯 pandas/numpy 实现，不依赖 torch
# ============================================================

import os

import pandas as pd

# ==================== 选股参数（可调） ====================
# 组合持股数上限（赛题约束 ≤5），环境变量 TOP_K 可覆盖
TOP_K = int(os.environ.get('TOP_K', '5'))
# 入选所需的最低预期收益（默认 0.0 = 只买预期上涨的股票；
# 设为负值可放宽，如 -0.01 允许小幅负收益标的入选）
MIN_ROI = float(os.environ.get('MIN_ROI', '0.0'))

# 各模型输出目录下的通用文件名
FULL_FILE_NAME = 'result_full.csv'
PORTFOLIO_FILE_NAME = 'result_portfolio.csv'


# ==================== Top-K 等权选股 ====================
def select_portfolio(result_df: pd.DataFrame, top_k: int = None, min_roi: float = None):
    """
    Top-K 等权选股（四模型统一口径）：
      1) 过滤 expected_roi >= min_roi（默认 0.0，即只买预期上涨的股票，避免买入预测下跌标的）
      2) 按 expected_roi 降序取前 top_k 只（默认 5，满足"最多不超过 5 只"约束）
      3) 等权 1/n 分配（n = 实际入选数，权重和恒为 1.0，尽量满仓）
    候选不足时按实际数量输出，不补位到不存在的标的。

    result_df 需含列: code, expected_roi
    返回 [{stock_id, weight}, ...]
    """
    k = TOP_K if top_k is None else int(top_k)
    floor = MIN_ROI if min_roi is None else float(min_roi)

    if result_df is None or result_df.empty:
        return []
    if 'expected_roi' not in result_df.columns:
        raise KeyError("result_df 缺少 expected_roi 列")

    cand = result_df[result_df['expected_roi'] >= floor]
    if cand.empty:
        return []

    # 降序取前 K 只；code 作为次级排序键，保证并列时结果稳定可复现
    top = cand.sort_values(['expected_roi', 'code'],
                           ascending=[False, True]).head(max(k, 1))
    n = len(top)
    weight = 1.0 / n
    return [{'stock_id': row['code'], 'weight': weight} for _, row in top.iterrows()]


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
