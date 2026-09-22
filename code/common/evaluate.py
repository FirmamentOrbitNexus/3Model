# ============================================================
# evaluate.py - 多模型统一评估脚本（预测精度 + 组合回测）
#
# 评估对象：output/{model}/ 下由各 test.py 产出的累积文件
#   - result_full.csv      全量预测（pred_date, code, expected_roi_pred, ...）
#   - result_portfolio.csv 组合选择（pred_date, stock_id, weight）
#
# 预测精度指标（按 pred_date 截面计算，再跨日期聚合）：
#   IC          预测 ROI 与实际 ROI 的皮尔逊相关
#   RankIC      斯皮尔曼相关（对异常值更稳健，论文主指标）
#   ICIR        RankIC 均值 / 标准差
#   t-stat      RankIC 显著性 t 统计量 = ICIR * sqrt(期数)
#   DirAcc      涨跌方向准确率
#   RMSE / MAE  ROI 误差
#
# 组合回测指标（实际 ROI 加权，每期 5 个交易日）：
#   期均收益 / 年化收益 / 夏普比率 / 最大回撤 / 胜率 / 超额(等权基准)
#
# 用法：
#   py code/common/evaluate.py                  # 自动评估 output/ 下所有有结果的模型
#   py code/common/evaluate.py lstm transformer # 只评估指定模型
# 输出：
#   output/evaluation_detail.csv   按模型×日期的明细指标
#   output/evaluation_summary.csv  按模型聚合的汇总表（论文用）
# ============================================================

import os
import sys
import glob

import numpy as np
import pandas as pd

# ---------- 路径 ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OUT_DIR = os.environ.get('OUT_DIR', os.path.join(BASE_DIR, 'output'))
DATA_FILE = os.environ.get('DATA_FILE', os.path.join(BASE_DIR, 'data', 'stock_data.csv'))
HS300_FILE = os.environ.get('HS300_FILE', os.path.join(BASE_DIR, 'data', 'hs300_stock_list.csv'))

MODELS = ['kronos', 'lstm', 'transformer', 'lgbm']
# 每期持有 5 个交易日，一年约 252 个交易日
HORIZON = 5
PERIODS_PER_YEAR = 252 / HORIZON
# 截面最少股票数，低于该值当期指标不可信
MIN_STOCKS = 30


# ==================== 实际收益计算 ====================
def load_actual_open() -> pd.DataFrame:
    """读取原始数据 -> 过滤沪深300 -> 透视为 (交易日 × 股票) 的开盘价宽表"""
    df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
    df = df.rename(columns={
        '股票代码': 'code', '日期': 'timestamps', '开盘': 'open', '收盘': 'close',
        '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount',
    })
    df['timestamps'] = pd.to_datetime(df['timestamps'])
    df['code'] = df['code'].astype(str).str.zfill(6)

    hs300 = pd.read_csv(HS300_FILE, encoding='utf-8-sig')
    hs300['code'] = hs300['code'].astype(str).str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    df = df[df['code'].isin(set(hs300['code'].unique()))]

    open_wide = df.pivot_table(index='timestamps', columns='code', values='open', aggfunc='last')
    return open_wide.sort_index()


def actual_future_roi(open_wide: pd.DataFrame, pred_date: str, horizon: int = HORIZON):
    """
    实际未来 5 日收益率：pred_date 后第 1 个与第 horizon 个交易日的开盘价
    与各模型 T+1_open / T+5_open 的定义完全一致
    返回按 code 索引的 Series
    """
    pred_date = pd.to_datetime(pred_date)
    dates = open_wide.index[open_wide.index > pred_date]
    if len(dates) < horizon:
        return None
    o1 = open_wide.loc[dates[0]]
    o5 = open_wide.loc[dates[horizon - 1]]
    roi = (o5 - o1) / o1
    return roi.replace([np.inf, -np.inf], np.nan).dropna()


# ==================== 预测精度指标 ====================
def date_metrics(pred_df: pd.DataFrame, actual_roi: pd.Series) -> dict:
    """单个 pred_date 的截面预测指标"""
    m = pred_df.merge(actual_roi.rename('actual_roi'),
                      left_on='code', right_index=True, how='inner')
    m = m.dropna(subset=['expected_roi', 'actual_roi'])
    if len(m) < MIN_STOCKS:
        return None
    p, a = m['expected_roi'], m['actual_roi']
    return {
        'n_stocks': len(m),
        'ic': p.corr(a),
        'rank_ic': p.rank().corr(a.rank()),
        'dir_acc': float((np.sign(p) == np.sign(a)).mean()),
        'rmse': float(np.sqrt(((p - a) ** 2).mean())),
        'mae': float((p - a).abs().mean()),
    }


def aggregate_prediction_metrics(detail: pd.DataFrame) -> dict:
    """跨 pred_date 聚合：均值 / 标准差 / ICIR / t 统计量"""
    n = len(detail)
    out = {'n_dates': n}
    for col in ['ic', 'rank_ic', 'dir_acc', 'rmse', 'mae']:
        out[f'{col}_mean'] = detail[col].mean()
        out[f'{col}_std'] = detail[col].std()
    ic_std = detail['rank_ic'].std()
    out['icir'] = detail['rank_ic'].mean() / ic_std if ic_std and ic_std > 0 else np.nan
    out['t_stat'] = out['icir'] * np.sqrt(n) if n > 1 and not np.isnan(out['icir']) else np.nan
    return out


# ==================== 组合回测 ====================
def backtest_portfolio(port_df: pd.DataFrame, actual_lookup: dict) -> dict:
    """
    组合回测：每期组合实际收益 = Σ w_i × actual_roi_i
    基准 = 当期全部候选股票的等权平均收益（超额 = 组合 - 基准）
    """
    port_df = port_df.copy()
    port_df['weight'] = port_df['weight'].astype(float)
    port_df['stock_id'] = port_df['stock_id'].astype(str).str.zfill(6)

    rets, bench, dates = [], [], []
    for date, grp in port_df.groupby('pred_date'):
        roi = actual_lookup.get(date)
        if roi is None:
            continue
        r = grp.set_index('stock_id')['weight']
        valid = r.index.intersection(roi.index)
        if len(valid) == 0:
            continue
        rets.append((r[valid] * roi[valid]).sum())
        bench.append(roi[roi.index.isin(grp['stock_id'])].pipe(lambda s: s.mean() if len(s) else roi.mean()))
        dates.append(date)

    if not rets:
        return {}
    r = pd.Series(rets, index=pd.to_datetime(dates)).sort_index()
    b = pd.Series(bench, index=r.index)
    excess = r - b

    cum = (1 + r).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    std = r.std()
    return {
        'n_periods': len(r),
        'period_ret_mean': float(r.mean()),
        'ann_ret': float(r.mean() * PERIODS_PER_YEAR),
        'sharpe': float(r.mean() / std * np.sqrt(PERIODS_PER_YEAR)) if std and std > 0 else np.nan,
        'max_drawdown': max_dd,
        'win_rate': float((r > 0).mean()),
        'excess_ann': float(excess.mean() * PERIODS_PER_YEAR),
    }


# ==================== 单模型评估 ====================
def evaluate_model(model: str, open_wide: pd.DataFrame, actual_cache: dict,
                   detail_rows: list, port_rows: list) -> dict:
    """评估单个模型：预测精度 + 组合回测，结果写入 detail/port 行列表"""
    model_dir = os.path.join(OUT_DIR, model)
    full_file = os.path.join(model_dir, 'result_full.csv')
    port_file = os.path.join(model_dir, 'result_portfolio.csv')
    if not os.path.exists(full_file):
        print(f"[{model}] 未找到 {full_file}，跳过（请先运行对应 test.py）")
        return None

    pred = pd.read_csv(full_file, dtype=str)
    pred['expected_roi'] = pred['expected_roi'].astype(float)
    pred['rank'] = pred['rank'].astype(float)
    print(f"[{model}] 全量预测 {len(pred)} 行，覆盖 {pred['pred_date'].nunique()} 个预测日")

    # ---- 预测精度 ----
    for date, grp in pred.groupby('pred_date'):
        if date not in actual_cache:
            actual_cache[date] = actual_future_roi(open_wide, date)
        actual_roi = actual_cache[date]
        if actual_roi is None:
            print(f"[{model}] {date} 缺少足够的实际未来数据，跳过")
            continue
        met = date_metrics(grp[['code', 'expected_roi']], actual_roi)
        if met is None:
            continue
        detail_rows.append({'model': model, 'pred_date': date, **met})

    # ---- 组合回测 ----
    if os.path.exists(port_file):
        port_df = pd.read_csv(port_file, dtype=str)
        summary = backtest_portfolio(port_df, actual_cache)
    else:
        summary = {}

    # ---- 聚合 ----
    rows = [r for r in detail_rows if r['model'] == model]
    if not rows:
        print(f"[{model}] 无有效评估期")
        return None
    agg = aggregate_prediction_metrics(pd.DataFrame(rows))
    agg.update({f'port_{k}': v for k, v in summary.items()})
    agg['model'] = model
    return agg


# ==================== 主流程 ====================
def main():
    models = sys.argv[1:] or [m for m in MODELS
                              if os.path.exists(os.path.join(OUT_DIR, m, 'result_full.csv'))]
    if not models:
        print("output/ 下未找到任何模型的 result_full.csv，请先运行各模型的 test.py")
        return
    print(f"评估模型: {models}\n")

    open_wide = load_actual_open()
    actual_cache = {}
    detail_rows, summaries = [], []

    for model in models:
        agg = evaluate_model(model, open_wide, actual_cache, detail_rows, port_rows)
        if agg:
            summaries.append(agg)

    if not summaries:
        print("无任何可评估结果")
        return

    # ---- 保存明细与汇总 ----
    detail_df = pd.DataFrame(detail_rows).sort_values(['model', 'pred_date'])
    detail_df.to_csv(os.path.join(OUT_DIR, 'evaluation_detail.csv'), index=False)

    col_order = ['model', 'n_dates', 'rank_ic_mean', 'rank_ic_std', 'icir', 't_stat',
                 'ic_mean', 'dir_acc_mean', 'rmse_mean', 'mae_mean', 'n_stocks_mean',
                 'n_periods', 'port_period_ret_mean', 'port_ann_ret', 'port_sharpe',
                 'port_max_drawdown', 'port_win_rate', 'port_excess_ann']
    summary_df = pd.DataFrame(summaries)
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]
    summary_df.to_csv(os.path.join(OUT_DIR, 'evaluation_summary.csv'), index=False)

    # ---- 打印汇总表 ----
    print("\n========== 预测精度（跨预测日聚合） ==========")
    cols_pred = ['model', 'n_dates', 'rank_ic_mean', 'icir', 't_stat', 'dir_acc_mean', 'rmse_mean']
    print(summary_df[cols_pred].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print("\n========== 组合回测（实际 ROI 加权，每期 5 交易日） ==========")
    cols_port = ['model', 'n_periods', 'port_ann_ret', 'port_sharpe', 'port_max_drawdown',
                 'port_win_rate', 'port_excess_ann']
    port_cols = [c for c in cols_port if c in summary_df.columns]
    if port_cols:
        print(summary_df[port_cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f"\n明细: {os.path.join(OUT_DIR, 'evaluation_detail.csv')}")
    print(f"汇总: {os.path.join(OUT_DIR, 'evaluation_summary.csv')}")


if __name__ == "__main__":
    main()
