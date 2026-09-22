#!/usr/bin/env python3
"""
主控脚本: 按年份循环获取沪深300成分股数据，最后合并为 stock_data.csv
- 每年一个独立文件: data/stock_data_YYYY.csv
- 每年运行日志: logs/run_YYYY.log（屏幕和文件同时输出）
- 全部年份完成后自动合并
- 中断后重跑即可断点续传（脚本内部有增量补齐逻辑）

用法:
    python run_by_year.py              # 获取 2018~2026 全部年份
    python run_by_year.py 2020 2023    # 只获取指定年份区间
    python run_by_year.py 2020 2020    # 单独重跑某一年
"""

import sys
import os
import glob
import io
import logging
from datetime import datetime

import pandas as pd

# ---------- 让 utils 目录可以被 import ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.get_stock_data import run as fetch_one_year   # noqa: E402

# ---------- Windows 下防止控制台/日志 GBK 编码报错 ----------
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8',
                                  errors='replace')

# ---------- 配置 ----------
START_YEAR_DEFAULT = 2018
END_YEAR_DEFAULT = 2026
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
FINAL_OUTPUT = os.path.join(DATA_DIR, "stock_data.csv")

COLUMNS = ['股票代码', '日期', '开盘', '收盘', '最高', '最低',
           '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅']


def year_range_from_args():
    """解析命令行参数得到起止年份"""
    if len(sys.argv) >= 3:
        return int(sys.argv[1]), int(sys.argv[2])
    return START_YEAR_DEFAULT, END_YEAR_DEFAULT


def setup_year_logger(year):
    """为某一年创建同时输出到屏幕和文件的logger"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"run_{year}.log")

    logger = logging.getLogger(f"year_{year}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter('%(message)s')

    fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')  # 追加模式
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger, log_path


def merge_all_years():
    """合并所有年份文件为最终 stock_data.csv"""
    pattern = os.path.join(DATA_DIR, "stock_data_*.csv")
    files = sorted(glob.glob(pattern))

    # 排除最终输出文件本身（防止二次合并时把自己合进去）
    files = [f for f in files
             if os.path.abspath(f) != os.path.abspath(FINAL_OUTPUT)]

    if not files:
        print(f"未找到任何年份文件: {pattern}")
        return False

    print(f"\n找到 {len(files)} 个年份文件:")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig', dtype={'股票代码': str})
            print(f"  - {os.path.basename(f)}: {len(df)} 条")
            dfs.append(df)
        except Exception as e:
            print(f"  - 警告: 读取 {f} 失败: {e}")

    if not dfs:
        print("没有可合并的数据")
        return False

    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)

    # 去重: 同一股票同一天保留一条（文件按年份排序，keep='last' 即保留较新年份的数据）
    merged = merged.drop_duplicates(subset=['股票代码', '日期'], keep='last')
    merged = merged[COLUMNS]

    merged['日期_dt'] = pd.to_datetime(merged['日期'], format='%Y/%m/%d', errors='coerce')
    merged = merged.dropna(subset=['日期_dt'])
    merged = merged.sort_values(['股票代码', '日期_dt']).reset_index(drop=True)
    merged = merged.drop(columns=['日期_dt'])

    merged.to_csv(FINAL_OUTPUT, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"合并完成: {FINAL_OUTPUT}")
    print(f"  - 合并前总行数: {before}")
    print(f"  - 去除重复: {before - len(merged)} 条")
    print(f"  - 最终行数: {len(merged)}")
    print(f"  - 股票数量: {merged['股票代码'].nunique()}")
    print(f"  - 时间范围: {merged['日期'].min()} 至 {merged['日期'].max()}")
    print(f"  - 文件大小: {os.path.getsize(FINAL_OUTPUT) / 1024 / 1024:.2f} MB")

    dup = merged.duplicated(subset=['股票代码', '日期']).sum()
    print(f"  - 数据去重: {'✓ 无重复' if dup == 0 else f'警告，{dup} 条重复'}")
    return True


def main():
    start_year, end_year = year_range_from_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    today = datetime.now()
    today_str = today.strftime('%Y-%m-%d')
    current_year = today.year

    print("=" * 60)
    print(" 沪深300成分股历史数据获取 - 按年份执行")
    print(f" 年份范围: {start_year} ~ {end_year}")
    print(f" 数据目录: {DATA_DIR}")
    print(f" 当前日期: {today_str}")
    print("=" * 60)

    failed_years = []

    for year in range(start_year, end_year + 1):
        logger, log_path = setup_year_logger(year)

        logger.info("")
        logger.info("-" * 60)
        logger.info(f">>> 正在处理 {year} 年数据...")
        logger.info("-" * 60)

        year_start = f"{year}-01-01"
        # 未结束的年份用今天作为结束日期，已过去的年份用12-31
        year_end = today_str if year >= current_year else f"{year}-12-31"
        year_file = os.path.join(DATA_DIR, f"stock_data_{year}.csv")

        logger.info(f"时间范围: {year_start} ~ {year_end}")
        logger.info(f"输出文件: {year_file}")

        try:
            failed_count = fetch_one_year(year_start, year_end, year_file)
            if failed_count > 0:
                logger.warning(f">>> {year} 年有 {failed_count} 只股票获取失败，"
                               f"详见 {log_path}")
                failed_years.append(year)
            else:
                logger.info(f">>> {year} 年数据获取成功")
        except Exception as e:
            logger.error(f">>> {year} 年执行异常: {e}")
            failed_years.append(year)

    # ---------- 合并 ----------
    print("\n" + "=" * 60)
    print(">>> 开始合并所有年份数据...")
    print("=" * 60)

    if not merge_all_years():
        print(">>> 错误: 合并失败")
        exit(1)

    # ---------- 汇总 ----------
    print("\n" + "=" * 60)
    print(" 全部完成!")
    print(f" 最终文件: {FINAL_OUTPUT}")
    print("=" * 60)

    if failed_years:
        print("\n以下年份存在获取失败的股票（可单独重跑）:")
        for y in failed_years:
            print(f"  - {y} (重跑命令: python run_by_year.py {y} {y})")
        exit(2)

    exit(0)


if __name__ == "__main__":
    main()
