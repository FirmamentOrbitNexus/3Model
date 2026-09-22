#!/usr/bin/env python3
"""
合并所有年份的数据文件为最终的 stock_data.csv
- 读取 data/stock_data_*.csv
- 合并后按【股票代码+日期】去重（保留最新）
- 按 股票代码+日期 排序
- ✅合并成功后自动删除各个年份源文件
"""
import pandas as pd
import glob
import os
import argparse

COLUMNS = ['股票代码', '日期', '开盘', '收盘', '最高', '最低',
           '成交量', '成交额', '振幅', '涨跌额', '换手率', '涨跌幅']


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(script_dir, "..", "data")
    default_output = os.path.join(default_data_dir, "stock_data.csv")

    parser = argparse.ArgumentParser(description="合并各年份数据文件，合并成功后删除年份分片文件")
    parser.add_argument('--data-dir', default=default_data_dir, help="数据目录")
    parser.add_argument('--output', default=default_output, help="最终输出文件")
    parser.add_argument('--no-delete', action="store_true", help="设置该参数则不删除原始年份文件")
    args = parser.parse_args()

    pattern = os.path.join(args.data_dir, "stock_data_*.csv")
    files = sorted(glob.glob(pattern))
    # 排除输出文件本身，不要把合并结果加入待合并列表
    files = [f for f in files if os.path.abspath(f) != os.path.abspath(args.output)]

    if not files:
        print(f"未找到任何年份文件: {pattern}")
        exit(1)

    print(f"找到 {len(files)} 个年份文件:")
    for f in files:
        print(f"  - {f}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig', dtype={'股票代码': str})
            print(f"  读取 {os.path.basename(f)}: {len(df)} 条")
            dfs.append(df)
        except Exception as e:
            print(f"  警告: 读取 {f} 失败: {e}")

    if not dfs:
        print("没有可合并的数据，退出，不删除任何文件")
        exit(1)

    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)

    merged = merged.drop_duplicates(subset=['股票代码', '日期'], keep='last')
    merged = merged[COLUMNS]

    merged['日期_dt'] = pd.to_datetime(merged['日期'], errors='coerce')
    merged = merged.dropna(subset=['日期_dt'])

    merged = merged.sort_values(['股票代码', '日期_dt']).reset_index(drop=True)
    merged = merged.drop(columns=['日期_dt'])

    # 写入合并后文件
    merged.to_csv(args.output, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 60)
    print(f"合并完成: {args.output}")
    print(f"  - 合并前总行数: {before}")
    print(f"  - 去除重复: {before - len(merged)} 条")
    print(f"  - 最终行数: {len(merged)}")
    print(f"  - 股票数量: {merged['股票代码'].nunique()}")
    print(f"  - 时间范围: {merged['日期'].min()} 至 {merged['日期'].max()}")
    print(f"  - 文件大小: {os.path.getsize(args.output) / 1024 / 1024:.2f} MB")
    dup = merged.duplicated(subset=['股票代码', '日期']).sum()
    print(f"  - 数据去重: {'✓ 无重复' if dup == 0 else f'警告，{dup} 条重复'}")

    # ------------------------------
    # 合并成功，删除年份源文件
    # ------------------------------
    if not args.no_delete:
        print("\n开始删除原始年份分片文件...")
        deleted_count = 0
        for f in files:
            try:
                os.remove(f)
                print(f"  ✓ 删除: {os.path.basename(f)}")
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ 删除失败 {os.path.basename(f)} : {e}")
        print(f"原始文件处理完毕，成功删除 {deleted_count}/{len(files)} 个年份文件")
    else:
        print("\n--no‑delete 已开启，保留原始年份文件，不执行删除")


if __name__ == "__main__":
    main()
