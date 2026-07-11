"""回测(逃顶): 镜像 backtest_doji_signal 的抄底框架, 统计每日港股池满足"顶部反转+超买/倍量"信号的个股占比, 与HSI对比.

镜像逻辑: 对每个交易日, 统计同时满足以下条件的股票数 (顶部反转):
  1) 近5日有射击之星/长上影 (is_star_5d)
  2) 日线KDJ近5日最高J>90
  3) 周线KDJ近4周最高J>90
  4) 顶背离(近10日) OR 射击之星日倍量(≥2×30均量)
排除仅无量无背离的票.

输出: output/top_signal_daily.csv (date, top_signal_count, total_count, top_pct, active_top_pct,
      breadth_below_ma50, vol_surge_pct, big_up_pct, union_up_pct, hsi_close)
用法: python scripts/backtest_top_signal.py [--start 2014-01-01] [--end 2026-07-03]
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import pymysql, pandas as pd, numpy as np
from db_config import DB_CONFIG
import sector_cluster as sc
from backtest_doji_signal import compute_one


def compute_active_top(sig: pd.DataFrame) -> pd.DataFrame:
    """顶部卖盘衰竭状态机 (镜像 compute_active): 触发完整顶部信号入池, 当天J>80留池,
    J<=80 连续3天出池 (KDJ回落, 顶部动能消退)。"""
    sig = sig.sort_values(['code', 'date']).reset_index(drop=True)
    exhausted = np.zeros(len(sig), dtype=bool)
    for _, idx in sig.groupby('code').groups.items():
        idx = np.array(idx)
        j = sig.iloc[idx]['daily_j'].values
        triggered = sig.iloc[idx]['top_signal'].values
        in_pool = False
        days_below = 0
        for k in range(len(idx)):
            if triggered[k]:
                in_pool = True
                days_below = 0
            elif in_pool:
                if not np.isnan(j[k]) and j[k] > 80:
                    days_below = 0
                else:
                    days_below += 1
                    if days_below >= 3:
                        in_pool = False
            exhausted[idx[k]] = in_pool
    sig['active_top'] = exhausted
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2014-01-01')
    ap.add_argument('--end', default='2026-07-03')
    args = ap.parse_args()

    pool = sc.load_pool()
    print(f"=== Backtest TOP Signal ({args.start} ~ {args.end}) ===\n[Pool] {len(pool)} HK stocks")

    end = args.end.replace('-', '')
    start = (pd.to_datetime(args.start) - pd.Timedelta(days=400)).strftime('%Y%m%d')
    codes = list(pool['Ticker'])
    from hk_data import fetch_hk_stocks, fetch_hk_index
    print(f"[DB] 拉取 {len(codes)} 只港股 EOD ({start}~{end}) [jianxin→EODHD→yfinance] ...")
    raw = fetch_hk_stocks(codes, start, end).rename(columns={'S_INFO_WINDCODE': 'code'})

    # HSI
    print("[DB] 拉取 HSI.HI ...")
    idx_raw = fetch_hk_index('HSI.HI', start, end)
    hsi = pd.DataFrame({'date': idx_raw['TRADE_DT'], 'hsi_close': idx_raw['S_DQ_CLOSE'].astype(float)})
    hsi = hsi.sort_values('date').reset_index(drop=True)
    print(f"[Fetch] {raw['code'].nunique()} stocks, {len(hsi)} HSI days")

    # 预计算每只股票
    all_signals = []
    for i, (code, g) in enumerate(raw.groupby('code')):
        gg = sc.forward_adjust_group(g)
        if gg is None:
            continue
        df = compute_one(gg)
        if df is None:
            continue
        df['code'] = code
        all_signals.append(df)
        if (i + 1) % 50 == 0:
            print(f"  computed {i+1}/{raw['code'].nunique()}")

    sig = pd.concat(all_signals, ignore_index=True)
    print(f"[Compute] {len(sig)} stock-days")

    # 筛回测区间
    bt_start = args.start.replace('-', '')
    bt_end = args.end.replace('-', '')
    sig = sig[(sig['date'] >= bt_start) & (sig['date'] <= bt_end)].copy()

    # 顶部完整信号 (镜像抄底信号; 高位gate: 仅上升趋势中的射击之星才计)
    sig['top_signal'] = (
        sig['is_uptrend'] &
        sig['is_star_5d'] &
        (sig['daily_j_high5'] > 90) &
        (sig['weekly_j_high4'] > 90) &
        (sig['has_bear_div'] | sig['has_surge_5d'])
    )

    # 顶部 active 状态机
    sig = compute_active_top(sig)

    daily = sig.groupby('date').agg(
        top_signal_count=('top_signal', 'sum'),
        active_top_count=('active_top', 'sum'),
        total_count=('code', 'nunique'),
        below_ma50_count=('below_ma50', 'sum'),
        vol_surge_count=('is_vol_surge', 'sum'),
        big_up_count=('is_big_up', 'sum'),
        union_up_count=('is_union_up', 'sum'),
        sky_vol_count=('is_sky_vol', 'sum'),
        vol_price_div_count=('is_vol_price_div', 'sum'),
        dist_top_count=('is_dist_top', 'sum'),
        shrink_new_high_count=('is_shrink_new_high', 'sum'),
    ).reset_index()
    daily['top_pct'] = daily['top_signal_count'] / daily['total_count'] * 100
    daily['active_top_pct'] = daily['active_top_count'] / daily['total_count'] * 100
    daily['breadth_below_ma50'] = daily['below_ma50_count'] / daily['total_count'] * 100
    daily['vol_surge_pct'] = daily['vol_surge_count'] / daily['total_count'] * 100
    daily['big_up_pct'] = daily['big_up_count'] / daily['total_count'] * 100
    daily['union_up_pct'] = daily['union_up_count'] / daily['total_count'] * 100
    daily['sky_vol_pct'] = daily['sky_vol_count'] / daily['total_count'] * 100
    daily['vol_price_div_pct'] = daily['vol_price_div_count'] / daily['total_count'] * 100
    daily['dist_top_pct'] = daily['dist_top_count'] / daily['total_count'] * 100
    daily['shrink_new_high_pct'] = daily['shrink_new_high_count'] / daily['total_count'] * 100
    daily = daily.merge(hsi, on='date', how='left')

    out = os.path.join(os.path.dirname(__file__), '..', 'output', 'top_signal_daily.csv')
    daily.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n[Done] {len(daily)} trading days → {out}")
    print(f"  顶部信号日(>0): {(daily['top_signal_count'] > 0).sum()} / {len(daily)}")
    print(f"  平均占比: {daily['top_pct'].mean():.2f}%  活跃占比: {daily['active_top_pct'].mean():.2f}%")
    print(f"  最大活跃占比: {daily['active_top_pct'].max():.2f}% (on {daily.loc[daily['active_top_pct'].idxmax(), 'date']})")


if __name__ == '__main__':
    main()
