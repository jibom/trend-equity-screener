"""HSI 抄底+逃顶 合并 backtest (一次 compute_one 过, 出两个 CSV)。

universe: sector_cluster.load_pool() 的 305 只港股池。
复用 backtest_doji_signal.compute_one + sector_cluster.forward_adjust_group + backtest_top_signal.compute_active_top。
输出:
  output/doji_signal_daily.csv  (抄底)
  output/top_signal_daily.csv   (逃顶)
用法: python scripts/backtest_hsi_combined.py [--start 2014-01-01] [--end 2026-07-07]
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import pymysql, pandas as pd, numpy as np
from db_config import DB_CONFIG
import sector_cluster as sc
from backtest_doji_signal import compute_one
from backtest_top_signal import compute_active_top

IDX_CODE = 'HSI.HI'
STOCK_TABLE = 'hkshareeodprices'
IDX_TABLE = 'hkindexeodprices'


def compute_active_bottom(sig: pd.DataFrame) -> pd.DataFrame:
    """抄底卖盘衰竭状态机: 触发完整信号入池, 当天J<20留池, J>=20连3天出池。"""
    sig = sig.sort_values(['code', 'date']).reset_index(drop=True)
    exhausted = np.zeros(len(sig), dtype=bool)
    for _, idx in sig.groupby('code').groups.items():
        idx = np.array(idx)
        j = sig.iloc[idx]['daily_j'].values
        triggered = sig.iloc[idx]['signal'].values
        in_pool = False
        days_above = 0
        for k in range(len(idx)):
            if triggered[k]:
                in_pool = True
                days_above = 0
            elif in_pool:
                if not np.isnan(j[k]) and j[k] < 20:
                    days_above = 0
                else:
                    days_above += 1
                    if days_above >= 3:
                        in_pool = False
            exhausted[idx[k]] = in_pool
    sig['active'] = exhausted
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2014-01-01')
    ap.add_argument('--end', default='2026-07-07')
    args = ap.parse_args()

    pool = sc.load_pool()
    codes = list(pool['Ticker'])
    print(f"=== HSI Combined Backtest ({args.start} ~ {args.end}) ===\n[Pool] {len(codes)} HK stocks")

    conn = pymysql.connect(**DB_CONFIG)
    end = args.end.replace('-', '')
    start = (pd.to_datetime(args.start) - pd.Timedelta(days=400)).strftime('%Y%m%d')
    BATCH = 50
    raw_parts = []
    print(f"[DB] 分批拉取 EOD ({start}~{end}) ...")
    for bi in range(0, len(codes), BATCH):
        batch = codes[bi:bi + BATCH]
        codes_sql = ','.join(f"'{c}'" for c in batch)
        df_b = pd.read_sql(
            f"SELECT S_INFO_WINDCODE AS code, TRADE_DT, S_DQ_CLOSE, S_DQ_ADJOPEN, S_DQ_ADJHIGH, "
            f"S_DQ_ADJLOW, S_DQ_ADJCLOSE, S_DQ_VOLUME, S_DQ_AMOUNT FROM {STOCK_TABLE} "
            f"WHERE TRADE_DT BETWEEN '{start}' AND '{end}' AND S_INFO_WINDCODE IN ({codes_sql}) "
            f"ORDER BY S_INFO_WINDCODE, TRADE_DT", conn)
        raw_parts.append(df_b)
    raw = pd.concat(raw_parts, ignore_index=True)
    print(f"[DB] 拉取 {IDX_CODE} ...")
    idx = pd.read_sql(f"SELECT TRADE_DT, S_DQ_CLOSE FROM {IDX_TABLE} WHERE S_INFO_WINDCODE='{IDX_CODE}' AND TRADE_DT BETWEEN '{start}' AND '{end}' ORDER BY TRADE_DT", conn)
    conn.close()
    idx['date'] = idx['TRADE_DT']
    idx['hsi_close'] = idx['S_DQ_CLOSE'].astype(float)
    idx = idx[['date', 'hsi_close']]
    print(f"[Fetch] {raw['code'].nunique()} stocks, {len(idx)} index days")

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

    bt_start = args.start.replace('-', '')
    bt_end = args.end.replace('-', '')
    sig = sig[(sig['date'] >= bt_start) & (sig['date'] <= bt_end)].copy()

    sig['signal'] = (
        sig['is_doji_5d'] &
        (sig['daily_j_low5'] < 10) &
        (sig['weekly_j_low4'] < 10) &
        (sig['has_div'] | sig['has_surge_5d'])
    )
    sig = compute_active_bottom(sig)
    sig['top_signal'] = (
        sig['is_uptrend'] &
        sig['is_star_5d'] &
        (sig['daily_j_high5'] > 90) &
        (sig['weekly_j_high4'] > 90) &
        (sig['has_bear_div'] | sig['has_surge_5d'])
    )
    sig = compute_active_top(sig)

    # 抄底聚合
    daily_b = sig.groupby('date').agg(
        signal_count=('signal', 'sum'), active_count=('active', 'sum'),
        total_count=('code', 'nunique'), below_ma50_count=('below_ma50', 'sum'),
        vol_surge_count=('is_vol_surge', 'sum'), big_drop_count=('is_big_drop', 'sum'),
        panic_count=('is_panic', 'sum'), union_count=('is_union', 'sum'),
    ).reset_index()
    daily_b['pct'] = daily_b['signal_count'] / daily_b['total_count'] * 100
    daily_b['active_pct'] = daily_b['active_count'] / daily_b['total_count'] * 100
    daily_b['breadth_below_ma50'] = daily_b['below_ma50_count'] / daily_b['total_count'] * 100
    daily_b['vol_surge_pct'] = daily_b['vol_surge_count'] / daily_b['total_count'] * 100
    daily_b['big_drop_pct'] = daily_b['big_drop_count'] / daily_b['total_count'] * 100
    daily_b['panic_pct'] = daily_b['panic_count'] / daily_b['total_count'] * 100
    daily_b['union_pct'] = daily_b['union_count'] / daily_b['total_count'] * 100
    daily_b = daily_b.merge(idx, on='date', how='left')
    out_b = os.path.join(os.path.dirname(__file__), '..', 'output', 'doji_signal_daily.csv')
    daily_b.to_csv(out_b, index=False, encoding='utf-8-sig')
    print(f"\n[Bottom] {len(daily_b)} days -> {out_b}  信号日(>0): {(daily_b['signal_count']>0).sum()}  最大活跃占比: {daily_b['active_pct'].max():.2f}%")

    # 逃顶聚合
    daily_t = sig.groupby('date').agg(
        top_signal_count=('top_signal', 'sum'), active_top_count=('active_top', 'sum'),
        total_count=('code', 'nunique'), below_ma50_count=('below_ma50', 'sum'),
        vol_surge_count=('is_vol_surge', 'sum'), big_up_count=('is_big_up', 'sum'),
        union_up_count=('is_union_up', 'sum'), sky_vol_count=('is_sky_vol', 'sum'),
        vol_price_div_count=('is_vol_price_div', 'sum'), dist_top_count=('is_dist_top', 'sum'),
        shrink_new_high_count=('is_shrink_new_high', 'sum'),
    ).reset_index()
    daily_t['top_pct'] = daily_t['top_signal_count'] / daily_t['total_count'] * 100
    daily_t['active_top_pct'] = daily_t['active_top_count'] / daily_t['total_count'] * 100
    daily_t['breadth_below_ma50'] = daily_t['below_ma50_count'] / daily_t['total_count'] * 100
    daily_t['vol_surge_pct'] = daily_t['vol_surge_count'] / daily_t['total_count'] * 100
    daily_t['big_up_pct'] = daily_t['big_up_count'] / daily_t['total_count'] * 100
    daily_t['union_up_pct'] = daily_t['union_up_count'] / daily_t['total_count'] * 100
    daily_t['sky_vol_pct'] = daily_t['sky_vol_count'] / daily_t['total_count'] * 100
    daily_t['vol_price_div_pct'] = daily_t['vol_price_div_count'] / daily_t['total_count'] * 100
    daily_t['dist_top_pct'] = daily_t['dist_top_count'] / daily_t['total_count'] * 100
    daily_t['shrink_new_high_pct'] = daily_t['shrink_new_high_count'] / daily_t['total_count'] * 100
    daily_t = daily_t.merge(idx, on='date', how='left')
    out_t = os.path.join(os.path.dirname(__file__), '..', 'output', 'top_signal_daily.csv')
    daily_t.to_csv(out_t, index=False, encoding='utf-8-sig')
    print(f"[Top] {len(daily_t)} days -> {out_t}  顶部信号日(>0): {(daily_t['top_signal_count']>0).sum()}  最大活跃占比: {daily_t['active_top_pct'].max():.2f}%")


if __name__ == '__main__':
    main()
