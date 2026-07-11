"""回测: 港股 GICS sector 维度的底部择时 (HSI底部择时的 sector 版).

对每个 GICS 板块, 复用 backtest_doji_signal.compute_one 的单股信号预计算,
  1) 按 (sector, date) 聚合成板块层的占比口径 (active_pct / breadth / union_pct ...)
  2) 用恒生综合行业指数 (HSICS) 真实 OHLC 作为 sector_index, 在其上算 RSI/周KDJ/投降Z
  3) 6 项 expanding 阈值条件打分 (与 plot_doji_cumulative.py 同口径), score, 前瞻收益

输出: output/doji_signal_daily_by_sector.csv  (长表: date, sector, sector_index, 6条件, score, fwd_5/20/60, 各占比)
用法: python scripts/backtest_sector_doji.py [--start 2014-01-01] [--end 2026-07-03]
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

# GICS sector → 恒生综合行业指数 (Hang Seng Composite Industry Index) Wind 代码
# 11 个 HSICS 子指数与 GICS 11 sector 一一对应 (HSCICO 综合企业无 GICS 对应, 跳过)
SECTOR_INDEX_MAP = {
    'Energy': 'HSCIEN.HI',
    'Materials': 'HSCIMT.HI',
    'Industrials': 'HSCIIN.HI',
    'Consumer Discretionary': 'HSCICD.HI',
    'Consumer Staples': 'HSCICS.HI',
    'Health Care': 'HSCIH.HI',
    'Financials': 'HSCIFN.HI',
    'Information Technology': 'HSCIIT.HI',
    'Communication Services': 'HSCITC.HI',
    'Utilities': 'HSCIUT.HI',
    'Real Estate': 'HSCIPC.HI',
}


def compute_active(sig: pd.DataFrame) -> pd.DataFrame:
    """卖盘衰竭状态机 (与 backtest_doji_signal 同逻辑): 触发完整信号入池, 当天J<20留池,
    J>=20 连续3天出池。按 code 分组逐行跑。返回带 'active' 列的 sig。"""
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


def fetch_hsics(conn, start: str, end: str) -> pd.DataFrame:
    """拉 11 个恒生综合行业指数 OHLC, 返回 [Sector, date, sector_index(close), sector_high, sector_low]。
    sector_index = HSICS 收盘价 (真实指数, 非自建)。"""
    inv = {v: k for k, v in SECTOR_INDEX_MAP.items()}
    codes_sql = ','.join(f"'{c}'" for c in SECTOR_INDEX_MAP.values())
    raw = pd.read_sql(
        f"SELECT S_INFO_WINDCODE AS code, TRADE_DT AS date, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE "
        f"FROM hkindexeodprices WHERE TRADE_DT BETWEEN '{start}' AND '{end}' "
        f"AND S_INFO_WINDCODE IN ({codes_sql}) ORDER BY code, TRADE_DT", conn)
    raw['Sector'] = raw['code'].map(inv)
    raw = raw.rename(columns={'S_DQ_CLOSE': 'sector_index', 'S_DQ_HIGH': 'sector_high', 'S_DQ_LOW': 'sector_low'})
    for c in ['sector_index', 'sector_high', 'sector_low']:
        raw[c] = raw[c].astype(float)
    return raw[['Sector', 'date', 'sector_index', 'sector_high', 'sector_low']]


def score_sector(df: pd.DataFrame) -> pd.DataFrame:
    """单个 sector 的日级 df 上算 RSI/周KDJ/cap_z/expanding阈值/6条件/score/前瞻收益。
    与 plot_doji_cumulative.py 打分段同口径, HSI→HSICS 真实指数 (sector_index=close,
    sector_high/sector_low 用于周KDJ 的 HHV/LLV, 与 HSI 版同口径)。"""
    df = df.sort_values('date').reset_index(drop=True)
    close = df['sector_index'].astype(float)
    high = df['sector_high'].astype(float)
    low = df['sector_low'].astype(float)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = (100 - 100 / (1 + rs)).fillna(50)

    # Capitulation Z (20d): 当日 logret 在前20日 logret 分布的 z-score
    logret = np.log(close / close.shift(1))
    _prior = logret.shift(1)
    df['cap_z'] = (logret - _prior.rolling(20).mean()) / _prior.rolling(20).std()

    # 周线 KDJ (用真实 high/low, resample W-FRI: high=max, low=min, close=last)
    idx = pd.to_datetime(df['date'], format='%Y%m%d')
    dw = pd.DataFrame({'high': high.values, 'low': low.values, 'close': close.values}, index=idx)
    df_w = dw.resample('W-FRI').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    wc = df_w['close'].values; wh = df_w['high'].values; wl = df_w['low'].values
    llv = pd.Series(wl).rolling(9, min_periods=9).min()
    hhv = pd.Series(wh).rolling(9, min_periods=9).max()
    denom = hhv - llv
    rsv = np.where(denom == 0, 50.0, (wc - llv) / denom * 100)
    rsv = pd.Series(rsv).fillna(50).values
    nw = len(wc)
    wk = np.full(nw, 50.0); wd = np.full(nw, 50.0)
    for i in range(1, nw):
        wk[i] = 2 / 3 * wk[i - 1] + 1 / 3 * rsv[i]
        wd[i] = 2 / 3 * wd[i - 1] + 1 / 3 * wk[i]
    wj = 3 * wk - 2 * wd
    w_j_low4 = pd.Series(wj).rolling(4, min_periods=4).min().values
    w_records = pd.DataFrame({'date': df_w.index, 'w_kdj_j': wj, 'w_kdj_j_low4': w_j_low4})
    df['w_kdj_j'] = np.nan
    df['w_kdj_j_low4'] = np.nan
    d_dates = pd.to_datetime(df['date'], format='%Y%m%d')
    for _, wr in w_records.iterrows():
        mask = d_dates >= wr['date']
        df.loc[mask, 'w_kdj_j'] = wr['w_kdj_j']
        df.loc[mask, 'w_kdj_j_low4'] = wr['w_kdj_j_low4']

    # 4周累计 active_pct 百分位 (expanding rank)
    df['cum_pct_4w'] = df['active_pct'].rolling(20, min_periods=20).sum()
    df['pctile_4w'] = df['cum_pct_4w'].expanding(min_periods=60).rank(pct=True) * 100

    # expanding 阈值 (无未来信息, shift(1), 252日热身)
    df['b90'] = df['breadth_below_ma50'].expanding(min_periods=252).quantile(0.90).shift(1)
    df['b95'] = df['breadth_below_ma50'].expanding(min_periods=252).quantile(0.95).shift(1)
    df['union99'] = df['union_pct'].expanding(min_periods=252).quantile(0.99).shift(1)

    rsi_lb = 10
    df['c_rsi'] = (df['rsi'].rolling(rsi_lb, min_periods=1).min() < 30).astype(int)
    df['c_kdj'] = (df['w_kdj_j_low4'] < 10).fillna(False).astype(int)
    df['c_doji'] = (df['pctile_4w'] >= 90).astype(int)
    df['c_brd'] = (df['breadth_below_ma50'] >= df['b90']).fillna(False).astype(int)
    df['c_cap'] = (df['cap_z'] <= -2.5).astype(int)
    df['c_union'] = (df['union_pct'] >= df['union99']).fillna(False).astype(int)
    df['score'] = df[['c_rsi', 'c_kdj', 'c_doji', 'c_brd', 'c_cap', 'c_union']].sum(axis=1)

    for N in (5, 20, 60):
        df[f'fwd_{N}'] = df['sector_index'].shift(-N) / df['sector_index'] - 1
    return df


def score_sector_top(df: pd.DataFrame) -> pd.DataFrame:
    """单个 sector 逃顶打分 (镜像 HSI plot_top 逻辑): 10 条件 + score + 前瞻收益。
    df 需含 sector_index/high/low + top breadth 列 (active_top_pct, union_up_pct, sky_vol_pct,
    vol_price_div_pct, dist_top_pct, shrink_new_high_pct, breadth_below_ma50)。"""
    df = df.sort_values('date').reset_index(drop=True)
    close = df['sector_index'].astype(float)
    high = df['sector_high'].astype(float)
    low = df['sector_low'].astype(float)

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    df['rsi'] = (100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))).fillna(50)

    # Blow-off Z (20d)
    logret = np.log(close / close.shift(1))
    _prior = logret.shift(1)
    df['cap_z'] = (logret - _prior.rolling(20).mean()) / _prior.rolling(20).std()

    # 周线 KDJ (high4 用于顶)
    idx = pd.to_datetime(df['date'], format='%Y%m%d')
    dw = pd.DataFrame({'high': high.values, 'low': low.values, 'close': close.values}, index=idx)
    df_w = dw.resample('W-FRI').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    wc = df_w['close'].values; wh = df_w['high'].values; wl = df_w['low'].values
    llv = pd.Series(wl).rolling(9, min_periods=9).min()
    hhv = pd.Series(wh).rolling(9, min_periods=9).max()
    denom = hhv - llv
    rsv = np.where(denom == 0, 50.0, (wc - llv) / denom * 100)
    rsv = pd.Series(rsv).fillna(50).values
    nw = len(wc)
    wk = np.full(nw, 50.0); wd = np.full(nw, 50.0)
    for i in range(1, nw):
        wk[i] = 2 / 3 * wk[i - 1] + 1 / 3 * rsv[i]
        wd[i] = 2 / 3 * wd[i - 1] + 1 / 3 * wk[i]
    wj = 3 * wk - 2 * wd
    w_j_high4 = pd.Series(wj).rolling(4, min_periods=4).max().values
    w_records = pd.DataFrame({'date': df_w.index, 'w_kdj_j': wj, 'w_kdj_j_high4': w_j_high4})
    df['w_kdj_j'] = np.nan
    df['w_kdj_j_high4'] = np.nan
    d_dates = pd.to_datetime(df['date'], format='%Y%m%d')
    for _, wr in w_records.iterrows():
        mask = d_dates >= wr['date']
        df.loc[mask, 'w_kdj_j'] = wr['w_kdj_j']
        df.loc[mask, 'w_kdj_j_high4'] = wr['w_kdj_j_high4']

    # BIAS20
    df['bias20'] = (close / close.rolling(20, min_periods=20).mean() - 1) * 100

    # 4周累计 active_top_pct 百分位
    df['cum_pct_4w'] = df['active_top_pct'].rolling(20, min_periods=20).sum()
    df['pctile_4w'] = df['cum_pct_4w'].expanding(min_periods=60).rank(pct=True) * 100

    # expanding 阈值
    b10 = df['breadth_below_ma50'].expanding(min_periods=252).quantile(0.10).shift(1)
    union_up99 = df['union_up_pct'].expanding(min_periods=252).quantile(0.99).shift(1)
    sky99 = df['sky_vol_pct'].expanding(min_periods=252).quantile(0.99).shift(1)
    dist95 = df['dist_top_pct'].expanding(min_periods=252).quantile(0.95).shift(1)
    shrink95 = df['shrink_new_high_pct'].expanding(min_periods=252).quantile(0.95).shift(1)
    bias95 = df['bias20'].expanding(min_periods=252).quantile(0.95).shift(1)

    LOOK = 10
    recent = lambda s: s.rolling(LOOK, min_periods=1).max().fillna(0).astype(int)
    df['c_rsi'] = (df['rsi'].rolling(10, min_periods=1).max() > 70).astype(int)
    df['c_kdj'] = recent((df['w_kdj_j_high4'] > 100).fillna(False))
    df['c_star'] = (df['pctile_4w'] >= 90).astype(int)
    df['c_brd'] = (df['breadth_below_ma50'] <= b10).fillna(False).astype(int)
    df['c_cap'] = recent((df['cap_z'] >= 2.5).fillna(False))
    df['c_union'] = (df['union_up_pct'] >= union_up99).fillna(False).astype(int)
    df['c_sky'] = (df['sky_vol_pct'] >= sky99).fillna(False).astype(int)
    df['c_div'] = (df['vol_price_div_pct'] > 0).astype(int)
    df['c_dist'] = (df['dist_top_pct'] >= dist95).fillna(False).astype(int)
    df['c_shrink'] = (df['shrink_new_high_pct'] >= shrink95).fillna(False).astype(int)
    df['c_bias'] = recent((df['bias20'] >= bias95).fillna(False))
    df['score'] = df[['c_rsi', 'c_kdj', 'c_star', 'c_brd', 'c_cap', 'c_union',
                      'c_sky', 'c_div', 'c_dist', 'c_shrink', 'c_bias']].sum(axis=1)
    for N in (5, 20, 60):
        df[f'fwd_{N}'] = df['sector_index'].shift(-N) / df['sector_index'] - 1
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2014-01-01')
    ap.add_argument('--end', default='2026-07-03')
    args = ap.parse_args()

    pool = sc.load_pool()[['Ticker', 'Sector']]
    print(f"=== Backtest Sector Doji ({args.start} ~ {args.end}) ===\n[Pool] {len(pool)} HK stocks, {pool['Sector'].nunique()} sectors")

    end = args.end.replace('-', '')
    start = (pd.to_datetime(args.start) - pd.Timedelta(days=400)).strftime('%Y%m%d')
    codes = list(pool['Ticker'])
    from hk_data import fetch_hk_stocks
    print(f"[DB] 拉取 {len(codes)} 只港股 EOD ({start}~{end}) [jianxin→EODHD→yfinance] + HSICS 行业指数 ...")
    raw = fetch_hk_stocks(codes, start, end).rename(columns={'S_INFO_WINDCODE': 'code'})
    conn = pymysql.connect(**DB_CONFIG)
    hsics = fetch_hsics(conn, start, end)
    conn.close()
    print(f"[Fetch] {raw['code'].nunique()} stocks, {hsics['Sector'].nunique()} HSICS indices")

    # 预计算每只股票的信号 (compute_one 内部用 fwd_close, 不需要再收集价格)
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

    # 筛回测区间 (热身数据已用于 compute_one 内的 rolling, 这里裁到 bt 区间做聚合/打分)
    bt_start = args.start.replace('-', '')
    bt_end = args.end.replace('-', '')
    sig = sig[(sig['date'] >= bt_start) & (sig['date'] <= bt_end)].copy()
    hsics = hsics[(hsics['date'] >= bt_start) & (hsics['date'] <= bt_end)].copy()

    # merge sector
    sig = sig.merge(pool, left_on='code', right_on='Ticker', how='left').drop(columns='Ticker')
    sig = sig.dropna(subset=['Sector'])

    # 完整信号 + active 状态机 (按 code 跑, 与原版一致)
    sig['signal'] = (
        sig['is_doji_5d'] &
        (sig['daily_j_low5'] < 10) &
        (sig['weekly_j_low4'] < 10) &
        (sig['has_div'] | sig['has_surge_5d'])
    )
    sig = compute_active(sig)

    # 按 (sector, date) 聚合
    daily = sig.groupby(['Sector', 'date']).agg(
        signal_count=('signal', 'sum'),
        active_count=('active', 'sum'),
        total_count=('code', 'nunique'),
        below_ma50_count=('below_ma50', 'sum'),
        vol_surge_count=('is_vol_surge', 'sum'),
        big_drop_count=('is_big_drop', 'sum'),
        panic_count=('is_panic', 'sum'),
        union_count=('is_union', 'sum'),
    ).reset_index()
    daily['pct'] = daily['signal_count'] / daily['total_count'] * 100
    daily['active_pct'] = daily['active_count'] / daily['total_count'] * 100
    daily['breadth_below_ma50'] = daily['below_ma50_count'] / daily['total_count'] * 100
    daily['vol_surge_pct'] = daily['vol_surge_count'] / daily['total_count'] * 100
    daily['big_drop_pct'] = daily['big_drop_count'] / daily['total_count'] * 100
    daily['panic_pct'] = daily['panic_count'] / daily['total_count'] * 100
    daily['union_pct'] = daily['union_count'] / daily['total_count'] * 100

    # HSICS 真实行业指数作为 sector_index (close) + high/low (用于周KDJ)
    daily = daily.merge(hsics, on=['Sector', 'date'], how='left')

    # 每个 sector 打分
    out = []
    for sector, g in daily.groupby('Sector'):
        out.append(score_sector(g))
    daily = pd.concat(out, ignore_index=True)
    daily = daily.sort_values(['Sector', 'date']).reset_index(drop=True)

    keep = ['date', 'Sector', 'sector_index', 'rsi', 'w_kdj_j', 'w_kdj_j_low4',
            'active_pct', 'cum_pct_4w', 'pctile_4w', 'breadth_below_ma50', 'b90', 'b95',
            'cap_z', 'union_pct', 'union99', 'panic_pct', 'vol_surge_pct', 'big_drop_pct',
            'c_rsi', 'c_kdj', 'c_doji', 'c_brd', 'c_cap', 'c_union', 'score',
            'fwd_5', 'fwd_20', 'fwd_60', 'total_count', 'signal_count']
    daily = daily[keep]
    out_path = os.path.join(os.path.dirname(__file__), '..', 'output', 'doji_signal_daily_by_sector.csv')
    daily.to_csv(out_path, index=False, encoding='utf-8-sig')

    print(f"\n[Done] {len(daily)} rows ({daily['Sector'].nunique()} sectors × {daily['date'].nunique()} days) → {out_path}")
    print(summary_text(daily))

    # ============ 逃顶 (top) ============
    sig['top_signal'] = (
        sig['is_uptrend'] &
        sig['is_star_5d'] &
        (sig['daily_j_high5'] > 90) &
        (sig['weekly_j_high4'] > 90) &
        (sig['has_bear_div'] | sig['has_surge_5d'])
    )
    sig = compute_active_top(sig)
    daily_t = sig.groupby(['Sector', 'date']).agg(
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
    daily_t = daily_t.merge(hsics, on=['Sector', 'date'], how='left')
    out_t = []
    for sector, g in daily_t.groupby('Sector'):
        out_t.append(score_sector_top(g))
    daily_t = pd.concat(out_t, ignore_index=True).sort_values(['Sector', 'date']).reset_index(drop=True)
    keep_t = ['date', 'Sector', 'sector_index', 'rsi', 'w_kdj_j', 'w_kdj_j_high4', 'bias20',
              'active_top_pct', 'cum_pct_4w', 'pctile_4w', 'breadth_below_ma50', 'cap_z',
              'union_up_pct', 'sky_vol_pct', 'vol_price_div_pct', 'dist_top_pct', 'shrink_new_high_pct',
              'c_rsi', 'c_kdj', 'c_star', 'c_brd', 'c_cap', 'c_union', 'c_sky', 'c_div', 'c_dist', 'c_shrink', 'c_bias',
              'score', 'fwd_5', 'fwd_20', 'fwd_60', 'total_count', 'top_signal_count']
    daily_t = daily_t[keep_t]
    out_path_t = os.path.join(os.path.dirname(__file__), '..', 'output', 'top_signal_daily_by_sector.csv')
    daily_t.to_csv(out_path_t, index=False, encoding='utf-8-sig')
    print(f"\n[Top] {len(daily_t)} rows → {out_path_t}")
    print(summary_text_top(daily_t))


def summary_text(daily: pd.DataFrame) -> str:
    lines = ["\n--- 各 sector score≥4 信号统计 ---"]
    for sector, g in daily.groupby('Sector'):
        mask = g['score'] >= 4
        n = int(mask.sum())
        if n:
            wr5 = (g.loc[mask, 'fwd_5'] > 0).mean() * 100
            wr20 = (g.loc[mask, 'fwd_20'] > 0).mean() * 100
            wr60 = (g.loc[mask, 'fwd_60'] > 0).mean() * 100
            lines.append(f"  {sector:28s} n={n:4d}  胜率 5/20/60 = {wr5:3.0f}/{wr20:3.0f}/{wr60:3.0f}")
        else:
            lines.append(f"  {sector:28s} n=   0  (无 score≥4 信号)")
    return "\n".join(lines)


def summary_text_top(daily: pd.DataFrame) -> str:
    lines = ["\n--- 各 sector 逃顶 score≥5 信号统计 (下跌概率) ---"]
    for sector, g in daily.groupby('Sector'):
        mask = g['score'] >= 5
        n = int(mask.sum())
        if n:
            wr5 = (g.loc[mask, 'fwd_5'].dropna() < 0).mean() * 100
            wr20 = (g.loc[mask, 'fwd_20'].dropna() < 0).mean() * 100
            wr60 = (g.loc[mask, 'fwd_60'].dropna() < 0).mean() * 100
            lines.append(f"  {sector:28s} n={n:4d}  下跌概率 5/20/60 = {wr5:3.0f}/{wr20:3.0f}/{wr60:3.0f}")
        else:
            lines.append(f"  {sector:28s} n=   0  (无 score≥5 信号)")
    return "\n".join(lines)


if __name__ == '__main__':
    main()
