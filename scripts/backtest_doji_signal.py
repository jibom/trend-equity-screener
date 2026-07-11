"""回测: 统计过去5年每日港股池满足"低位十字星+背离/倍量"信号的个股占比, 与HSI对比.

逻辑: 对每个交易日, 统计同时满足以下条件的股票数:
  1) 近5日有十字星 (阶梯阈值)
  2) 日线KDJ近5日最低J<10
  3) 周线KDJ近4周最低J<10
  4) J或MACD柱底背离 (近10日) OR 十字星日倍量(≥2×30均量)
  排除"仅缩量无背离"的票.

输出: output/doji_signal_daily.csv (date, signal_count, total_count, pct, hsi_close)
用法: python scripts/backtest_doji_signal.py [--start 2021-07-01] [--end 2026-07-02]
"""
from __future__ import annotations
import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pymysql, pandas as pd, numpy as np
from db_config import DB_CONFIG
import sector_cluster as sc
from kdj_div_basic import calc_kdj


def doji_body_threshold(close):
    if close < 10: return 0.015
    if close < 50: return 0.03
    if close < 100: return 0.10
    return 0.20


def compute_macd(close, fast=12, slow=26, signal=9):
    s = pd.Series(close)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    return ((dif - dea) * 2).values


def compute_one(gg):
    """对单只股票的全部历史预计算: 每日的十字星/J/背离/量能标记.
    返回 DataFrame: date, is_doji_5d, daily_j_low5, weekly_j_low4, has_div, has_surge."""
    if gg is None or len(gg) < 60:
        return None

    c = gg['fwd_close'].values
    o = gg['fwd_open'].values if 'fwd_open' in gg else gg['S_DQ_ADJOPEN'].values
    h = gg['fwd_high'].values if 'fwd_high' in gg else gg['S_DQ_ADJHIGH'].values
    lo = gg['fwd_low'].values if 'fwd_low' in gg else gg['S_DQ_ADJLOW'].values
    vol = gg['vol'].values
    dates = gg['TRADE_DT'].values
    n = len(c)

    # 日线 KDJ
    kdj_df = calc_kdj(pd.DataFrame({'fwd_close': c, 'fwd_high': h, 'fwd_low': lo}))
    daily_j = kdj_df['j'].values

    # MACD
    hist = compute_macd(c)

    # 十字星标记 (每日)
    is_doji = np.zeros(n, dtype=bool)
    for i in range(n):
        rng = h[i] - lo[i]
        if rng <= 0:
            continue
        body = abs(c[i] - o[i])
        thr = doji_body_threshold(c[i])
        if body <= thr or (body / rng <= 0.05 and body <= thr * 1.5):
            is_doji[i] = True

    # 近5日有十字星 (滚动)
    doji_5d = pd.Series(is_doji).rolling(5, min_periods=1).max().values.astype(bool)

    # 射击之星/长上影 (顶部反转K线, is_doji 的镜像): 上影≥2×实体, 下影≤0.5×实体, 实体≤40%振幅
    is_star = np.zeros(n, dtype=bool)
    for i in range(n):
        rng = h[i] - lo[i]
        if rng <= 0:
            continue
        body = abs(c[i] - o[i])
        upper = h[i] - max(c[i], o[i])
        lower = min(c[i], o[i]) - lo[i]
        b = body if body > 0 else 1e-9
        if upper >= 2 * b and lower <= 0.5 * b and body / rng <= 0.4:
            is_star[i] = True
    star_5d = pd.Series(is_star).rolling(5, min_periods=1).max().values.astype(bool)

    # 日线J近5日最低 / 最高 (镜像)
    daily_j_low5 = pd.Series(daily_j).rolling(5, min_periods=5).min().values
    daily_j_high5 = pd.Series(daily_j).rolling(5, min_periods=5).max().values

    # 周线 KDJ
    df_w = pd.DataFrame({'close': c, 'high': h, 'low': lo, 'date': pd.to_datetime(dates, format='%Y%m%d')})
    df_w = df_w.set_index('date').resample('W-FRI').agg({'close': 'last', 'high': 'max', 'low': 'min'}).dropna()
    wc = df_w['close'].values; wh = df_w['high'].values; wl = df_w['low'].values
    wdates = df_w.index
    if len(wc) < 9:
        return None
    llv = pd.Series(wl).rolling(9, min_periods=9).min()
    hhv = pd.Series(wh).rolling(9, min_periods=9).max()
    denom = hhv - llv
    rsv = np.where(denom == 0, 50.0, (wc - llv) / denom * 100)
    rsv = pd.Series(rsv).fillna(50).values
    nw = len(wc)
    wk = np.full(nw, 50.0); wd = np.full(nw, 50.0)
    for i in range(1, nw):
        wk[i] = 2/3 * wk[i-1] + 1/3 * rsv[i]
        wd[i] = 2/3 * wd[i-1] + 1/3 * wk[i]
    wj = 3 * wk - 2 * wd
    weekly_j_low4 = pd.Series(wj).rolling(4, min_periods=4).min().values
    weekly_j_high4 = pd.Series(wj).rolling(4, min_periods=4).max().values

    # 把周线J映射到日线
    weekly_j_map = np.full(n, np.nan)
    weekly_j_high_map = np.full(n, np.nan)
    for i in range(n):
        d = pd.Timestamp(dates[i], ).to_period('W-FRI').end_time.normalize()
        for wi in range(len(wdates)):
            if wdates[wi] >= d:
                weekly_j_map[i] = weekly_j_low4[wi] if wi < len(weekly_j_low4) else np.nan
                weekly_j_high_map[i] = weekly_j_high4[wi] if wi < len(weekly_j_high4) else np.nan
                break

    # 背离检测 (近10日, 在每个交易日检查)
    has_div = np.zeros(n, dtype=bool)
    for i in range(10, n):
        p = c[i-9:i+1]; j = daily_j[i-9:i+1]; hh = hist[i-9:i+1]
        pmin_idx = np.argmin(p)
        if pmin_idx < 2:
            continue
        p_low = p[pmin_idx]
        prev_low = np.min(p[:pmin_idx])
        if p_low > prev_low * 1.001:
            continue
        j_low = j[pmin_idx]; j_prev = j[np.argmin(p[:pmin_idx])]
        if not (np.isnan(j_low) or np.isnan(j_prev)) and j_low > j_prev:
            has_div[i] = True; continue
        h_low = hh[pmin_idx]; h_prev = hh[np.argmin(p[:pmin_idx])]
        if not (np.isnan(h_low) or np.isnan(h_prev)) and h_low > h_prev:
            has_div[i] = True

    # 顶背离 (镜像): 价格创新高, J/MACD柱 高点走低
    has_bear_div = np.zeros(n, dtype=bool)
    for i in range(10, n):
        p = c[i-9:i+1]; j = daily_j[i-9:i+1]; hh = hist[i-9:i+1]
        pmax_idx = np.argmax(p)
        if pmax_idx < 2:
            continue
        p_high = p[pmax_idx]
        prev_high = np.max(p[:pmax_idx])
        if p_high < prev_high * 0.999:
            continue
        j_high = j[pmax_idx]; j_prev = j[np.argmax(p[:pmax_idx])]
        if not (np.isnan(j_high) or np.isnan(j_prev)) and j_high < j_prev:
            has_bear_div[i] = True; continue
        h_high = hh[pmax_idx]; h_prev = hh[np.argmax(p[:pmax_idx])]
        if not (np.isnan(h_high) or np.isnan(h_prev)) and h_high < h_prev:
            has_bear_div[i] = True

    # 量能: 十字星日倍量
    vol_ma30 = pd.Series(vol).rolling(30, min_periods=30).mean().values
    has_surge = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_doji[i] and not np.isnan(vol_ma30[i]) and vol_ma30[i] > 0:
            if vol[i] >= 2.0 * vol_ma30[i]:
                has_surge[i] = True
    # 近5日有倍量
    surge_5d = pd.Series(has_surge).rolling(5, min_periods=1).max().values.astype(bool)

    # MA50 + 收盘是否低于MA50 (用于宽度指标)
    ma50 = pd.Series(c).rolling(50, min_periods=50).mean().values
    below_ma50 = (c < ma50).astype(bool)

    # 量比 = 当日量 / 过去5日平均量; 放量 = 量比 > 2
    vol_ma5 = pd.Series(vol).rolling(5, min_periods=5).mean().shift(1).values
    vol_ratio = np.where((vol_ma5 is not None) & (vol_ma5 > 0), vol / vol_ma5, np.nan)
    is_vol_surge = vol_ratio > 2.0
    # 大跌 = 单日跌幅 > 3% / 大涨 = 单日涨幅 > 3% (forward-adjusted close)
    ret = pd.Series(c).pct_change().values
    is_big_drop = ret <= -0.03
    is_big_up = ret >= 0.03
    # 恐慌 = 放量 ∩ 大跌 (带量恐慌抛售 = 情绪大宣泄)
    is_panic = is_vol_surge & is_big_drop
    # 合集 = 放量 ∪ 大跌 (含放量但十字星/上行的潜在吸筹)
    is_union = is_vol_surge | is_big_drop
    # 合集(顶) = 放量 ∪ 大涨 (含放量冲高/突破的潜在派发)
    is_union_up = is_vol_surge | is_big_up

    # ---- P1 量价见顶 detector (顶部派发结构) ----
    vol_s = pd.Series(vol)
    c_s = pd.Series(c)
    h_s = pd.Series(h)
    # 高位 gate: 收盘在 MA60 之上 且 MA20>MA60 (处于上升趋势, 顶部形态才有效)
    ma20 = c_s.rolling(20, min_periods=20).mean().values
    ma60 = c_s.rolling(60, min_periods=60).mean().values
    is_uptrend = ((c > ma60) & (ma20 > ma60))

    vol_max250_prev = vol_s.rolling(250, min_periods=250).max().shift(1).values   # 前250日最大量(不含今日)
    close_max60_prev = c_s.rolling(60, min_periods=60).max().shift(1).values       # 前60日最高价
    close_max20_prev = c_s.rolling(20, min_periods=20).max().shift(1).values
    vol_ma20 = vol_s.rolling(20, min_periods=20).mean().values
    vol_ma5_mean = vol_s.rolling(5, min_periods=5).mean().values
    vol_max60_prev = vol_s.rolling(60, min_periods=60).max().shift(1).values
    vol_prev = vol_s.shift(1).values
    h_prev = h_s.shift(1).values
    ret5 = c_s.pct_change(5).values

    # 天量天价(同步): 今日量≥0.95×前250日最大量 且 创60日新高
    is_sky_vol = (vol >= 0.95 * vol_max250_prev) & (c >= close_max60_prev)
    # 量价背离天量: 昨日天量, 今日创新高(>昨高) 但量缩(<0.85×昨量) — 因果, 今日收盘判定
    is_sky_vol_prev = np.concatenate([[False], is_sky_vol[:-1]])
    is_vol_price_div = is_sky_vol_prev & (c > h_prev) & (vol_prev > 0) & (vol < 0.85 * vol_prev)
    # 放量滞涨: 近5日均量>1.5×20日均量 且 未创20日新高 且 5日涨跌幅<3% (高位横盘派发)
    is_dist_top = (vol_ma5_mean > 1.5 * vol_ma20) & (c < close_max20_prev) & (np.abs(ret5) < 0.03)
    # 缩量新高: 创60日新高 但 量<0.6×前60日最大量 (无量诱多顶)
    is_shrink_new_high = (c >= close_max60_prev) & (vol < 0.6 * vol_max60_prev)

    return pd.DataFrame({
        'date': dates,
        'is_doji_5d': doji_5d,
        'is_star_5d': star_5d,
        'daily_j': daily_j,
        'daily_j_low5': daily_j_low5,
        'daily_j_high5': daily_j_high5,
        'weekly_j_low4': weekly_j_map,
        'weekly_j_high4': weekly_j_high_map,
        'has_div': has_div,
        'has_bear_div': has_bear_div,
        'has_surge_5d': surge_5d,
        'below_ma50': below_ma50,
        'is_uptrend': is_uptrend,
        'is_vol_surge': is_vol_surge,
        'is_big_drop': is_big_drop,
        'is_big_up': is_big_up,
        'is_panic': is_panic,
        'is_union': is_union,
        'is_union_up': is_union_up,
        'is_sky_vol': is_sky_vol,
        'is_vol_price_div': is_vol_price_div,
        'is_dist_top': is_dist_top,
        'is_shrink_new_high': is_shrink_new_high,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2021-07-01')
    ap.add_argument('--end', default='2026-07-02')
    args = ap.parse_args()

    pool = sc.load_pool()
    print(f"=== Backtest Doji Signal ({args.start} ~ {args.end}) ===\n[Pool] {len(pool)} HK stocks")

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

    # 筛选回测区间
    bt_start = args.start.replace('-', '')
    bt_end = args.end.replace('-', '')
    sig = sig[(sig['date'] >= bt_start) & (sig['date'] <= bt_end)].copy()

    # 每日统计: 满足全部条件的股票
    # 条件: doji_5d AND daily_j_low5<10 AND weekly_j_low4<10 AND (has_div OR has_surge_5d)
    sig['signal'] = (
        sig['is_doji_5d'] &
        (sig['daily_j_low5'] < 10) &
        (sig['weekly_j_low4'] < 10) &
        (sig['has_div'] | sig['has_surge_5d'])
    )

    # --- 卖盘衰竭状态机: 不用时间窗口, 用KDJ状态判断 ---
    # 入池: 触发完整信号 (doji + J<10 + div/vol)
    # 留池: 日J < 20 (仍超卖, 卖盘衰竭持续)
    # 出池: 日J >= 20 连续3天 (KDJ回升, 卖压消退)
    sig = sig.sort_values(['code', 'date']).reset_index(drop=True)

    # 需要每日的J值 (不是5日最低, 是当天的J)
    # 重新算: daily_j 已经在 compute_one 里算过, 但返回的是 daily_j_low5 (5日最低)
    # 需要原始 daily_j, 在 compute_one 里加一列
    # 这里用 daily_j_low5<20 作为近似 (5日内最低J<20 = 5日内有J<20的超卖)
    # 但更准确的是当天J值. compute_one 返回的 daily_j_low5 是5日最低, 当天J值可能更高
    # 修改: 用 daily_j_low5 < 20 表示"近5日仍有超卖" = 卖盘衰竭仍在持续

    exhausted = np.zeros(len(sig), dtype=bool)
    for code, idx in sig.groupby('code').groups.items():
        idx = list(idx)
        in_pool = False
        days_above_20 = 0
        for i in idx:
            row = sig.iloc[i]
            triggered = row['signal']
            j_today = row['daily_j']  # 当天J值

            if triggered:
                in_pool = True
                days_above_20 = 0
            elif in_pool:
                if pd.notna(j_today) and j_today < 20:
                    # 当天仍超卖, 留池
                    days_above_20 = 0
                else:
                    days_above_20 += 1
                    if days_above_20 >= 3:
                        in_pool = False
            exhausted[i] = in_pool

    sig['active'] = exhausted

    daily = sig.groupby('date').agg(
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
    daily = daily.merge(hsi, on='date', how='left')

    out = os.path.join(os.path.dirname(__file__), '..', 'output', 'doji_signal_daily.csv')
    daily.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n[Done] {len(daily)} trading days → {out}")
    print(f"  信号日(>0): {(daily['signal_count'] > 0).sum()} / {len(daily)}")
    print(f"  平均占比: {daily['pct'].mean():.2f}%  活跃占比: {daily['active_pct'].mean():.2f}%")
    print(f"  最大占比: {daily['pct'].max():.2f}%  最大活跃占比: {daily['active_pct'].max():.2f}% (on {daily.loc[daily['active_pct'].idxmax(), 'date']})")


if __name__ == '__main__':
    main()
