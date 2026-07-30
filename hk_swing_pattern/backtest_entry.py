"""建仓策略回测: 长期趋势强 + 回调超卖/反转信号 + SOS + 趋势建立 → 买入
回测过去3年, 看 forward return 和胜率。

策略:
  池筛选(每日截面): OR(RS_HSI > 75百分位, 年线斜率 > 75百分位)
  买入条件(全部AND):
    1. 周线J极值 <= 25 AND 至少1个反转信号(周度背离/周度DeMark9或13/日度背离=1/climax=-1)
    2. SOS = 1
    3. 至少1个趋势建立(日度DeMark/日MACD=1/5_10 Cross=1/10_50 Cross=1)
  退出: 固定持有期 5/10/20/60/120/250 交易日; 期间触及 -8% 固定止损 或 峰值回撤12%移动止损 则提前平仓
"""
from __future__ import annotations
import sys, io, time, argparse, datetime as dt
from pathlib import Path
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

import eodhd
from data_provider import forward_adjust
from kdj_divergence import calc_kdj, resample_weekly
import patterns as pat
import climax as CX

HORIZONS = [5, 10, 20, 60, 120, 250]
LOOKBACK_DAYS = 1100  # ~3年 + MA200缓冲
STOP_LOSS = 0.08  # 固定止损: 入场价 -8%
TRAIL_STOP = 0.12  # 移动止损: 入场后峰值close回撤 12%


# ============ Helper: DeMark 逐日 ============
def demark_per_day(close, high, low, recency=8):
    """返回 buy9, sell9, buy13, sell13 逐日 bool 数组。"""
    n = len(close)
    buy_setup, sell_setup = pat._td_setup_counts(close)
    buy9 = np.zeros(n, dtype=bool); sell9 = np.zeros(n, dtype=bool)
    buy13 = np.zeros(n, dtype=bool); sell13 = np.zeros(n, dtype=bool)

    # buy9/sell9: count>=9 OR 首次达9在recency日内
    b9_events = [i for i in range(n) if buy_setup[i] >= 9 and (i == 0 or buy_setup[i-1] < 9)]
    s9_events = [i for i in range(n) if sell_setup[i] >= 9 and (i == 0 or sell_setup[i-1] < 9)]
    for i in range(n):
        if buy_setup[i] >= 9 or any(s <= i <= s + recency for s in b9_events):
            buy9[i] = True
        if sell_setup[i] >= 9 or any(s <= i <= s + recency for s in s9_events):
            sell9[i] = True

    # countdown 13
    bcd_done = -1; scd_done = -1; b_setup_hi = np.inf; s_setup_lo = -np.inf
    bcd = 0; scd = 0; bcd_completions = []; scd_completions = []
    prev_b = 0; prev_s = 0
    for i in range(n):
        if buy_setup[i] >= 9 and prev_b < 9 and i >= 8:
            bcd_done = i; b_setup_hi = float(np.max(high[i-8:i+1])); scd = 0
        if sell_setup[i] >= 9 and prev_s < 9 and i >= 8:
            scd_done = i; s_setup_lo = float(np.min(low[i-8:i+1])); bcd = 0
        if bcd_done >= 0 and i > bcd_done and i >= 2:
            if close[i] > b_setup_hi:
                bcd = 0
            elif bcd < 13 and close[i] < low[i-2]:
                bcd += 1
                if bcd == 13: bcd_completions.append(i)
        if scd_done >= 0 and i > scd_done and i >= 2:
            if close[i] < s_setup_lo:
                scd = 0
            elif scd < 13 and close[i] > high[i-2]:
                scd += 1
                if scd == 13: scd_completions.append(i)
        prev_b = buy_setup[i]; prev_s = sell_setup[i]

    for i in range(n):
        if any(i - recency <= j <= i for j in bcd_completions):
            buy13[i] = True
        if any(i - recency <= j <= i for j in scd_completions):
            sell13[i] = True
    return buy9, sell9, buy13, sell13


# ============ Helper: Cross 逐日 ============
def cross_per_day(fast, slow, close, filter_ma=None):
    """返回逐日: 1=近2日金叉, -1=近2日死叉, 0=无。filter_ma: close需>=该MA才报。"""
    n = len(fast)
    result = np.zeros(n)
    gap = fast - slow
    last_up = -1; last_dn = -1
    for i in range(1, n):
        if np.isnan(gap[i]) or np.isnan(gap[i-1]):
            continue
        if gap[i-1] <= 0 < gap[i]:
            last_up = i
        elif gap[i-1] >= 0 > gap[i]:
            last_dn = i
        if filter_ma is not None and (np.isnan(filter_ma[i]) or close[i] < filter_ma[i]):
            continue
        if last_up > last_dn and last_up >= 0 and i - last_up <= 2:
            result[i] = 1
        elif last_dn > last_up and last_dn >= 0 and i - last_dn <= 2:
            result[i] = -1
    return result


# ============ Helper: SOS 逐日 ============
# 位置门 = A OR B:
#   A: 6个月(pos_window=126)区间pos<=0.50; 6个月数据不足回退3个月(pos_window_fallback=63)  (从长期低位反强)
#   B: 近 entangle_lookback 日 4均线(5/10/15/20)纠缠 max/min-1<entangle_thresh  (从均线纠缠平台启动)
# 强度条件: vol>=1.5x & 阳线(body>=3% & range>=1.5x & close_pos>=0.70)。无十字星路径
def sos_per_day(c, o, h, l, v, pos_window=126, pos_window_fallback=63, entangle_thresh=0.05, entangle_lookback=3):
    n = len(c)
    sos = np.zeros(n, dtype=bool)
    rng = h - l; body = c - o
    avg_rng = pd.Series(rng).rolling(10, min_periods=5).mean().values
    vol_ma = pd.Series(v).rolling(20, min_periods=5).mean().values
    # A: pos 6个月, 不足回退3个月
    rmin1 = pd.Series(c).rolling(pos_window, min_periods=pos_window).min().values
    rmax1 = pd.Series(c).rolling(pos_window, min_periods=pos_window).max().values
    pos1 = (c - rmin1) / np.where(rmax1 > rmin1, rmax1 - rmin1, np.nan)
    rmin2 = pd.Series(c).rolling(pos_window_fallback, min_periods=pos_window_fallback).min().values
    rmax2 = pd.Series(c).rolling(pos_window_fallback, min_periods=pos_window_fallback).max().values
    pos2 = (c - rmin2) / np.where(rmax2 > rmin2, rmax2 - rmin2, np.nan)
    pos = np.where(np.isnan(pos1), pos2, pos1)
    # 4均线(5/10/15/20)纠缠逐日
    mas = np.column_stack([pd.Series(c).rolling(w, min_periods=w).mean().values for w in (5,10,15,20)])
    ma_valid = ~np.isnan(mas).any(axis=1)
    ma_max = np.full(n, np.nan); ma_min = np.full(n, np.nan)
    if ma_valid.any():
        ma_max[ma_valid] = np.nanmax(mas[ma_valid], axis=1)
        ma_min[ma_valid] = np.nanmin(mas[ma_valid], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        spread = ma_max / np.where(ma_min > 0, ma_min, np.nan) - 1
    entangled = (spread < entangle_thresh) & ma_valid
    ent_recent = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - entangle_lookback + 1)
        ent_recent[i] = bool(entangled[lo:i+1].any())
    for i in range(n):
        for j in range(max(0, i-2), i+1):
            if j < 25 or np.isnan(avg_rng[j]) or np.isnan(vol_ma[j]) or rng[j] <= 0:
                continue
            # 位置门 A OR B (pos NaN时只看B)
            pos_ok = (not np.isnan(pos[j])) and pos[j] <= 0.50
            if not (pos_ok or ent_recent[j]):
                continue
            if v[j] < 1.5 * vol_ma[j]:
                continue
            if body[j] > 0 and o[j] > 0 and body[j] / o[j] >= 0.03:
                if rng[j] >= 1.5 * avg_rng[j] and (c[j] - l[j]) / rng[j] >= 0.70:
                    sos[i] = True; break
    return sos


# ============ Helper: 周线J swing 逐日 ============
def weekly_j_swing_per_day(dates, wk_dates, wk_j):
    n = len(dates)
    result = np.full(n, np.nan)
    for i in range(n):
        mask = wk_dates <= dates[i]
        cnt = np.sum(mask)
        if cnt < 4:
            continue
        win = wk_j[mask][-4:]
        valid = win[~np.isnan(win)]
        if len(valid) == 0:
            continue
        ext_idx = [j for j, val in enumerate(win) if not np.isnan(val) and (val < 15 or val > 95)]
        if ext_idx:
            last = ext_idx[-1]
            result[i] = float(np.nanmax(win)) if win[last] > 95 else float(np.nanmin(win))
        else:
            cur = win[-1]; prev = win[-2] if len(win) >= 2 and not np.isnan(win[-2]) else cur
            result[i] = float(np.nanmin(win)) if cur >= prev else float(np.nanmax(win))
    return result


# ============ 逐股信号计算 ============
def compute_signals(daily, bench_2800):
    d = calc_kdj(daily.copy())
    c = d["fwd_close"].values; o = d["fwd_open"].values
    h = d["fwd_high"].values; l = d["fwd_low"].values; v = d["volume"].values
    n = len(d); dates = d["date"].values
    if n < 260:
        return None

    k_arr = d["k"].values; jj = d["j"].values; d_arr = d["d"].values
    dif, dea, _ = pat.macd(c)
    rsi_arr = pat.rsi(c)
    ma5 = pat.ma(c, 5); ma10 = pat.ma(c, 10); ma50 = pat.ma(c, 50); ma200 = pat.ma(c, 200)

    # 年线斜率
    slope200 = np.full(n, np.nan)
    for i in range(5, n):
        if not np.isnan(ma200[i]) and not np.isnan(ma200[i-5]) and ma200[i-5] > 0:
            slope200[i] = (ma200[i] / ma200[i-5] - 1) * 52 * 100

    # RS_HSI (Mansfield RS vs 2800.HK)
    rs_hsi = np.full(n, np.nan)
    if bench_2800 is not None and len(bench_2800) > 252:
        bench_s = bench_2800.set_index("date")["fwd_close"].astype(float)
        stk = pd.Series(c, index=pd.Index(dates))
        rs_ratio = (stk / bench_s).dropna()
        if len(rs_ratio) >= 252:
            mrs = (rs_ratio / rs_ratio.rolling(252, min_periods=252).mean() - 1) * 100
            for idx, val in mrs.dropna().items():
                pos = np.searchsorted(dates, np.datetime64(idx))
                if pos < n:
                    rs_hsi[pos] = val

    # Climax
    fl = CX.climax_flags(daily)
    climax_neg1 = (fl["flag"].values == -1)
    climax_pos1 = (fl["flag"].values == 1)

    # DeMark 逐日 (日线): 买9/13 + 卖9/13
    db9, ds9, db13, ds13 = demark_per_day(c, h, l, recency=8)
    daily_demark = db9 | db13
    daily_sell_demark = ds9 | ds13

    # Crossovers 逐日
    kdj_cross = cross_per_day(k_arr, d_arr, c)
    macd_cross = cross_per_day(dif, dea, c)
    cross_510 = cross_per_day(ma5, ma10, c, ma10)
    cross_1050 = cross_per_day(ma10, ma50, c, ma50)

    # 周线信号
    wk = _complete_weekly(daily)
    if len(wk) < 20:
        return None
    wk = calc_kdj(wk)
    wc = wk["fwd_close"].values; wj = wk["j"].values
    wh = wk["fwd_high"].values; wl = wk["fwd_low"].values
    wk_dates = wk["date"].values

    # 周线J swing 逐日
    wk_j_swing = weekly_j_swing_per_day(dates, wk_dates, wj)
    # 近4周J最低/最高值 (简单min/max, 不用swing逻辑)
    wk_j_min_4w = np.full(n, np.nan)
    wk_j_max_4w = np.full(n, np.nan)
    for i in range(n):
        mask = wk_dates <= dates[i]
        if np.sum(mask) >= 4:
            win = wj[mask][-4:]
            valid = win[~np.isnan(win)]
            if len(valid) > 0:
                wk_j_min_4w[i] = float(np.min(valid))
                wk_j_max_4w[i] = float(np.max(valid))

    # 周线 DeMark 逐周 → 映射到逐日 (买9/13 + 卖9/13)
    wb9, ws9, wb13, ws13 = demark_per_day(wc, wh, wl, recency=4)
    weekly_demark_9_13 = np.zeros(n, dtype=bool)
    weekly_sell_demark = np.zeros(n, dtype=bool)
    for wi in range(len(wk)):
        wk_date = wk_dates[wi]
        next_date = wk_dates[wi+1] if wi+1 < len(wk) else dates[-1] + np.timedelta64(1, 'D')
        mask = (dates >= wk_date) & (dates < next_date)
        if wb9[wi] or wb13[wi]:
            weekly_demark_9_13[mask] = True
        if ws9[wi] or ws13[wi]:
            weekly_sell_demark[mask] = True

    # 周线背离 逐周 → 映射到逐日 (分离顶/底)
    wdif, _, _ = pat.macd(wc)
    wrsi = pat.rsi(wc)
    weekly_div = np.zeros(n, dtype=bool)
    weekly_top_div = np.zeros(n, dtype=bool)
    for wi in range(40, len(wk)):
        d_kdj = pat._divergence(wc[:wi+1], wj[:wi+1], lookback=40, recent=4, zone_lo=10, zone_hi=90)
        d_macd = pat._divergence(wc[:wi+1], wdif[:wi+1], lookback=40, recent=4)
        d_rsi = pat._divergence(wc[:wi+1], wrsi[:wi+1], lookback=40, recent=4)
        # 周J swing 极值用于抑制
        sw_j = None
        win = wj[max(0, wi-3):wi+1]
        valid = win[~np.isnan(win)]
        if len(valid) > 0:
            ext_idx = [j for j, val in enumerate(win) if not np.isnan(val) and (val < 15 or val > 95)]
            if ext_idx:
                last = ext_idx[-1]
                sw_j = float(np.nanmax(win)) if win[last] > 95 else float(np.nanmin(win))
            else:
                cur = win[-1]; prev = win[-2] if len(win) >= 2 and not np.isnan(win[-2]) else cur
                sw_j = float(np.nanmin(win)) if cur >= prev else float(np.nanmax(win))
        wk_oversold = sw_j is not None and sw_j < 20
        wk_overbought = sw_j is not None and sw_j > 80
        has_bot = (d_kdj == "底背离" and not wk_overbought) or (d_macd == "底背离" and not wk_overbought) or (d_rsi == "底背离" and not wk_overbought)
        has_top = (d_kdj == "顶背离" and not wk_oversold) or (d_macd == "顶背离" and not wk_oversold) or (d_rsi == "顶背离" and not wk_oversold)
        if has_bot or has_top:
            wk_date = wk_dates[wi]
            next_date = wk_dates[wi+1] if wi+1 < len(wk) else dates[-1] + np.timedelta64(1, 'D')
            mask = (dates >= wk_date) & (dates < next_date)
            weekly_div[mask] = True
            if has_top:
                weekly_top_div[mask] = True

    # 日线背离: 底背离(J_swing<=25时算) + 顶背离(J_swing>=75时算)
    daily_div_1 = np.zeros(n, dtype=bool)
    daily_top_div_1 = np.zeros(n, dtype=bool)
    for i in range(60, n):
        sv = wk_j_swing[i]
        if np.isnan(sv):
            continue
        if sv > 25 and sv < 75:
            continue
        d_kdj = pat._divergence(c[:i+1], jj[:i+1], recent=10, min_spacing=22)
        d_macd = pat._divergence(c[:i+1], dif[:i+1], recent=10, min_spacing=22)
        d_rsi = pat._divergence(c[:i+1], rsi_arr[:i+1], recent=10, min_spacing=22)
        has_bot = d_kdj == "底背离" or d_macd == "底背离" or d_rsi == "底背离"
        has_top = d_kdj == "顶背离" or d_macd == "顶背离" or d_rsi == "顶背离"
        if sv <= 25 and has_bot and not has_top:
            daily_div_1[i] = True
        if sv >= 75 and has_top and not has_bot:
            daily_top_div_1[i] = True

    # SOS 逐日
    sos = sos_per_day(c, o, h, l, v)

    # 短期趋势分: Price/5DMV, Price/10DMV, 5D/10D, 10D/20D, 10D斜率
    mv5 = pd.Series(c * v).rolling(5, min_periods=5).sum() / pd.Series(v).rolling(5, min_periods=5).sum()
    mv10 = pd.Series(c * v).rolling(10, min_periods=10).sum() / pd.Series(v).rolling(10, min_periods=10).sum()
    mv20 = pd.Series(c * v).rolling(20, min_periods=20).sum() / pd.Series(v).rolling(20, min_periods=20).sum()
    ma10_arr = ma10
    short_trend = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(mv5.iloc[i]) or np.isnan(mv10.iloc[i]) or np.isnan(mv20.iloc[i]) or np.isnan(ma10_arr[i]):
            continue
        s1 = 1 if c[i] > mv5.iloc[i] else -1
        s2 = 1 if c[i] > mv10.iloc[i] else -1
        s3 = 1 if mv5.iloc[i] > mv10.iloc[i] else -1
        s4 = 1 if mv10.iloc[i] > mv20.iloc[i] else -1
        if i > 0 and not np.isnan(ma10_arr[i-1]):
            s5 = 1 if ma10_arr[i] > ma10_arr[i-1] else -1
        else:
            s5 = -1
        short_trend[i] = s1 + s2 + s3 + s4 + s5

    return pd.DataFrame({
        "date": dates, "close": c, "low": l, "high": h,
        "rs_hsi": rs_hsi, "slope200": slope200,
        "wk_j_swing": wk_j_swing, "wk_j_min_4w": wk_j_min_4w, "wk_j_max_4w": wk_j_max_4w,
        "weekly_div": weekly_div, "weekly_top_div": weekly_top_div,
        "weekly_demark_9_13": weekly_demark_9_13, "weekly_sell_demark": weekly_sell_demark,
        "daily_div_1": daily_div_1, "daily_top_div_1": daily_top_div_1,
        "climax_neg1": climax_neg1, "climax_pos1": climax_pos1,
        "sos": sos, "daily_demark": daily_demark, "daily_sell_demark": daily_sell_demark,
        "macd_cross_1": macd_cross == 1, "macd_cross_neg1": macd_cross == -1,
        "cross_510_1": cross_510 == 1, "cross_510_neg1": cross_510 == -1,
        "cross_1050_1": cross_1050 == 1, "cross_1050_neg1": cross_1050 == -1,
        "short_trend": short_trend,
    })


def _complete_weekly(daily):
    wk = resample_weekly(daily)
    if wk.empty:
        return wk
    return wk[wk["date"] <= daily["date"].max()].reset_index(drop=True)


# ============ 数据获取 ============
def fetch_data(pool, asof, cache_path):
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return {c: g.drop(columns=["code"]).reset_index(drop=True) for c, g in df.groupby("code")}
    codes = [c for c, _, _ in pool] + ["2800.HK"]
    raw = eodhd.fetch_all_eodhd(codes, asof, LOOKBACK_DAYS, workers=5)
    bag = {}
    for code, df in raw.items():
        if df is None or df.empty:
            continue
        fa = forward_adjust(df)
        if fa is not None and not fa.empty:
            bag[code] = fa
    frames = [df.assign(code=c) for c, df in bag.items()]
    pd.concat(frames, ignore_index=True).to_parquet(cache_path)
    return bag


# ============ 入场评估 (消融/主流程共用) ============
def evaluate_entries(df, mask, direction="long", label="", verbose=False, use_stops=True):
    """给定信号宽表 df 和 entry mask -> 30日去重 + forward return + 统计打印。返回 edf。
    use_stops=True: 8%固定+12%trailing止损; False: 纯持有到horizon(收盘对收盘)。
    供 run_backtest 和消融实验复用同一套止损数学。verbose=True 时打印按年分布。"""
    entries = df[mask].sort_values(["code", "date"]).reset_index(drop=True)
    entries = entries.drop_duplicates(subset=["code"], keep="first")
    final_entries = []
    last_entry = {}
    for _, r in entries.iterrows():
        code = r["code"]; d = r["date"]
        if code in last_entry and (d - last_entry[code]).days < 30:
            continue
        last_entry[code] = d
        final_entries.append(r)
    if not final_entries:
        print(f"[{label}] 无入场信号")
        return None

    edf = pd.DataFrame(final_entries)
    is_short = (direction == "short")
    for h in HORIZONS:
        edf[f"fwd{h}"] = np.nan
    edf["exit_reason"] = ""
    stock_cache = {}
    for ri, r in edf.iterrows():
        code = r["code"]; entry_date = r["date"]; entry_close = r["close"]
        if code not in stock_cache:
            stock_cache[code] = df[df["code"] == code].sort_values("date").reset_index(drop=True)
        stock_df = stock_cache[code]
        pos = stock_df.index[stock_df["date"] == entry_date]
        if len(pos) == 0 or entry_close <= 0:
            continue
        idx = pos[0]
        closes = stock_df["close"].values
        exit_day = None; exit_px = None; reason = ""
        if use_stops:
            lows = stock_df["low"].values
            highs = stock_df["high"].values
            if not is_short:
                fixed_px = entry_close * (1 - STOP_LOSS)
                peak = entry_close
                for k in range(idx + 1, len(stock_df)):
                    if not np.isnan(closes[k]) and closes[k] > peak:
                        peak = closes[k]
                    trail_px = peak * (1 - TRAIL_STOP)
                    if np.isnan(lows[k]):
                        continue
                    if lows[k] <= fixed_px or lows[k] <= trail_px:
                        exit_day = k; exit_px = max(fixed_px, trail_px)
                        reason = "fixed" if fixed_px >= trail_px else "trail"
                        break
            else:
                fixed_px = entry_close * (1 + STOP_LOSS)
                trough = entry_close
                for k in range(idx + 1, len(stock_df)):
                    if not np.isnan(closes[k]) and closes[k] < trough:
                        trough = closes[k]
                    trail_px = trough * (1 + TRAIL_STOP)
                    if np.isnan(highs[k]):
                        continue
                    if highs[k] >= fixed_px or highs[k] >= trail_px:
                        exit_day = k; exit_px = min(fixed_px, trail_px)
                        reason = "fixed" if fixed_px <= trail_px else "trail"
                        break
        if exit_day is not None:
            edf.at[ri, "exit_reason"] = reason
        for h in HORIZONS:
            j = idx + h
            if j >= len(stock_df):
                continue
            if exit_day is not None and exit_day <= j:
                ret = exit_px / entry_close - 1
            else:
                ret = closes[j] / entry_close - 1
            edf.at[ri, f"fwd{h}"] = -ret if is_short else ret

    if use_stops:
        stopped = edf["exit_reason"].replace("", "hold").value_counts()
        ext = "  退出: " + "  ".join(f"{k}={v}" for k, v in stopped.items())
    else:
        ext = "  (无止损, 收盘对收盘)"
    print(f"\n[{label}] 入场={len(edf)}{ext}")
    if verbose:
        edf["year"] = pd.to_datetime(edf["date"]).dt.year
        print(edf["year"].value_counts().sort_index().to_string())
    print(f"{'horizon':<10} {'avg':>8} {'median':>8} {'win_rate':>8} {'payoff':>8} {'expect':>8} {'avg_win':>8} {'avg_loss':>8} {'n':>6}")
    for h in HORIZONS:
        col = edf[f"fwd{h}"].dropna()
        if len(col) == 0:
            continue
        avg = col.mean(); med = col.median()
        wins = col[col > 0]; losses = col[col <= 0]
        wr = len(wins) / len(col)
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = losses.mean() if len(losses) else 0.0
        payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.nan
        expect = wr * avg_win + (1 - wr) * avg_loss
        print(f"fwd{h:<7} {avg:>8.3f} {med:>8.3f} {wr:>8.1%} {payoff:>8.2f} {expect:>8.3f} {avg_win:>8.3f} {avg_loss:>8.3f} {len(col):>6}")
    return edf


# ============ 主函数 ============
def compute_all_signals(pool, bag, bench, progress_every=50):
    """逐股计算信号 -> 合并宽表 (未加 universe/cond)。供 run_backtest 和消融复用。"""
    all_sigs = []
    t0 = time.time()
    for idx, (code, name, sector) in enumerate(pool):
        daily = bag.get(code)
        if daily is None or len(daily) < 260:
            continue
        try:
            sig = compute_signals(daily, bench)
            if sig is not None:
                sig["code"] = code
                all_sigs.append(sig)
        except Exception as e:
            print(f"  [ERR] {code}: {e}")
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"  [{idx+1}/{len(pool)}] {time.time()-t0:.0f}s")
    print(f"[Signals] {len(all_sigs)} 只完成, 耗时 {time.time()-t0:.0f}s")
    if not all_sigs:
        return None
    return pd.concat(all_sigs, ignore_index=True)


def run_backtest(pool, bag, bench, asof, mode, out_csv, label="建仓策略", direction="long", screen="bottom"):
    """强势/弱势股建仓回测的通用引擎 (港股/美股/A股共用)。
    mode: "strong" = RS或年线斜率在前25%; "weak" = 不在前25%。
    direction: "long"=做多; "short"=反向做空(对称止损)。
    screen: "bottom"=抄底信号(超卖/底背离/买DeMark/climax-1/金叉); "top"=见顶镜像(超买/顶背离/卖DeMark/climax+1/死叉)。"""
    df = compute_all_signals(pool, bag, bench)
    if df is None:
        print("无信号数据"); return
    df = build_conds(df, mode, screen)

    # 入场评估 (去重 + 止损 forward return + 统计)
    dir_tag = "做空(反向)" if direction == "short" else "做多"
    screen_tag = "见顶镜像" if screen == "top" else "抄底"
    print(f"\n{'='*60}")
    print(f"{label}回测 | {asof} | 3年 | mode={mode} | {screen_tag}{dir_tag}")
    print(f"{'='*60}")
    edf = evaluate_entries(df, df["entry"], direction=direction,
                           label=f"{label} | {screen_tag}{dir_tag}", verbose=True)
    if edf is not None:
        out_csv = Path(out_csv)
        edf.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[CSV] -> {out_csv}")


def build_conds(df, mode, screen="bottom"):
    """在信号宽表上设置 universe + cond_j/cond_reversal/cond_sos/cond_trend + entry。返回 df。"""
    # 截面排名: 每日75百分位
    df = df.copy()
    df["universe"] = False
    for d, grp in df.groupby("date"):
        rs_valid = grp["rs_hsi"].dropna()
        sl_valid = grp["slope200"].dropna()
        rs_p75 = rs_valid.quantile(0.75) if len(rs_valid) >= 20 else np.nan
        sl_p75 = sl_valid.quantile(0.75) if len(sl_valid) >= 20 else np.nan
        for idx in grp.index:
            rs = grp.loc[idx, "rs_hsi"]; sl = grp.loc[idx, "slope200"]
            in_rs = not np.isnan(rs) and not np.isnan(rs_p75) and rs > rs_p75
            in_sl = not np.isnan(sl) and not np.isnan(sl_p75) and sl > sl_p75
            in_top = in_rs or in_sl
            df.loc[idx, "universe"] = in_top if mode == "strong" else (not in_top)

    if screen != "top":
        df["cond_j"] = df["wk_j_min_4w"] <= 25
        df["cond_reversal"] = df["weekly_div"] | df["weekly_demark_9_13"] | df["daily_div_1"] | df["climax_neg1"]
        df["cond_sos"] = df["sos"]
        trend_count = (df["daily_demark"].astype(int) + df["macd_cross_1"].astype(int) +
                        df["cross_510_1"].astype(int) + df["cross_1050_1"].astype(int))
        df["cond_trend"] = trend_count >= 2
        df["entry"] = df["universe"] & df["cond_j"] & (df["short_trend"] >= 3) & (df["cond_sos"] | (df["cond_reversal"] & df["cond_trend"]))
    else:
        df["cond_j"] = df["wk_j_max_4w"] >= 75
        df["cond_reversal"] = df["weekly_top_div"] | df["weekly_sell_demark"] | df["daily_top_div_1"] | df["climax_pos1"]
        trend_count = (df["daily_sell_demark"].astype(int) + df["macd_cross_neg1"].astype(int) +
                        df["cross_510_neg1"].astype(int) + df["cross_1050_neg1"].astype(int))
        df["cond_trend"] = trend_count >= 2
        df["entry"] = df["universe"] & df["cond_j"] & (df["short_trend"] <= -3) & (df["cond_reversal"] & df["cond_trend"])
    return df


def main():
    ap = argparse.ArgumentParser(description="港股建仓策略回测 (强势/弱势)")
    ap.add_argument("--asof", default="2026-07-25")
    ap.add_argument("--mode", choices=["strong", "weak"], default="weak",
                    help="strong=RS或年线斜率前25%%; weak=不在前25%% (默认)")
    ap.add_argument("--inverse", action="store_true", help="反向做空: 同一信号对称止损")
    ap.add_argument("--top", action="store_true", help="筛选镜像: 见顶信号(超买/顶背离/死叉)代替抄底")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    out_dir = ROOT / cfg["output"]["dir"]; out_dir.mkdir(exist_ok=True)

    import provider as P
    pool = P.load_pool(cfg["data"]["pool_csv"])
    print(f"[Pool] {len(pool)} 只 | asof={args.asof} | mode={args.mode} | inverse={args.inverse} | top={args.top}")

    cache = out_dir / "backtest_entry_bag.parquet"
    bag = fetch_data(pool, args.asof, cache)
    bench = bag.get("2800.HK")
    print(f"[Data] {len(bag)} 只有数据 (含基准)")

    suffix = ("_top" if args.top else "") + ("_inverse" if args.inverse else "")
    run_backtest(pool, bag, bench, args.asof, args.mode,
                 out_dir / f"backtest_entry_{args.mode}{suffix}_results.csv",
                 label="港股建仓策略",
                 direction="short" if args.inverse else "long",
                 screen="top" if args.top else "bottom")


if __name__ == "__main__":
    main()
