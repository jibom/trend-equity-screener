"""Pattern 检测模块 (纯函数, 输入前复权日线 DataFrame)

包含:
  - 超卖/超买: 周J近期极值, RSI14
  - 背离: 周线/日线 MACD/KDJ/RSI 背离 (价格 vs 指标 极值对比)
  - DeMark: TD Sequential (setup 9 / countdown 13) + TD Combo countdown, 周/日 × 买/卖
  - K线形态: 十字星, 射击之星, climax top
  - 量: 极度缩量, 涨放量跌缩量
  - Crossover: 日KDJ/日MACD/5-10/10-50 金叉, 含预告(≤2日内将交叉)与刚交叉(≤2日内已交叉)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from kdj_divergence import calc_kdj, resample_weekly


def _complete_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """周线, 剔除不完整的当前周 (周五标签 > 最后交易日)。"""
    wk = resample_weekly(daily)
    if wk.empty:
        return wk
    last_daily = daily["date"].max()
    return wk[wk["date"] <= last_daily].reset_index(drop=True)


# ---------------- 指标 ----------------
def macd(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(close)
    dif = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    return dif.values, dea.values, (dif - dea).values


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder RSI, 前 n 根用 SMA 做种子 (与 TradingView ta.rsi 对齐)。返回与 close 等长(首根NaN)。"""
    close = np.asarray(close, dtype=float)
    m = len(close)
    out = np.full(m, np.nan)
    if m < n + 1:
        return out
    delta = np.diff(close)
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    avg_up = np.nan; avg_dn = np.nan
    for i in range(len(delta)):
        if i == n - 1:
            avg_up = up[:n].mean(); avg_dn = dn[:n].mean()
        elif i >= n:
            avg_up = (avg_up * (n - 1) + up[i]) / n
            avg_dn = (avg_dn * (n - 1) + dn[i]) / n
        if i >= n - 1 and avg_dn > 0:
            out[i + 1] = 100 - 100 / (1 + avg_up / avg_dn)
        elif i >= n - 1:
            out[i + 1] = 100.0
    return out


def ma(close: np.ndarray, w: int) -> np.ndarray:
    return pd.Series(close).rolling(w, min_periods=w).mean().values


# ---------------- 超卖/超买 ----------------
def _swing_extreme(vals: np.ndarray, lo: float, hi: float, win: int = 4):
    """swing 极值公用逻辑。vals=近期值数组(取最后 win 根)。
    有极值柱(<lo/>hi)取最近极值柱侧极值(翻转取最新侧); 无则按方向(涨取min/跌取max)。始终返回值。"""
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return None
    win_vals = vals[-win:]
    valid = win_vals[~np.isnan(win_vals)]
    if len(valid) == 0:
        return None
    ext_idx = [i for i, v in enumerate(win_vals) if not np.isnan(v) and (v < lo or v > hi)]
    if ext_idx:
        last = ext_idx[-1]
        return float(np.nanmax(win_vals)) if win_vals[last] > hi else float(np.nanmin(win_vals))
    cur = vals[-1]
    prev = vals[-2] if len(vals) >= 2 and not np.isnan(vals[-2]) else cur
    return float(np.nanmin(win_vals)) if cur >= prev else float(np.nanmax(win_vals))


def weekly_j_extreme(daily: pd.DataFrame) -> dict:
    """周J swing 极值 (近4周, <15/>95)。在完整周线上算 KDJ 再取末尾。"""
    wk = calc_kdj(_complete_weekly(daily))
    j = wk["j"].dropna()
    if len(j) == 0:
        return {"oversold": False, "overbought": False, "weekly_j": None, "swing_extreme": None}
    cur = float(wk["j"].iloc[-1])
    swing = _swing_extreme(wk["j"].values, 15.0, 95.0, 4)
    return {"oversold": cur < 10, "overbought": cur > 90,
            "weekly_j": round(cur, 2), "swing_extreme": round(swing, 2) if swing is not None else None}


def weekly_rsi_swing(daily: pd.DataFrame) -> float | None:
    """周RSI14 swing 极值 (近4周, <30/>70)。在完整周线上算 Wilder RSI 再取末尾。"""
    wk = _complete_weekly(daily)
    if len(wk) < 20:
        return None
    r = rsi(wk["fwd_close"].values, 14)
    swing = _swing_extreme(r, 30.0, 70.0, 4)
    return round(swing, 1) if swing is not None else None


def rsi_stats(daily: pd.DataFrame, lookback: int = 60) -> dict:
    r = rsi(daily["fwd_close"].values)
    cur = r[-1] if not np.isnan(r[-1]) else None
    seg = r[-lookback:]
    seg = seg[~np.isnan(seg)]
    return {"rsi": round(float(cur), 1) if cur is not None else None,
            "rsi_min": round(float(seg.min()), 1) if len(seg) else None,
            "rsi_max": round(float(seg.max()), 1) if len(seg) else None}


# ---------------- 背离 (通用: 价格 vs 指标) ----------------
def _divergence(prices: np.ndarray, ind: np.ndarray, order: int = 5,
                lookback: int = 60, recent: int = 15, min_pdiff: float = 0.02,
                min_spacing: int = 4, zone_lo: float | None = None,
                zone_hi: float | None = None, cancel_k: float = 6.0) -> str:
    """返回 '底背离' / '顶背离' / '' 。取 lookback 内最后两个同向极值比较。
    要求: 第二极值在近 recent 根内(时效); 价差>min_pdiff; 间距≥min_spacing;
    zone(可选): 底背离要求两极值指标<zone_lo, 顶背离要求>zone_hi。
    cancel_k: 波动率取消倍数。价格从第二极值沿背离方向走 > cancel_k×日收益标准差 视为已兑现→取消。
      (银行低波~10%到位, 高波股~50%到位, 自适应)。
    反向信号: 若底/顶背离都存在, 取第二极值更晚的那个(后来者取代)。"""
    n = len(prices)
    if n < 10:
        return ""
    s = min(lookback, n)
    p = prices[-s:]; iv = ind[-s:]
    # 波动率 (20日日收益标准差)
    rets = np.diff(np.log(np.where(p > 0, p, 1.0)))
    vol = float(np.std(rets[-20:])) if len(rets) >= 20 else (float(np.std(rets)) if len(rets) >= 2 else 0.02)
    cancel_thr = cancel_k * vol
    lows = argrelextrema(p, np.less_equal, order=order)[0]
    highs = argrelextrema(p, np.greater_equal, order=order)[0]
    if len(p) > 0 and p[-1] == p.min():
        if len(lows) == 0 or lows[-1] != len(p) - 1:
            lows = np.append(lows, len(p) - 1)
    if len(p) > 0 and p[-1] == p.max():
        if len(highs) == 0 or highs[-1] != len(p) - 1:
            highs = np.append(highs, len(p) - 1)
    # 检测底背离和顶背离, 各记录 (i2, 类型)
    bot = None; top = None
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if (i2 >= len(p) - recent and (i2 - i1) >= min_spacing
                and not np.isnan(iv[i1]) and not np.isnan(iv[i2])
                and p[i2] < p[i1] and iv[i2] > iv[i1]
                and abs(p[i1] - p[i2]) / max(p[i1], 1e-9) > min_pdiff
                and (zone_lo is None or (iv[i1] < zone_lo and iv[i2] < zone_lo))):
            if not (p[-1] > p[i2] * (1 + cancel_thr)):
                bot = (i2, "底背离")
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if (i2 >= len(p) - recent and (i2 - i1) >= min_spacing
                and not np.isnan(iv[i1]) and not np.isnan(iv[i2])
                and p[i2] > p[i1] and iv[i2] < iv[i1]
                and abs(p[i2] - p[i1]) / max(p[i1], 1e-9) > min_pdiff
                and (zone_hi is None or (iv[i1] > zone_hi and iv[i2] > zone_hi))):
            if not (p[-1] < p[i2] * (1 - cancel_thr)):
                top = (i2, "顶背离")
    # 反向信号: 第二极值更晚的取代
    if bot and top:
        return bot[1] if bot[0] >= top[0] else top[1]
    if bot:
        return bot[1]
    if top:
        return top[1]
    return ""

def all_divergences(daily: pd.DataFrame) -> dict:
    """周线/日线 × MACD/KDJ/RSI 背离。周KDJ用J且要求J<10(底)/J>90(顶)极值区。"""
    out = {}
    # 日线
    d = calc_kdj(daily.copy())
    dif, dea, _ = macd(d["fwd_close"].values)
    out["日KDJ背离"] = _divergence(d["fwd_close"].values, d["j"].values, recent=10, min_spacing=22)
    out["日MACD背离"] = _divergence(d["fwd_close"].values, dif, recent=10, min_spacing=22)
    out["日RSI背离"] = _divergence(d["fwd_close"].values, rsi(d["fwd_close"].values), recent=10, min_spacing=22)
    # 周线 (KDJ 用 J + 极值区 J<10/J>90)
    wk = calc_kdj(_complete_weekly(d))
    wdif, wdea, _ = macd(wk["fwd_close"].values)
    out["周KDJ背离"] = _divergence(wk["fwd_close"].values, wk["j"].values, lookback=40,
                                  zone_lo=10.0, zone_hi=90.0, recent=4)
    out["周MACD背离"] = _divergence(wk["fwd_close"].values, wdif, lookback=40, recent=4)
    out["周RSI背离"] = _divergence(wk["fwd_close"].values, rsi(wk["fwd_close"].values), lookback=40, recent=4)
    return out


# ---------------- DeMark TD Sequential + Combo ----------------
def _td_setup_counts(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (buy_setup_count, sell_setup_count) 逐根。setup=连续 close<close[4](买)/>close[4](卖)。"""
    n = len(close)
    buy = np.zeros(n, dtype=int); sell = np.zeros(n, dtype=int)
    for i in range(4, n):
        buy[i] = buy[i - 1] + 1 if close[i] < close[i - 4] else 0
        sell[i] = sell[i - 1] + 1 if close[i] > close[i - 4] else 0
    return buy, sell


def _td_sequential_countdown(close, high, low, buy_setup, sell_setup) -> dict:
    """Sequential countdown: 买setup完成后, 数 close<low[2] 至13; 卖: close>high[2] 至13。
    取消规则: 收盘突破 setup 极值则重置(买CD: close>买setup最高high; 卖CD: close<卖setup最低low)。
    反向/同向新 setup 完成也重置(recycling)。返回 current_count / complete / complete_idx(最新)。"""
    n = len(close)
    res = {"buy_cd": 0, "sell_cd": 0, "buy_cd_complete": False, "sell_cd_complete": False,
           "buy_cd_date": "", "sell_cd_date": ""}
    buy_setup_done = -1; sell_setup_done = -1
    buy_setup_high = np.inf; sell_setup_low = -np.inf
    bcd = 0; scd = 0
    bcd_complete_idx = -1; scd_complete_idx = -1
    prev_buy = 0; prev_sell = 0
    for i in range(n):
        # setup 完成检测 (count 从<9 升到 >=9), 记录 setup 极值, 重置反向 countdown
        if buy_setup[i] >= 9 and prev_buy < 9 and i >= 8:
            buy_setup_done = i
            buy_setup_high = float(np.max(high[i - 8:i + 1]))
            scd = 0
        if sell_setup[i] >= 9 and prev_sell < 9 and i >= 8:
            sell_setup_done = i
            sell_setup_low = float(np.min(low[i - 8:i + 1]))
            bcd = 0
        # 买 countdown (close<low[2]); 取消: close>买setup最高high
        if buy_setup_done >= 0 and i > buy_setup_done and i >= 2:
            if close[i] > buy_setup_high:
                bcd = 0
            elif bcd < 13 and close[i] < low[i - 2]:
                bcd += 1
                if bcd == 13:
                    bcd_complete_idx = i
        # 卖 countdown (close>high[2]); 取消: close<卖setup最低low
        if sell_setup_done >= 0 and i > sell_setup_done and i >= 2:
            if close[i] < sell_setup_low:
                scd = 0
            elif scd < 13 and close[i] > high[i - 2]:
                scd += 1
                if scd == 13:
                    scd_complete_idx = i
        prev_buy = buy_setup[i]; prev_sell = sell_setup[i]
    res["buy_cd"] = bcd
    res["sell_cd"] = scd
    res["buy_cd_complete"] = bcd_complete_idx >= 0
    res["sell_cd_complete"] = scd_complete_idx >= 0
    return res, bcd_complete_idx, scd_complete_idx


def _td_combo_countdown(close, high, low, buy_setup, sell_setup, dates) -> dict:
    """Combo countdown: 买setup完成后, 计数1=close<low[2]; 2-13 需 close<low[2] 且 low<前一计数根low。
    卖对称。"""
    n = len(close)
    res = {"buy_cd": 0, "sell_cd": 0, "buy_cd_complete": False, "sell_cd_complete": False}
    buy_setup_done = -1; sell_setup_done = -1
    bcd = 0; scd = 0; prev_buy_low = np.inf; prev_sell_high = -np.inf
    bcd_done = False; scd_done = False
    for i in range(n):
        if buy_setup[i] >= 9 and buy_setup_done < 0:
            buy_setup_done = i
        if sell_setup[i] >= 9 and sell_setup_done < 0:
            sell_setup_done = i
        if sell_setup[i] >= 9:
            bcd = 0; prev_buy_low = np.inf
        if buy_setup[i] >= 9:
            scd = 0; prev_sell_high = -np.inf
        if buy_setup_done >= 0 and i > buy_setup_done and i >= 2 and not bcd_done:
            if close[i] < low[i - 2]:
                if bcd == 0 or low[i] < prev_buy_low:
                    bcd += 1; prev_buy_low = low[i]
                    if bcd >= 13:
                        bcd_done = True
        if sell_setup_done >= 0 and i > sell_setup_done and i >= 2 and not scd_done:
            if close[i] > high[i - 2]:
                if scd == 0 or high[i] > prev_sell_high:
                    scd += 1; prev_sell_high = high[i]
                    if scd >= 13:
                        scd_done = True
    res["buy_cd"] = bcd; res["sell_cd"] = scd
    res["buy_cd_complete"] = bcd_done; res["sell_cd_complete"] = scd_done
    return res


def td_sequential(close, high, low, dates) -> dict:
    buy, sell = _td_setup_counts(close)
    cd, bidx, sidx = _td_sequential_countdown(close, high, low, buy, sell)
    n = len(close)
    # setup-9 完成时点 (最新一次 count==9 的 index)
    buy9_idx = np.where(buy == 9)[0]
    sell9_idx = np.where(sell == 9)[0]
    buy9_idx = int(buy9_idx[-1]) if len(buy9_idx) > 0 else -1
    sell9_idx = int(sell9_idx[-1]) if len(sell9_idx) > 0 else -1
    out = {
        "buy_setup": int(buy[-1]), "sell_setup": int(sell[-1]),
        "buy_cd": cd["buy_cd"], "sell_cd": cd["sell_cd"],
        "buy_cd_complete": cd["buy_cd_complete"], "sell_cd_complete": cd["sell_cd_complete"],
        "buy_cd_bars_ago": (n - 1 - bidx) if bidx >= 0 else None,
        "sell_cd_bars_ago": (n - 1 - sidx) if sidx >= 0 else None,
        "buy9_bars_ago": (n - 1 - buy9_idx) if buy9_idx >= 0 else None,
        "sell9_bars_ago": (n - 1 - sell9_idx) if sell9_idx >= 0 else None,
    }
    return out


def td_combo(close, high, low, dates) -> dict:
    buy, sell = _td_setup_counts(close)
    cd = _td_combo_countdown(close, high, low, buy, sell, dates)
    return {
        "buy_setup": int(buy[-1]), "sell_setup": int(sell[-1]),
        "buy_cd": cd["buy_cd"], "sell_cd": cd["sell_cd"],
        "buy_cd_complete": cd["buy_cd_complete"], "sell_cd_complete": cd["sell_cd_complete"],
    }


def demark_all(daily: pd.DataFrame) -> dict:
    """DeMark TD Sequential 精简: 周/日 × 买9/买13/卖9/卖13 (8 个 flag)。
    9=setup完成近8日/4周内; 13=countdown完成(周近4周/日近8日内)。"""
    out = {}
    dd = daily.reset_index(drop=True)
    c = dd["fwd_close"].values; h = dd["fwd_high"].values; l = dd["fwd_low"].values
    dt = dd["date"].dt.strftime("%Y-%m-%d").values
    seq_d = td_sequential(c, h, l, dt)
    out["日买9"] = bool(seq_d["buy_setup"] >= 9 or (seq_d["buy9_bars_ago"] is not None and seq_d["buy9_bars_ago"] <= 8))
    out["日卖9"] = bool(seq_d["sell_setup"] >= 9 or (seq_d["sell9_bars_ago"] is not None and seq_d["sell9_bars_ago"] <= 8))
    out["日买13"] = bool(seq_d["buy_cd_complete"] and seq_d["buy_cd_bars_ago"] is not None
                         and seq_d["buy_cd_bars_ago"] <= 8)
    out["日卖13"] = bool(seq_d["sell_cd_complete"] and seq_d["sell_cd_bars_ago"] is not None
                         and seq_d["sell_cd_bars_ago"] <= 8)
    # 周线
    wk = _complete_weekly(daily).reset_index(drop=True)
    if len(wk) >= 20:
        c = wk["fwd_close"].values; h = wk["fwd_high"].values; l = wk["fwd_low"].values
        dt = wk["date"].dt.strftime("%Y-%m-%d").values
        seq_w = td_sequential(c, h, l, dt)
        out["周买9"] = bool(seq_w["buy_setup"] >= 9 or (seq_w["buy9_bars_ago"] is not None and seq_w["buy9_bars_ago"] <= 4))
        out["周卖9"] = bool(seq_w["sell_setup"] >= 9 or (seq_w["sell9_bars_ago"] is not None and seq_w["sell9_bars_ago"] <= 4))
        out["周买13"] = bool(seq_w["buy_cd_complete"] and seq_w["buy_cd_bars_ago"] is not None
                             and seq_w["buy_cd_bars_ago"] <= 4)
        out["周卖13"] = bool(seq_w["sell_cd_complete"] and seq_w["sell_cd_bars_ago"] is not None
                             and seq_w["sell_cd_bars_ago"] <= 4)
    else:
        for k in ("周买9", "周卖9", "周买13", "周卖13"):
            out[k] = False
    return out


def demark_col(dm: dict, pfx: str):
    """DeMark 单格: 正数=买方(9 setup/13 countdown), 负数=卖方; 双向同时出现用 '+13/-9' 文本。空=无。"""
    buy = 13 if dm.get(pfx + "买13") else (9 if dm.get(pfx + "买9") else None)
    sell = -13 if dm.get(pfx + "卖13") else (-9 if dm.get(pfx + "卖9") else None)
    if buy is not None and sell is not None:
        return f"+{buy}/{sell}"
    if buy is not None:
        return buy
    if sell is not None:
        return sell
    return ""


# ---------------- K线形态 + 量 ----------------
def candlestick_patterns(daily: pd.DataFrame, lookback: int = 10, doji_lookback: int = 3) -> dict:
    d = daily.tail(lookback).reset_index(drop=True)
    o = d["fwd_open"].values; c = d["fwd_close"].values
    h = d["fwd_high"].values; l = d["fwd_low"].values
    body = np.abs(c - o)
    # doji (只看近 doji_lookback 日)
    dd = daily.tail(doji_lookback).reset_index(drop=True)
    ob = np.abs(dd["fwd_close"].values - dd["fwd_open"].values)
    cb = dd["fwd_close"].values; rngb = dd["fwd_high"].values - dd["fwd_low"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        b2r = np.where(rngb > 0, ob / rngb, 1.0)
        r2o = np.where(ob + cb > 0, rngb / np.where(dd["fwd_open"].values > 0, dd["fwd_open"].values, np.nan), 0.0)
    doji_b = ((b2r <= 0.10) & (r2o >= 0.005)) | (ob <= np.where(cb >= 5, 0.02, 0.01))
    doji_count = int(doji_b.sum())
    # climax: 大实体(>3%开盘) + 放量(>2×20日量均). top=大阳, bottom=大阴
    vol = d["volume"].values
    vol_ma20 = pd.Series(daily["volume"]).rolling(20, min_periods=5).mean().iloc[-1]
    vr = vol / vol_ma20 if vol_ma20 > 0 else np.zeros(len(vol))
    big_body = body / np.where(o > 0, o, np.nan) > 0.03
    climax_top = bool(((c > o) & big_body & (vr > 2.0)).any())
    climax_bot = bool(((c < o) & big_body & (vr > 2.0)).any())
    return {"doji_count": doji_count, "climax_top": climax_top, "climax_bot": climax_bot}


def volume_patterns(daily: pd.DataFrame, lookback: int = 10) -> dict:
    """涨放量跌缩量: 近10日 ≥2上涨日 且 ≥1下跌日; 上涨日均量≥2×下跌日均量;
    下跌日量持续<20日量均(每个下跌日均低于当日20日量均)。"""
    d = daily.tail(lookback).reset_index(drop=True)
    vol = d["volume"].values
    pct = pd.Series(d["fwd_close"]).pct_change().values
    vma20 = pd.Series(daily["volume"]).rolling(20, min_periods=5).mean().values[-lookback:]
    up_mask = pct > 0; dn_mask = pct < 0
    n_up = int(up_mask.sum()); n_dn = int(dn_mask.sum())
    up_vol = vol[up_mask]; dn_vol = vol[dn_mask]
    ok = False
    if n_up >= 2 and n_dn >= 1:
        up_avg = float(up_vol.mean()); dn_avg = float(dn_vol.mean())
        dn_vma = vma20[dn_mask]
        ok = (up_avg >= 1.5 * dn_avg) and bool((dn_vol < dn_vma).all())
    return {"up_vol_dn_shrink": ok}


# ---------------- Crossover (含预告) ----------------
def _cross_status(fast: np.ndarray, slow: np.ndarray, dates, forecast: bool = True,
                  forecast_days: int = 2, forecast_guard: bool = False) -> dict:
    """fast 上穿 slow 的状态。返回 {state, date, days_ago}。
    state: '金叉≤2日' / '金叉Nd前' / '预告金叉≤Nd日' / '死叉...' / '无'。
    forecast_days: 预告窗口(默认2日)。
    forecast_guard: 预告需过去3根单边且加速(MACD专用)。"""
    n = len(fast)
    state = "无"; date = ""; days = None
    if n < 3 or np.isnan(fast[-1]) or np.isnan(slow[-1]):
        return {"state": "无", "date": "", "days_ago": None}
    gap = fast - slow
    # 找最近一次上穿
    last_up = -1
    for i in range(1, n):
        if np.isnan(gap[i]) or np.isnan(gap[i - 1]):
            continue
        if gap[i - 1] <= 0 < gap[i]:
            last_up = i
    # 找最近一次下穿
    last_dn = -1
    for i in range(1, n):
        if np.isnan(gap[i]) or np.isnan(gap[i - 1]):
            continue
        if gap[i - 1] >= 0 > gap[i]:
            last_dn = i
    if last_up > last_dn and last_up >= 0:
        days = n - 1 - last_up
        date = str(dates[last_up])[:10]
        state = "金叉≤2日" if days <= 2 else f"金叉{days}d前"
    elif last_dn > last_up and last_dn >= 0:
        days = n - 1 - last_dn
        date = str(dates[last_dn])[:10]
        state = "死叉≤2日" if days <= 2 else f"死叉{days}d前"
    # 预告: 当前 gap<0 但斜率正向, 预计≤forecast_days 日上穿
    if forecast and gap[-1] < 0 and n >= 2 and not np.isnan(gap[-2]):
        rate = gap[-1] - gap[-2]
        if rate > 0:
            btx = -gap[-1] / rate
            if 0 < btx <= forecast_days and (not forecast_guard or _mono_accel(gap, "up")):
                state = f"预告金叉≤{forecast_days}日"
    elif forecast and gap[-1] > 0 and n >= 2 and not np.isnan(gap[-2]):
        rate = gap[-1] - gap[-2]
        if rate < 0:
            btx = gap[-1] / -rate
            if 0 < btx <= forecast_days and (not forecast_guard or _mono_accel(gap, "down")):
                state = f"预告死叉≤{forecast_days}日"
    return {"state": state, "date": date, "days_ago": days}


def _mono_accel(gap: np.ndarray, direction: str) -> bool:
    """过去3根 gap 是否单边且加速。direction='up'(依次上行加速) / 'down'(依次下行加速)。"""
    if len(gap) < 3:
        return False
    a, b, c = gap[-3], gap[-2], gap[-1]
    if np.isnan(a) or np.isnan(b) or np.isnan(c):
        return False
    if direction == "up":
        return a < b < c and (c - b) > (b - a)   # 依次上行 且 加速
    return a > b > c and (b - c) > (a - b)       # 依次下行 且 加速


def cross_code(state: str):
    """金叉状态 → 1/-1/预1/预-1; 非近2日/非预告 → 空。预告窗口可≤2或≤3日(MACD)。"""
    if state.startswith("预告金叉"): return "预1"
    if state.startswith("预告死叉"): return "预-1"
    if state == "金叉≤2日": return 1
    if state == "死叉≤2日": return -1
    return ""


def cross_5_10_stable(daily: pd.DataFrame, entangle_thresh: float = 0.002,
                      narrow_window: int = 5, cross_lookback: int = 2):
    """5/10 均线穿过信号 (稳定穿过版, 替代频繁的预告式 cross):
    1) 稳定穿过: gap(MA5-MA10) 近 cross_lookback 日内过零, 且过零前 narrow_window 日 |gap| 持续收窄
       (允许 1 次反复) → 1(金叉)/-1(死叉)
    2) 有过零但不稳定 (whipsaw/暴涨驱动等脏交叉) → 0 (纠缠/无方向)
    3) 无近期过零但两线持续重叠 (近 narrow_window 日 rel gap 均<entangle_thresh) → 0
    4) 其余 (无过零且分离) → 空。"""
    c = daily["fwd_close"].values
    ma5 = pd.Series(c).rolling(5, min_periods=5).mean().values
    ma10 = pd.Series(c).rolling(10, min_periods=10).mean().values
    n = len(c)
    if n < 15 or np.isnan(ma5[-1]) or np.isnan(ma10[-1]):
        return ""
    gap = ma5 - ma10
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.abs(gap) / np.where(ma5 > 0, ma5, np.nan)
    # 找近 cross_lookback 日内最近一次过零
    cross_idx = -1
    for i in range(n - 1, max(n - 1 - cross_lookback, 4), -1):
        if np.isnan(gap[i]) or np.isnan(gap[i - 1]):
            continue
        if (gap[i - 1] <= 0 < gap[i]) or (gap[i - 1] >= 0 > gap[i]):
            cross_idx = i; break
    if cross_idx >= 0:
        # 过零前 narrow_window 日 |gap| 持续收窄 (允许 1 次增大=反复)
        lo = max(0, cross_idx - narrow_window)
        seg = np.abs(gap[lo:cross_idx])
        if len(seg) >= 3:
            increases = int(np.sum(np.diff(seg) > 0))
            if increases <= 1:
                return 1 if gap[cross_idx] > 0 else -1  # 稳定穿过
        return 0  # 有过零但不稳定 → 纠缠/无方向
    # 无近期过零: 两线持续重叠 → 0
    last_rel = rel[-narrow_window:]
    if not np.isnan(last_rel).any() and np.all(last_rel < entangle_thresh):
        return 0
    return ""  # 无过零且分离 → 空


def crossovers(daily: pd.DataFrame) -> dict:
    d = calc_kdj(daily.copy()).reset_index(drop=True)
    dates = d["date"].dt.strftime("%Y-%m-%d").values
    out = {}
    c = d["fwd_close"].values
    dif, dea, _ = macd(c)
    ma5 = ma(c, 5); ma10 = ma(c, 10); ma50 = ma(c, 50)
    out["KDJ金叉"] = _cross_status(d["k"].values, d["d"].values, dates, forecast=True)["state"]
    out["MACD金叉"] = _cross_status(dif, dea, dates, forecast=True, forecast_days=3, forecast_guard=True)["state"]
    # 5/10 Cross: 仅在 close >= MA10 时报信号 (股价站上10日线才有意义)
    s510 = _cross_status(ma5, ma10, dates, forecast=True)["state"]
    if np.isnan(ma10[-1]) or c[-1] < ma10[-1]:
        s510 = ""
    out["5_10金叉"] = s510
    # 10/50 Cross: 仅在 close >= MA50 时报信号 (股价站上50日线才有意义)
    s1050 = _cross_status(ma10, ma50, dates, forecast=True)["state"]
    if np.isnan(ma50[-1]) or c[-1] < ma50[-1]:
        s1050 = ""
    out["10_50金叉"] = s1050
    return out


# ---------------- SOS (Sign of Strength, Wyckoff) ----------------
def detect_sos(daily: pd.DataFrame, lookback: int = 3, pos_n: int = 126,
               pos_n_fallback: int = 63,
               pos_max: float = 0.50, range_mult: float = 1.5, vol_mult: float = 1.5,
               close_ratio: float = 0.70, bull_body_pct: float = 0.03,
               entangle_thresh: float = 0.05, entangle_lookback: int = 3) -> int:
    """Sign of Strength (Wyckoff): 位置门 = A OR B, 强度共享放量。
    A: pos_n(默认126=6个月)区间 pos<=pos_max; 6个月数据不足时回退 pos_n_fallback(默认63=3个月)
    B: 近 entangle_lookback 日 4均线(5/10/15/20)纠缠 max/min-1<entangle_thresh  (从均线纠缠平台启动)
    阳线path: 中大阳(实体>3%) + 波幅扩张(1.5×) + 收盘靠高(0.70) + 放量 + (A或B)
    十字星path: 十字星 + 放量 + (A或B) (无波幅/收盘要求)
    近 lookback 日内任一根满足任一路径 → 返回 1, 否则 0。
    pos 仍NaN(<fallback日)时只看B; B只需20日历史, 兼容次新股。"""
    need = max(pos_n + 10, lookback + 65, 65)
    d = daily.tail(need).reset_index(drop=True)
    if len(d) < 65:
        return 0
    o = d["fwd_open"].values; c = d["fwd_close"].values
    h = d["fwd_high"].values; l = d["fwd_low"].values; v = d["volume"].values
    n = len(d)
    rng = h - l; body = c - o
    # A: pos 6个月, 不足回退3个月
    rmin1 = pd.Series(c).rolling(pos_n, min_periods=pos_n).min().values
    rmax1 = pd.Series(c).rolling(pos_n, min_periods=pos_n).max().values
    pos1 = (c - rmin1) / np.where(rmax1 > rmin1, rmax1 - rmin1, np.nan)
    rmin2 = pd.Series(c).rolling(pos_n_fallback, min_periods=pos_n_fallback).min().values
    rmax2 = pd.Series(c).rolling(pos_n_fallback, min_periods=pos_n_fallback).max().values
    pos2 = (c - rmin2) / np.where(rmax2 > rmin2, rmax2 - rmin2, np.nan)
    pos = np.where(np.isnan(pos1), pos2, pos1)
    avg_rng = pd.Series(rng).rolling(10, min_periods=5).mean().values
    vol_ma = pd.Series(v).rolling(20, min_periods=5).mean().values
    # B: 4均线(5/10/15/20)纠缠逐日
    mas = np.column_stack([pd.Series(c).rolling(w, min_periods=w).mean().values for w in (5, 10, 15, 20)])
    ma_valid = ~np.isnan(mas).any(axis=1)
    ma_max = np.full(n, np.nan); ma_min = np.full(n, np.nan)
    if ma_valid.any():
        ma_max[ma_valid] = np.nanmax(mas[ma_valid], axis=1)
        ma_min[ma_valid] = np.nanmin(mas[ma_valid], axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        spread = ma_max / np.where(ma_min > 0, ma_min, np.nan) - 1
    entangled = (spread < entangle_thresh) & ma_valid
    for i in range(n - lookback, n):
        if i < 25 or np.isnan(avg_rng[i]) or np.isnan(vol_ma[i]) or rng[i] <= 0:
            continue
        # 位置门 A OR B (pos NaN时只看B)
        pos_ok = (not np.isnan(pos[i])) and pos[i] <= pos_max
        lo = max(0, i - entangle_lookback + 1)
        ent_ok = bool(entangled[lo:i + 1].any())
        if not (pos_ok or ent_ok):
            continue
        # 放量(共享)
        if v[i] < vol_mult * vol_ma[i]:
            continue
        # 十字星判定
        abs_body = abs(body[i])
        b2r = abs_body / rng[i]
        r2o = rng[i] / o[i] if o[i] > 0 else 0.0
        is_doji = (b2r <= 0.10 and r2o >= 0.005) or (abs_body <= (0.02 if c[i] >= 5 else 0.01))
        # 阳线 path: 中大阳(实体>3%) + 波幅扩张 + 收盘靠高
        if body[i] > 0 and o[i] > 0 and body[i] / o[i] >= bull_body_pct:
            if rng[i] < range_mult * avg_rng[i]: continue
            if (c[i] - l[i]) / rng[i] < close_ratio: continue
            return 1
        # 十字星 path: 已满足 放量+位置门 → SOS
        if is_doji:
            return 1
    return 0


# ---------------- SOS 原版 (60日pos, 无均线纠缠, 无3月回退) ----------------
def detect_sos_orig(daily: pd.DataFrame, lookback: int = 3, pos_n: int = 60,
                    pos_max: float = 0.50, range_mult: float = 1.5, vol_mult: float = 1.5,
                    close_ratio: float = 0.70, bull_body_pct: float = 0.03) -> int:
    """原版 Sign of Strength: 60日区间 pos<=0.50 + 放量 + (阳线body>=3%&range>=1.5x&close_pos>=0.70 或 十字星)。
    无均线纠缠OR、无6月->3月回退。近 lookback 日任一根满足任一路径 → 1, 否则 0。"""
    need = max(pos_n + 10, lookback + 35)
    d = daily.tail(need).reset_index(drop=True)
    if len(d) < 65:
        return 0
    o = d["fwd_open"].values; c = d["fwd_close"].values
    h = d["fwd_high"].values; l = d["fwd_low"].values; v = d["volume"].values
    n = len(d)
    rng = h - l; body = c - o
    rmin = pd.Series(c).rolling(pos_n, min_periods=pos_n).min().values
    rmax = pd.Series(c).rolling(pos_n, min_periods=pos_n).max().values
    pos = (c - rmin) / np.where(rmax > rmin, rmax - rmin, np.nan)
    avg_rng = pd.Series(rng).rolling(10, min_periods=5).mean().values
    vol_ma = pd.Series(v).rolling(20, min_periods=5).mean().values
    for i in range(n - lookback, n):
        if i < 25 or np.isnan(avg_rng[i]) or np.isnan(vol_ma[i]) or np.isnan(pos[i]) or rng[i] <= 0:
            continue
        if pos[i] > pos_max:
            continue
        if v[i] < vol_mult * vol_ma[i]:
            continue
        abs_body = abs(body[i])
        b2r = abs_body / rng[i]
        r2o = rng[i] / o[i] if o[i] > 0 else 0.0
        is_doji = (b2r <= 0.10 and r2o >= 0.005) or (abs_body <= (0.02 if c[i] >= 5 else 0.01))
        if body[i] > 0 and o[i] > 0 and body[i] / o[i] >= bull_body_pct:
            if rng[i] < range_mult * avg_rng[i]: continue
            if (c[i] - l[i]) / rng[i] < close_ratio: continue
            return 1
        if is_doji:
            return 1
    return 0