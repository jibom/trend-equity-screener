"""SOS (Sign of Strength, Wyckoff) 检测 — 从 hk_div_doji_chip/src/patterns.py 移植 (2026-08)。

用于趋势五部曲的 Part3: 替代原"异动放量"(单日>4%+量 或 3日>10%+量),
改为 swing-pattern 的 SOS 信号: 低位/均线纠缠平台启动的放量中大阳 (实体>3% + 波幅扩张1.5x + 收盘靠高 + 量>1.5x)。

与旧 src/sos.py (Wyckoff v5.3 classify_sos) 无关, 勿混淆。
需要 daily DataFrame 列: fwd_open, fwd_high, fwd_low, fwd_close, volume。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def detect_sos(daily: pd.DataFrame, lookback: int = 3, pos_n: int = 200,
               pos_n_fallback: int = 63,
               pos_max: float = 0.80, range_mult: float = 1.5, vol_mult: float = 1.5,
               close_ratio: float = 0.70, bull_body_pct: float = 0.03,
               entangle_thresh: float = 0.05, entangle_lookback: int = 3,
               skip_latest: bool = False) -> int:
    """Sign of Strength (Wyckoff): 位置门 = A OR B, 强度共享放量。
    A: pos_n(默认200日)区间 pos<=pos_max; 200日数据不足时回退 pos_n_fallback(默认63=3个月)
    B: 近 entangle_lookback 日 4均线(5/10/15/20)纠缠 max/min-1<entangle_thresh  (从均线纠缠平台启动)
    阳线path: 中大阳(实体>3%) + 波幅扩张(1.5×) + 收盘靠高(0.70) + 放量 + (A或B)
    扫描窗口: 末尾 lookback 根。skip_latest=True 时跳过最新一根, 扫它之前的 lookback 根
    (用于 "SOS 过去3天": lookback=3, skip_latest=True → 最新根前3根)。
    任一根满足阳线路径 → 返回 1, 否则 0。
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
    end = n - 1 if skip_latest else n          # skip_latest: 跳过最新一根, 扫之前 lookback 根
    for i in range(max(end - lookback, 0), end):
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
        # 阳线 path: 中大阳(实体>3%) + 波幅扩张 + 收盘靠高
        if body[i] > 0 and o[i] > 0 and body[i] / o[i] >= bull_body_pct:
            if rng[i] < range_mult * avg_rng[i]: continue
            if (c[i] - l[i]) / rng[i] < close_ratio: continue
            return 1
    return 0


def sos_flags(gg: pd.DataFrame) -> tuple[int, int]:
    """对前复权后的个股 df 返回 (SOS最新一根, SOS过去3天)。gg 需含 fwd_* 与 vol 列。"""
    daily = pd.DataFrame({
        "fwd_open": gg["fwd_open"].values, "fwd_high": gg["fwd_high"].values,
        "fwd_low": gg["fwd_low"].values, "fwd_close": gg["fwd_close"].values,
        "volume": gg["vol"].values,
    })
    sos_today = detect_sos(daily, lookback=1)
    sos_3d = detect_sos(daily, lookback=3, skip_latest=True)
    return sos_today, sos_3d
