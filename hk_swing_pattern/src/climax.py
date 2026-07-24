"""Climax 信号: 极端价格变动(ATR倍数 或 历史百分位) + 放量 + 极端位置(低/高)

climax top (+1): 放量大涨在高位(最后一涨)
climax bottom (-1): 放量大跌在低位(最后一跌)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def atr_series(daily: pd.DataFrame, n: int = 14):
    h = daily["fwd_high"].values; l = daily["fwd_low"].values; c = daily["fwd_close"].values
    m = len(h)
    tr = np.zeros(m)
    tr[0] = h[0] - l[0]
    for i in range(1, m):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr_v = pd.Series(tr).rolling(n, min_periods=n).mean().values
    return atr_v, tr


def climax_flags(daily: pd.DataFrame, k_atr: float = 3.0, v_mult: float = 2.0,
                 pos_lo: float = 0.20, pos_hi: float = 0.80, pct_lookback: int = 250,
                 pct: float = 0.995, atr_n: int = 14, vol_n: int = 20, pos_n: int = 60) -> pd.DataFrame:
    """返回逐日 DataFrame: date/close/flag(+1/-1/0) 及 move_mult/ret_extreme/vol_mult/pos 指标。"""
    d = daily.reset_index(drop=True)
    c = d["fwd_close"].values; v = d["volume"].values
    prev_c = np.concatenate([[np.nan], c[:-1]])
    delta = np.abs(c - prev_c)                       # |close - prevclose|
    atr_v, _ = atr_series(d, atr_n)
    move_mult = delta / np.where(atr_v > 0, atr_v, np.nan)   # 净变动 / ATR
    ret_abs = np.abs(pd.Series(c).pct_change().values)
    # |收益率| 是否达近 pct_lookback 日 99.5 分位 (用 quantile 比 rank 快)
    ret_q = pd.Series(ret_abs).rolling(pct_lookback, min_periods=60).quantile(pct).values
    ret_extreme = ret_abs >= ret_q
    vol_ma = pd.Series(v).rolling(vol_n, min_periods=5).mean().values
    vol_mult = v / np.where(vol_ma > 0, vol_ma, np.nan)
    roll_min = pd.Series(c).rolling(pos_n, min_periods=pos_n).min().values
    roll_max = pd.Series(c).rolling(pos_n, min_periods=pos_n).max().values
    pos = (c - roll_min) / np.where(roll_max > roll_min, roll_max - roll_min, np.nan)
    up = c > prev_c
    extreme = (move_mult >= k_atr) | ret_extreme
    high_vol = vol_mult >= v_mult
    top = extreme & high_vol & up & (pos >= pos_hi)
    bot = extreme & high_vol & (~up) & (pos <= pos_lo)
    flag = np.where(top, 1, np.where(bot, -1, 0))
    return pd.DataFrame({"date": d["date"].values, "close": c, "flag": flag,
                         "move_mult": move_mult, "ret_extreme": ret_extreme,
                         "vol_mult": vol_mult, "pos": pos})


def events_with_fwd(daily: pd.DataFrame, horizons=(5, 10, 20, 60), **kw) -> pd.DataFrame:
    """返回每个 climax 事件 + 各 horizon forward return。"""
    d = daily.reset_index(drop=True)
    c = d["fwd_close"].values
    flags = climax_flags(daily, **kw)
    ev = flags[flags["flag"] != 0].reset_index(drop=True)
    date_to_i = {pd.Timestamp(d["date"].iloc[i]): i for i in range(len(d))}
    rows = []
    for _, r in ev.iterrows():
        i = date_to_i.get(pd.Timestamp(r["date"]))
        if i is None or i >= len(c):
            continue
        entry = c[i]
        row = {"date": pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
               "type": int(r["flag"]),
               "move_mult": round(float(r["move_mult"]), 2) if not np.isnan(r["move_mult"]) else None,
               "vol_mult": round(float(r["vol_mult"]), 2) if not np.isnan(r["vol_mult"]) else None,
               "pos": round(float(r["pos"]), 2) if not np.isnan(r["pos"]) else None}
        for h in horizons:
            j = i + h
            row[f"fwd{h}"] = (c[j] / entry - 1) if (j < len(c) and entry > 0) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
