"""
KDJ 指标计算 & 背离检测 (基础版, 供 pullback_buypoint 使用)

注意: 本项目另有 src/kdj_divergence.py (ATR 版, 用于 scripts/scan_kdj_*.py), API 不同;
      本文件为 sector_cluster/pullback_buypoint 依赖的 calc_kdj/detect_divergence 版本,
      独立存放以避免冲突。

KDJ 计算:
  RSV = (C - Ln) / (Hn - Ln) * 100      n=9 (默认)
  K   = 2/3 * prev_K + 1/3 * RSV         m1=3
  D   = 2/3 * prev_D + 1/3 * K           m2=3
  J   = 3*K - 2*D

背离检测:
  底背离 (bullish): 价格创新低, KDJ 的 K/D 未创新低 → 看涨信号
  顶背离 (bearish): 价格创新高, KDJ 的 K/D 未创新高 → 看跌信号

同时支持日线和周线级别。
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def calc_kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3,
             close_col: str = "fwd_close", high_col: str = "fwd_high",
             low_col: str = "fwd_low") -> pd.DataFrame:
    """
    计算 KDJ 指标, 在 df 上追加列: rsv, k, d, j
    返回同长度 DataFrame (前 n-1 行 K/D/J 为 NaN)
    """
    df = df.copy()
    low_n = df[low_col].rolling(n, min_periods=n).min()
    high_n = df[high_col].rolling(n, min_periods=n).max()
    denom = high_n - low_n
    rsv = np.where(denom == 0, 50.0, (df[close_col] - low_n) / denom * 100)
    df["rsv"] = rsv

    k = np.full(len(df), np.nan)
    d = np.full(len(df), np.nan)
    k[0] = 50.0
    d[0] = 50.0
    for i in range(1, len(df)):
        if pd.isna(rsv[i]):
            k[i] = 50.0
            d[i] = 50.0
        else:
            k[i] = (m1 - 1) / m1 * k[i - 1] + 1 / m1 * rsv[i]
            d[i] = (m2 - 1) / m2 * d[i - 1] + 1 / m2 * k[i]

    df["k"] = k
    df["d"] = d
    df["j"] = 3 * k - 2 * d
    return df


def resample_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """日线 → 周线 (周五对齐)"""
    if df.empty or "date" not in df.columns:
        return df
    w = (df.set_index("date")
          .resample("W-FRI")
          .agg({"fwd_open": "first", "fwd_high": "max", "fwd_low": "min",
                "fwd_close": "last", "volume": "sum", "raw_close": "last"})
          .dropna())
    return w.reset_index()


def detect_divergence(df: pd.DataFrame, lookback: int = 60,
                      k_col: str = "k", d_col: str = "d",
                      close_col: str = "fwd_close") -> dict:
    """
    检测最近 lookback 根 K 线内的背离

    返回:
      {
        "daily_kdj_k": float, "daily_kdj_d": float, "daily_kdj_j": float,
        "daily_divergence": str,    # "底背离" / "顶背离" / ""
        "weekly_kdj_k": float, "weekly_kdj_d": float, "weekly_kdj_j": float,
        "weekly_divergence": str,
      }

    背离判定逻辑:
      在最近 lookback 根 K 线中找两个极值点:
      - 底背离: 价格第二低点 < 第一低点 (创新低), 但 K 的第二低点 > 第一低点 (未创新低)
      - 顶背离: 价格第二高点 > 第一高点 (创新高), 但 K 的第二高点 < 第一高点 (未创新高)
    """
    result = {
        "daily_kdj_k": None, "daily_kdj_d": None, "daily_kdj_j": None,
        "daily_divergence": "",
        "weekly_kdj_k": None, "weekly_kdj_d": None, "weekly_kdj_j": None,
        "weekly_divergence": "",
    }

    if len(df) < lookback:
        lookback = len(df)
    if lookback < 20:
        return result

    # --- 日线 KDJ ---
    last = df.iloc[-1]
    result["daily_kdj_k"] = _round(last.get("k"))
    result["daily_kdj_d"] = _round(last.get("d"))
    result["daily_kdj_j"] = _round(last.get("j"))
    result["daily_divergence"] = _find_divergence(df.tail(lookback), k_col, close_col)

    # --- 周线 KDJ ---
    weekly = resample_weekly(df)
    if len(weekly) >= 20:
        weekly = calc_kdj(weekly)
        w_last = weekly.iloc[-1]
        result["weekly_kdj_k"] = _round(w_last.get("k"))
        result["weekly_kdj_d"] = _round(w_last.get("d"))
        result["weekly_kdj_j"] = _round(w_last.get("j"))
        w_lookback = min(lookback, len(weekly))
        result["weekly_divergence"] = _find_divergence(weekly.tail(w_lookback), k_col, close_col)

    return result


def _find_divergence(df: pd.DataFrame, k_col: str = "k",
                     close_col: str = "fwd_close") -> str:
    """
    在给定 DataFrame 中检测背离
    使用峰谷检测: 找局部极值点, 然后比较价格与指标的走势

    关键约束:
      - 底背离: 两个 K 低点都必须在超卖区 (K < 30), 否则不成立
      - 顶背离: 两个 K 高点都必须在超买区 (K > 70), 否则不成立
    """
    if k_col not in df.columns or close_col not in df.columns:
        return ""

    prices = df[close_col].values
    k_vals = df[k_col].values
    n = len(prices)

    if n < 10:
        return ""

    # 找局部极值点 (前后各 5 根 K 线内最大/最小, 降低噪声)
    window = 5
    lows_idx = []
    highs_idx = []
    for i in range(window, n - window):
        if prices[i] <= min(prices[i - window:i + window + 1]):
            lows_idx.append(i)
        if prices[i] >= max(prices[i - window:i + window + 1]):
            highs_idx.append(i)

    # 底背离: 价格创新低 + K 未创新低 + 两个K低点都在超卖区(K<30)
    if len(lows_idx) >= 2:
        p1, p2 = lows_idx[-2], lows_idx[-1]
        if (prices[p2] < prices[p1]
            and not np.isnan(k_vals[p2]) and not np.isnan(k_vals[p1])
            and k_vals[p1] < 30 and k_vals[p2] < 30
            and k_vals[p2] > k_vals[p1]):
            return "底背离"

    # 顶背离: 价格创新高 + K 未创新高 + 两个K高点都在超买区(K>70)
    if len(highs_idx) >= 2:
        p1, p2 = highs_idx[-2], highs_idx[-1]
        if (prices[p2] > prices[p1]
            and not np.isnan(k_vals[p2]) and not np.isnan(k_vals[p1])
            and k_vals[p1] > 70 and k_vals[p2] > 70
            and k_vals[p2] < k_vals[p1]):
            return "顶背离"

    return ""


def _round(val, digits=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), digits)
