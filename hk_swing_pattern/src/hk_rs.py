"""HK 相对强度指标 (HK 专属):
  1) 年线斜率 = MA200 近5日变化率 ×52 (年化, %)
  2) RS vs HSI = stock / 2800.HK (盈富基金, HSI代理; EODHD无HSI指数) 比值
  3) RS_HSI 1W/4W 变化 = (stock/2800.HK) 比值的 5日 / 20日 变化 %
注: HK股票池无sector → 不算 RS_行业。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from data_provider import forward_adjust
import eodhd

BENCHMARKS = ["2800.HK"]  # 盈富基金, HSI 代理


def fetch_benchmarks(asof: str, lookback_days: int = 520) -> dict:
    raw = eodhd.fetch_all_eodhd(BENCHMARKS, asof, lookback_days, workers=5)
    out = {}
    for t, df in raw.items():
        if df is None or df.empty:
            continue
        fa = forward_adjust(df)
        if fa is not None and not fa.empty:
            out[t] = fa.set_index("date")["fwd_close"].astype(float)
    return out


def compute_extras(stock_daily: pd.DataFrame, benchmarks: dict) -> dict:
    out = {"年线斜率": None, "RS_HSI": None, "RS_HSI_1W": None, "RS_HSI_4W": None}
    c = stock_daily["fwd_close"].values.astype(float)
    dates = stock_daily["date"]
    ma200 = pd.Series(c).rolling(200, min_periods=200).mean().values
    if len(ma200) >= 6 and not np.isnan(ma200[-1]) and not np.isnan(ma200[-6]) and ma200[-6] > 0:
        out["年线斜率"] = round((ma200[-1] / ma200[-6] - 1) * 52 * 100, 2)
    bench = benchmarks.get("2800.HK")
    if bench is not None:
        stk = pd.Series(c, index=dates)
        rs = (stk / bench).dropna()
        if len(rs) >= 21:
            out["RS_HSI"] = round(float(rs.iloc[-1]), 4)
            out["RS_HSI_1W"] = round((rs.iloc[-1] / rs.iloc[-6] - 1) * 100, 2)
            out["RS_HSI_4W"] = round((rs.iloc[-1] / rs.iloc[-21] - 1) * 100, 2)
    return out
