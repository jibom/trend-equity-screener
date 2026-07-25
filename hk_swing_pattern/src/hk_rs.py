"""HK 相对强度指标 (HK 专属, Mansfield RS):
  1) 年线斜率 = MA200 近5日变化率 ×52 (年化, %)
  2) MRS_HSI = Mansfield RS vs 2800.HK(盈富基金,HSI代理) = (stock/2800.HK)/(252日均线)-1, ×100
  3) MRS_HSI_1W / MRS_HSI_4W = MRS 分数的 5日 / 20日 变化
  4) MRS_HD = Mansfield RS vs 3110.HK(恒生高股息率ETF) 同法
注: HK股票池无sector → 不算 RS_行业。EODHD无HSI/高股息指数, 用ETF代理。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from data_provider import forward_adjust
import eodhd

BENCHMARKS = ["2800.HK", "3110.HK"]  # 盈富(HSI代理), 恒生高股息率ETF
MRS_PERIOD = 252


def _mrs(rs: pd.Series) -> pd.Series:
    return (rs / rs.rolling(MRS_PERIOD, min_periods=MRS_PERIOD).mean() - 1) * 100


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
    out = {"年线斜率": None, "MRS_HSI": None, "MRS_HSI_1W": None, "MRS_HSI_4W": None, "MRS_HD": None}
    c = stock_daily["fwd_close"].values.astype(float)
    dates = stock_daily["date"]
    ma200 = pd.Series(c).rolling(200, min_periods=200).mean().values
    if len(ma200) >= 6 and not np.isnan(ma200[-1]) and not np.isnan(ma200[-6]) and ma200[-6] > 0:
        out["年线斜率"] = round((ma200[-1] / ma200[-6] - 1) * 52 * 100, 2)
    stk = pd.Series(c, index=dates)
    bench = benchmarks.get("2800.HK")
    if bench is not None:
        mrs = _mrs((stk / bench).dropna()).dropna()
        if len(mrs) >= 21:
            out["MRS_HSI"] = round(float(mrs.iloc[-1]), 2)
            out["MRS_HSI_1W"] = round(float(mrs.iloc[-1] - mrs.iloc[-6]), 2)
            out["MRS_HSI_4W"] = round(float(mrs.iloc[-1] - mrs.iloc[-21]), 2)
    hd = benchmarks.get("3110.HK")
    if hd is not None:
        mrs_hd = _mrs((stk / hd).dropna()).dropna()
        if len(mrs_hd) >= 1:
            out["MRS_HD"] = round(float(mrs_hd.iloc[-1]), 2)
    return out
