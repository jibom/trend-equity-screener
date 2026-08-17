"""A股/ETF 相对强度指标 (A股专属, Mansfield RS):
  1) 年线斜率 = MA200 近5日变化率 ×52 (年化, %)
  2) MRS_沪深300 = Mansfield RS vs 510300.SH(沪深300ETF代理) = (stock/510300)/(252日均线)-1, ×100
  3) MRS_中证500 = Mansfield RS vs 510500.SH(中证500ETF代理) 同法 (仅 A股扫描用)
  4) _1W / _4W = MRS 分数的 5日 / 20日 变化
基准取自 chinaclosedfundeodprice (与 HK 用 2800.HK ETF 代理同思路; EODHD 无可靠 A股 ETF)。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from data_provider import WindFetcher, forward_adjust

BENCHMARKS = ["510300.SH", "510500.SH"]  # 沪深300ETF, 中证500ETF (代理指数)
MRS_PERIOD = 252


def _mrs(rs: pd.Series) -> pd.Series:
    return (rs / rs.rolling(MRS_PERIOD, min_periods=MRS_PERIOD).mean() - 1) * 100


def fetch_benchmarks(asof: str, lookback_days: int = 520) -> dict:
    """拉 510300.SH + 510500.SH 前复权收盘序列。"""
    f = WindFetcher(lookback_days=lookback_days, table="chinaclosedfundeodprice")
    out = {}
    try:
        for t in BENCHMARKS:
            raw = f.fetch(t, asof)
            if raw is None or raw.empty:
                continue
            fa = forward_adjust(raw)
            if fa is not None and not fa.empty:
                out[t] = fa.set_index("date")["fwd_close"].astype(float)
    finally:
        f.close()
    return out


def compute_extras(stock_daily: pd.DataFrame, benchmarks: dict) -> dict:
    """年线斜率 + MRS_沪深300(1W/4W) + MRS_中证500(1W/4W)。ETF 扫描只取沪深300列。"""
    out = {"年线斜率": None,
           "MRS_沪深300": None, "MRS_沪深300_1W": None, "MRS_沪深300_4W": None,
           "MRS_中证500": None, "MRS_中证500_1W": None, "MRS_中证500_4W": None}
    c = stock_daily["fwd_close"].values.astype(float)
    dates = stock_daily["date"]
    ma200 = pd.Series(c).rolling(200, min_periods=200).mean().values
    if len(ma200) >= 6 and not np.isnan(ma200[-1]) and not np.isnan(ma200[-6]) and ma200[-6] > 0:
        out["年线斜率"] = round((ma200[-1] / ma200[-6] - 1) * 52 * 100, 2)
    stk = pd.Series(c, index=dates)
    hs300 = benchmarks.get("510300.SH")
    if hs300 is not None:
        mrs = _mrs((stk / hs300).dropna()).dropna()
        if len(mrs) >= 21:
            out["MRS_沪深300"] = round(float(mrs.iloc[-1]), 2)
            out["MRS_沪深300_1W"] = round(float(mrs.iloc[-1] - mrs.iloc[-6]), 2)
            out["MRS_沪深300_4W"] = round(float(mrs.iloc[-1] - mrs.iloc[-21]), 2)
    zz500 = benchmarks.get("510500.SH")
    if zz500 is not None:
        mrs5 = _mrs((stk / zz500).dropna()).dropna()
        if len(mrs5) >= 21:
            out["MRS_中证500"] = round(float(mrs5.iloc[-1]), 2)
            out["MRS_中证500_1W"] = round(float(mrs5.iloc[-1] - mrs5.iloc[-6]), 2)
            out["MRS_中证500_4W"] = round(float(mrs5.iloc[-1] - mrs5.iloc[-21]), 2)
    return out
