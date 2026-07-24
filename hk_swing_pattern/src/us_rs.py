"""美股相对强度指标 (US 专属, Mansfield RS):
  1) 年线斜率 = MA200 近5日变化率 ×52 (年化, %)
  2) MRS_SPY = Mansfield RS vs SPY = (stock/SPY)/(252日均线) - 1, ×100
  3) MRS_行业 = Mansfield RS vs 行业ETF (GICS sector -> XLK/XLE/...)
  4) MRS_SPY_1W / MRS_SPY_4W = MRS_SPY 分数的 5日 / 20日 变化
基准: SPY + 11 行业ETF, 一次 EODHD 并行拉取。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from data_provider import forward_adjust
import eodhd

SECTOR_ETF = {
    "Information Technology": "XLK.US", "Health Care": "XLV.US", "Financials": "XLF.US",
    "Consumer Discretionary": "XLY.US", "Consumer Staples": "XLP.US", "Industrials": "XLI.US",
    "Energy": "XLE.US", "Utilities": "XLU.US", "Materials": "XLB.US",
    "Real Estate": "XLRE.US", "Communication Services": "XLC.US",
}
BENCHMARKS = ["SPY.US"] + sorted(set(SECTOR_ETF.values()))
MRS_PERIOD = 252  # ~52 周


def _mrs(rs: pd.Series) -> pd.Series:
    """Mansfield RS = (rs / rs.rolling(PERIOD).mean() - 1) * 100。"""
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


def compute_extras(stock_daily: pd.DataFrame, benchmarks: dict, sector: str) -> dict:
    out = {"年线斜率": None, "MRS_SPY": None, "MRS_行业": None, "MRS_SPY_1W": None, "MRS_SPY_4W": None}
    c = stock_daily["fwd_close"].values.astype(float)
    dates = stock_daily["date"]
    ma200 = pd.Series(c).rolling(200, min_periods=200).mean().values
    if len(ma200) >= 6 and not np.isnan(ma200[-1]) and not np.isnan(ma200[-6]) and ma200[-6] > 0:
        out["年线斜率"] = round((ma200[-1] / ma200[-6] - 1) * 52 * 100, 2)
    stk = pd.Series(c, index=dates)
    # MRS vs SPY + 1W/4W 变化
    spy = benchmarks.get("SPY.US")
    if spy is not None:
        mrs = _mrs((stk / spy).dropna()).dropna()
        if len(mrs) >= 21:
            out["MRS_SPY"] = round(float(mrs.iloc[-1]), 2)
            out["MRS_SPY_1W"] = round(float(mrs.iloc[-1] - mrs.iloc[-6]), 2)
            out["MRS_SPY_4W"] = round(float(mrs.iloc[-1] - mrs.iloc[-21]), 2)
    # MRS vs 行业 ETF
    etf = SECTOR_ETF.get((sector or "").strip())
    if etf:
        bench = benchmarks.get(etf)
        if bench is not None:
            mrs_et = _mrs((stk / bench).dropna()).dropna()
            if len(mrs_et) >= 1:
                out["MRS_行业"] = round(float(mrs_et.iloc[-1]), 2)
    return out
