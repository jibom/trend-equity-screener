"""美股相对强度指标 (US 专属):
  1) 年线斜率 = MA200 近5日变化率 ×52 (年化, %)
  2) RS vs S&P = stock / SPY 比值
  3) RS vs 行业 = stock / 行业ETF 比值 (GICS sector -> XLK/XLE/...)
  4) RS_SPY 1W/4W 变化 = (stock/SPY) 比值的 5日 / 20日 变化 %
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


def fetch_benchmarks(asof: str, lookback_days: int = 520) -> dict:
    """拉 SPY + 行业ETF, 返回 {ticker: pd.Series(date->fwd_close)}。"""
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
    """返回 5 个 US 专属指标 dict (None 表示算不出)。"""
    out = {"年线斜率": None, "RS_SPY": None, "RS_行业": None, "RS_SPY_1W": None, "RS_SPY_4W": None}
    c = stock_daily["fwd_close"].values.astype(float)
    dates = stock_daily["date"]
    # 1) MA200 年化斜率
    ma200 = pd.Series(c).rolling(200, min_periods=200).mean().values
    if len(ma200) >= 6 and not np.isnan(ma200[-1]) and not np.isnan(ma200[-6]) and ma200[-6] > 0:
        out["年线斜率"] = round((ma200[-1] / ma200[-6] - 1) * 52 * 100, 2)
    stk = pd.Series(c, index=dates)
    # 2) RS vs S&P + 4) 1W/4W 变化
    spy = benchmarks.get("SPY.US")
    if spy is not None:
        rs = (stk / spy).dropna()
        if len(rs) >= 21:
            out["RS_SPY"] = round(float(rs.iloc[-1]), 4)
            out["RS_SPY_1W"] = round((rs.iloc[-1] / rs.iloc[-6] - 1) * 100, 2)
            out["RS_SPY_4W"] = round((rs.iloc[-1] / rs.iloc[-21] - 1) * 100, 2)
    # 3) RS vs 行业 ETF
    etf = SECTOR_ETF.get((sector or "").strip())
    if etf:
        bench = benchmarks.get(etf)
        if bench is not None:
            rset = (stk / bench).dropna()
            if len(rset) >= 2:
                out["RS_行业"] = round(float(rset.iloc[-1]), 4)
    return out
