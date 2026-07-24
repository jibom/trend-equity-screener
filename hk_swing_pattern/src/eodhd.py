"""EODHD 港股日线 fetcher。返回与 WindFetcher.fetch 相同 schema (TRADE_DT/S_DQ_*/S_DQ_ADJ*/S_DQ_VOLUME),
使 forward_adjust 无需改动。EODHD 只给 adjusted_close, adjusted OHLC 用 close 因子推导。
API key 走环境变量 EODHD_TOKEN。支持并行拉取。"""
from __future__ import annotations
import os, json, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import pandas as pd

BASE = "https://eodhd.com/api"
WIND_COLS = ["TRADE_DT", "S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE",
             "S_DQ_ADJOPEN", "S_DQ_ADJHIGH", "S_DQ_ADJLOW", "S_DQ_ADJCLOSE", "S_DQ_VOLUME"]


def fetch_eodhd(code: str, asof: str, lookback_days: int = 520, timeout: int = 30) -> pd.DataFrame:
    """拉单股 EOD, 返回 Wind schema DataFrame (按 TRADE_DT 升序)。失败返回空。"""
    token = os.environ.get("EODHD_TOKEN", "")
    if not token:
        raise RuntimeError("缺少 EODHD_TOKEN 环境变量")
    frm = (datetime.strptime(asof, "%Y-%m-%d") - timedelta(days=int(lookback_days * 1.6))).strftime("%Y-%m-%d")
    url = f"{BASE}/eod/{code}?api_token={token}&fmt=json&from={frm}&to={asof}&period=d"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            r = json.load(resp)
    except Exception as e:
        print(f"  [EODHD err] {code}: {e}")
        return pd.DataFrame(columns=WIND_COLS)
    if not r:
        return pd.DataFrame(columns=WIND_COLS)
    df = pd.DataFrame(r)
    df["TRADE_DT"] = df["date"].str.replace("-", "")
    factor = df["adjusted_close"].astype(float) / df["close"].astype(float)
    out = pd.DataFrame({
        "TRADE_DT": df["TRADE_DT"],
        "S_DQ_OPEN": df["open"].astype(float),
        "S_DQ_HIGH": df["high"].astype(float),
        "S_DQ_LOW": df["low"].astype(float),
        "S_DQ_CLOSE": df["close"].astype(float),
        "S_DQ_ADJOPEN": df["open"].astype(float) * factor,
        "S_DQ_ADJHIGH": df["high"].astype(float) * factor,
        "S_DQ_ADJLOW": df["low"].astype(float) * factor,
        "S_DQ_ADJCLOSE": df["adjusted_close"].astype(float),
        "S_DQ_VOLUME": df["volume"].astype(float),
    })
    return out.sort_values("TRADE_DT").reset_index(drop=True)


def fetch_all_eodhd(codes: list, asof: str, lookback_days: int = 520,
                    workers: int = 12) -> dict:
    """并行拉取多股, 返回 {code: DataFrame}。"""
    out = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(fetch_eodhd, c, asof, lookback_days): c for c in codes}
        for i, f in enumerate(as_completed(fut), 1):
            code = fut[f]
            try:
                df = f.result()
                if not df.empty:
                    out[code] = df
            except Exception as e:
                print(f"  [EODHD err] {code}: {e}")
            if i % 50 == 0:
                print(f"  [EODHD] {i}/{len(codes)} ({time.time()-t0:.0f}s)")
    print(f"[EODHD] 完成 {len(out)}/{len(codes)}, 耗时 {time.time()-t0:.0f}s")
    return out
