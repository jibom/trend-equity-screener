"""数据层: 港股日线 (自包含 WindFetcher/forward_adjust) + 池加载"""
from __future__ import annotations
import sys, io, os, json, time, datetime as dt
from pathlib import Path
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")  # 本地 .env; GitHub Actions 用 Secrets (无 .env 时 no-op)

from data_provider import WindFetcher, forward_adjust  # noqa: E402

CACHE_FILE = ROOT / "output" / "float_shares_cache.json"


def load_pool(pool_csv: str) -> list:
    """加载股票池 -> [(code, name, sector)] (按 code 去重, 保留首条)。相对路径按项目根解析。"""
    p = Path(pool_csv)
    if not p.is_absolute():
        p = ROOT / pool_csv
    df = pd.read_csv(p)
    out, seen = [], set()
    for _, r in df.iterrows():
        code = str(r.get("code", "")).strip()
        if not code or code == "nan":
            continue
        if not code.endswith(".HK"):
            code = code + ".HK"
        if code in seen:
            continue
        seen.add(code)
        out.append((code, str(r.get("name_cn", "")).strip(),
                    str(r.get("gics_sector", "")).strip()))
    return out


def make_fetcher(lookback_days: int) -> WindFetcher:
    return WindFetcher(lookback_days=lookback_days)


def fetch_daily(fetcher: WindFetcher, code: str, asof: str) -> pd.DataFrame | None:
    """返回前复权日线 (含 fwd_open/high/low/close, volume, raw_close, date); 失败返回 None"""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            raw = fetcher.fetch(code, asof)
    except Exception as e:
        print(f"  [fetch err] {code}: {e}")
        return None
    if raw is None or raw.empty:
        return None
    daily = forward_adjust(raw)
    if daily is None or daily.empty:
        return None
    daily = daily[daily["date"] <= pd.to_datetime(asof)].reset_index(drop=True)
    return daily


# ---------------- 流通股本 (yfinance, 日缓存) ----------------
def _yf_code(wind_code: str) -> str:
    num, suf = wind_code.split(".")
    return f"{int(num):04d}.{suf}"


def fetch_float_shares(codes: list, today_iso: str) -> dict:
    """返回 {code: floatShares(float|None)}; 日缓存, 缺失增量拉取"""
    import yfinance as yf
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if CACHE_FILE.exists():
        try:
            cache = json.load(open(CACHE_FILE, encoding="utf-8"))
        except Exception:
            cache = {}
    if cache.get("_date") != today_iso:
        cache = {"_date": today_iso}

    miss = [c for c in codes if c not in cache]
    if not miss:
        return {c: cache.get(c) for c in codes}
    print(f"[Float] 增量拉取 {len(miss)} 只 ...")
    t0 = time.time()
    for i, code in enumerate(miss):
        try:
            info = yf.Ticker(_yf_code(code)).info
            fs = info.get("floatShares") or info.get("sharesOutstanding")
            cache[code] = float(fs) if fs else None
        except Exception:
            cache[code] = None
        if (i + 1) % 50 == 0:
            print(f"  [Float] {i+1}/{len(miss)} ({time.time()-t0:.0f}s)")
    cache["_date"] = today_iso
    try:
        json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"))
    except Exception as e:
        print(f"  [Float] 缓存写入失败: {e}")
    return {c: cache.get(c) for c in codes}
