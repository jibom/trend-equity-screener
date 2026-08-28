"""Twelve Data 客户端: 基本面首选 + 美股/A股价格&intraday 首选 (见 memory data-source-routing)。

通用: quote / price / time_series / statistics / profile。
US EOD 专用: fetch_us_eod / fetch_all_us 返回与 eodhd.py 相同的 WIND_COLS schema
(S_DQ_CLOSE 与 S_DQ_ADJ* 都填 Twelve 复权值, factor=1), 供 data_provider.forward_adjust
+ swing.analyze 无感复用, drop-in 替代 eodhd。

Twelve /time_series 返回复权价 (split-adjusted, 跨拆股无跳变, 适合 swing 连续序列)。
API key 走 TWELVEDATA_KEY env (.env 或 GitHub Actions secrets)。
"""
from __future__ import annotations
import os, json, time, threading
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request, urllib.parse, urllib.error
import pandas as pd

BASE = "https://api.twelvedata.com"
WIND_COLS = ["TRADE_DT", "S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE",
             "S_DQ_ADJOPEN", "S_DQ_ADJHIGH", "S_DQ_ADJLOW", "S_DQ_ADJCLOSE", "S_DQ_VOLUME"]


def _key() -> str:
    """key 加载: TWELVEDATA_KEY env → configs/twelvedata_key.local (gitignored) → ''。"""
    key = os.environ.get("TWELVEDATA_KEY", "")
    if key:
        return key
    local = os.path.join(os.path.dirname(__file__), "..", "configs", "twelvedata_key.local")
    if os.path.exists(local):
        return open(local, encoding="utf-8").read().strip()
    return ""


# 请求节流: 限制请求起始速率避免 Twelve 并发 429 (Pro ~800/min, 取保守 ~6.7 req/s)
_PACE_LOCK = threading.Lock()
_LAST_REQ = [0.0]
_PACE_INTERVAL = 0.15   # 秒/请求 ≈ 6.7 req/s ≈ 400 req/min

def _pace():
    """串行化请求起始时间, 保证全局 ≤ ~6.7 req/s; 网络IO 仍并发 (锁只在算/睡间隔时持有)。"""
    with _PACE_LOCK:
        now = time.time()
        wait = _PACE_INTERVAL - (now - _LAST_REQ[0])
        if wait > 0:
            time.sleep(wait)
            _LAST_REQ[0] = time.time()
        else:
            _LAST_REQ[0] = now


def _get(path, params=None, timeout=30, retries=6):
    """核心 GET。429 退避重试; 其它错误返回 {status:error,...} 结构不抛。"""
    p = dict(params or {}); p["apikey"] = _key()
    url = BASE + path + "?" + urllib.parse.urlencode(p)
    for attempt in range(retries):
        try:
            _pace()
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            if e.code == 429:
                time.sleep(min(8 * (attempt + 1), 30)); continue
            try:
                j = json.loads(body); j.setdefault("_http", e.code); return j
            except Exception:
                return {"status": "error", "_http": e.code, "message": body[:200]}
        except Exception as e:
            if attempt == retries - 1:
                return {"status": "error", "_err": str(e)}
            time.sleep(1.0)
    return {"status": "error", "_err": "429 重试耗尽"}


def _err(d) -> bool:
    return isinstance(d, dict) and (d.get("status") == "error" or "_http" in d or "_err" in d)


# ---------------- 通用端点 ----------------

def quote(symbol, exchange=None):
    p = {"symbol": symbol}
    if exchange: p["exchange"] = exchange
    return _get("/quote", p)

def price(symbol, exchange=None):
    p = {"symbol": symbol}
    if exchange: p["exchange"] = exchange
    return _get("/price", p)

def statistics(symbol, exchange=None):
    p = {"symbol": symbol}
    if exchange: p["exchange"] = exchange
    return _get("/statistics", p)

def profile(symbol, exchange=None):
    p = {"symbol": symbol}
    if exchange: p["exchange"] = exchange
    return _get("/profile", p)

def time_series(symbol, exchange=None, interval="1day", start_date=None,
                end_date=None, outputsize=None, timeout=30, retries=4) -> pd.DataFrame:
    """返回 DataFrame[date, open, high, low, close, volume] 按日期升序, 失败返回空。
    Twelve 返回复权价 (split-adjusted)。start/end 为 YYYY-MM-DD; **end_date 开区间(不含当天)**,
    要含某交易日需传 end_date = 该日 +1 天 (fetch_us_eod 已自动 +1)。"""
    p = {"symbol": symbol, "interval": interval}
    if exchange: p["exchange"] = exchange
    if start_date: p["start_date"] = start_date
    if end_date: p["end_date"] = end_date
    p["outputsize"] = str(outputsize) if outputsize else "5000"
    d = _get("/time_series", p, timeout=timeout, retries=retries)
    if _err(d) or not isinstance(d, dict):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    vals = d.get("values", [])
    if not vals:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(vals)
    df["date"] = pd.to_datetime(df["datetime"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["date", "open", "high", "low", "close", "volume"]] \
        .sort_values("date").reset_index(drop=True)


def fetch_dividends(symbol, start_date=None, end_date=None, exchange=None,
                    timeout=30, retries=4) -> pd.DataFrame:
    """返回 DataFrame[date, amount] (按日期升序)。start/end YYYY-MM-DD。
    ⚠️ 不带 date range 时 Twelve 只回最近 1 条; 必须传 start_date+end_date。"""
    p = {"symbol": symbol, "outputsize": "5000"}
    if exchange: p["exchange"] = exchange
    if start_date: p["start_date"] = start_date
    if end_date: p["end_date"] = end_date
    d = _get("/dividends", p, timeout=timeout, retries=retries)
    if _err(d) or not isinstance(d, dict):
        return pd.DataFrame(columns=["date", "amount"])
    dv = d.get("dividends", [])
    if not dv:
        return pd.DataFrame(columns=["date", "amount"])
    df = pd.DataFrame(dv)
    df["date"] = pd.to_datetime(df["ex_date"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df[["date", "amount"]].sort_values("date").reset_index(drop=True)


def _div_adjust(df: pd.DataFrame, divs: pd.DataFrame) -> pd.DataFrame:
    """股息后复权: adj(t)=raw(t)×Π_{d>t}(1-amt/prev_close_d), prev_close=除息前一交易日收盘。
    仅用 first_bar<ex_date≤last_bar 的股息; adj(last)=raw(last) → forward_adjust factor=1。
    使 S_DQ_ADJ* 与 EODHD adjusted_close (split+div) 对齐 (Twelve 原始只 split 不 div)。"""
    df = df.copy()
    df["adj_factor"] = 1.0
    if divs is None or divs.empty:
        return df
    first_d = df["date"].iloc[0]; last_d = df["date"].iloc[-1]
    for _, dv in divs.iterrows():
        ex = dv["date"]; amt = dv["amount"]
        if pd.isna(amt) or amt <= 0 or not (first_d < ex <= last_d):
            continue
        prior = df[df["date"] < ex]
        if prior.empty:
            continue
        prev_close = float(prior["close"].iloc[-1])
        if prev_close <= 0:
            continue
        r = amt / prev_close
        if not (0 < r < 1):   # 异常股息(>100%) 跳过
            continue
        df.loc[df["date"] < ex, "adj_factor"] *= (1 - r)
    return df


# ---------------- US EOD → WIND_COLS (drop-in 替代 eodhd) ----------------

def _us_symbol(code: str) -> str:
    """AAPL.US -> AAPL; 不带后缀原样返回。"""
    return code.split(".")[0] if "." in code else code

def fetch_us_eod(code: str, asof: str, lookback_days: int = 520,
                 timeout: int = 30, retries: int = 4,
                 adjust_dividends: bool = True) -> pd.DataFrame:
    """拉美股 EOD 日线, 返回 WIND_COLS DataFrame (按 TRADE_DT 升序)。
    S_DQ_CLOSE=Twelve 原始(split-adjusted)收盘; S_DQ_ADJ*=股息后复权(与 EODHD adjusted_close 对齐)。
    forward_adjust: factor=adj_last/raw_last=1 → fwd=split+div 复权序列, 末日=当日收盘。
    dash 符号 (BRK-B) 404 回退 '-'->'.'。adjust_dividends=False 跳过 /dividends (省调用, 仅 split)。"""
    base = datetime.strptime(asof, "%Y-%m-%d")
    frm = (base - timedelta(days=int(lookback_days * 1.6))).strftime("%Y-%m-%d")
    end = (base + timedelta(days=1)).strftime("%Y-%m-%d")   # Twelve end_date 开区间, +1 天以含 asof 当日
    sym = _us_symbol(code)
    df = time_series(sym, interval="1day", start_date=frm, end_date=end,
                     outputsize=5000, timeout=timeout, retries=retries)
    if df.empty and "-" in sym:   # BRK-B -> BRK.B
        df = time_series(sym.replace("-", "."), interval="1day", start_date=frm, end_date=end,
                         outputsize=5000, timeout=timeout, retries=retries)
    if df.empty:
        return pd.DataFrame(columns=WIND_COLS)
    if adjust_dividends:
        df = _div_adjust(df, fetch_dividends(sym, start_date=frm, end_date=end,
                                             timeout=timeout, retries=retries))
    else:
        df["adj_factor"] = 1.0
    df["TRADE_DT"] = df["date"].dt.strftime("%Y%m%d")
    out = pd.DataFrame({
        "TRADE_DT": df["TRADE_DT"],
        "S_DQ_OPEN": df["open"], "S_DQ_HIGH": df["high"], "S_DQ_LOW": df["low"],
        "S_DQ_CLOSE": df["close"],
        "S_DQ_ADJOPEN": df["open"] * df["adj_factor"],
        "S_DQ_ADJHIGH": df["high"] * df["adj_factor"],
        "S_DQ_ADJLOW": df["low"] * df["adj_factor"],
        "S_DQ_ADJCLOSE": df["close"] * df["adj_factor"],
        "S_DQ_VOLUME": df["volume"],
    })
    return out.sort_values("TRADE_DT").reset_index(drop=True)

def fetch_all_us(codes: list, asof: str, lookback_days: int = 520,
                 workers: int = 6) -> dict:
    """并行拉美股, 返回 {code: DataFrame}。无 key 时早退 (走 Wind 回退)。"""
    if not _key():
        print("[Twelve] 未配置 TWELVEDATA_KEY, 跳过"); return {}
    out = {}; t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut = {ex.submit(fetch_us_eod, c, asof, lookback_days): c for c in codes}
        for i, f in enumerate(as_completed(fut), 1):
            code = fut[f]
            try:
                df = f.result()
                if not df.empty:
                    out[code] = df
            except Exception as e:
                print(f"  [Twelve err] {code}: {e}")
            if i % 50 == 0:
                print(f"  [Twelve] {i}/{len(codes)} ({time.time()-t0:.0f}s)")
    print(f"[Twelve] 完成 {len(out)}/{len(codes)}, 耗时 {time.time()-t0:.0f}s")
    return out
