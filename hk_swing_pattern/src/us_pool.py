"""美股 universe: S&P500 ∪ Nasdaq100 成分股 ∪ EU/JP/KR 主要 ADR(策展) 的并集

数据源: Wikipedia (带 UA 抓 HTML 再 read_html)。
输出: [(code, name, sector)] 其中 code = "{SYMBOL}.US" (EODHD 格式)。
Sector: S&P500 取 GICS Sector; NDX 独有票取 ICB Industry; ADR 手工赋。
缓存: output/us_pool.csv, 用 --rebuild 强制刷新。
"""
from __future__ import annotations
import io, urllib.request
import pandas as pd

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
NDX_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
_UA = {"User-Agent": "Mozilla/5.0"}


# 策展 ADR: 仅欧洲/日本/韩国主要大盘 (ASML/ARM 已在 NDX100, 并集自动覆盖, 此处只列不在指数里的)
# (code, name, sector)
CURATED_ADRS = [
    ("SAP.US",   "SAP SE",              "Information Technology"),
    ("NTDOY.US", "Nintendo",            "Communication Services"),
    ("SONY.US",  "Sony Group",          "Communication Services"),
    ("TM.US",    "Toyota Motor",        "Consumer Discretionary"),
    ("MRAAY.US", "Murata Manufacturing", "Information Technology"),  # 村田 (OTC sponsored ADR)
    ("LPL.US",   "LG Display",          "Information Technology"),
    ("PKX.US",   "POSCO",               "Materials"),
    ("SKM.US",   "SK Telecom",          "Communication Services"),
    ("SKHY.US",  "SK Hynix",            "Information Technology"),   # 2026-07 新上 NASDAQ ADR, <60日会被跳过
]


def _fetch_html(url: str) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", "replace")


def _sp500() -> pd.DataFrame:
    """S&P500 成分股 -> DataFrame[Symbol, Security, GICS Sector]"""
    df = pd.read_html(io.StringIO(_fetch_html(SP500_URL)))[0]
    df = df[["Symbol", "Security", "GICS Sector"]].copy()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    return df


def _ndx100() -> pd.DataFrame:
    """Nasdaq-100 成分股 -> DataFrame[Ticker, Company, ICB Industry]"""
    df = pd.read_html(io.StringIO(_fetch_html(NDX_URL)))[0]
    # ICB Industry 列名带 "[1]" 后缀, 按前缀匹配
    icb_col = next(c for c in df.columns if str(c).startswith("ICB Industry"))
    df = df[["Ticker", "Company", icb_col]].copy()
    df.columns = ["Ticker", "Company", "ICB Industry"]
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    return df


def _to_eodhd(sym: str) -> str:
    """S&P/NDX 符号 -> EODHD 代码。EODHD 美股用连字符表share class (BRK.B->BRK-B)。"""
    return f"{sym.replace('.', '-')}.US"


def build_pool() -> list[tuple[str, str, str]]:
    """并集(S&P500 ∪ NDX100 ∪ 策展ADR), 按 code 去重。返回 [(code, name, sector)]。
    Sector 优先级: S&P500 GICS > NDX ICB Industry > ADR 手工。"""
    sp = _sp500()
    ndx = _ndx100()
    out, seen = [], set()
    # 1) S&P500 (GICS Sector)
    for _, r in sp.iterrows():
        sym = str(r["Symbol"]).strip()
        if not sym or sym == "nan":
            continue
        code = _to_eodhd(sym)
        if code in seen:
            continue
        seen.add(code)
        out.append((code, str(r["Security"]).strip(), str(r["GICS Sector"]).strip()))
    # 2) NDX 独有 (ICB Industry 作 Sector)
    for _, r in ndx.iterrows():
        t = str(r["Ticker"]).strip()
        if not t or t == "nan":
            continue
        code = _to_eodhd(t)
        if code in seen:
            continue
        seen.add(code)
        out.append((code, str(r["Company"]).strip(), str(r["ICB Industry"]).strip()))
    # 3) 策展 ADR
    for code, name, sector in CURATED_ADRS:
        if code in seen:
            continue
        seen.add(code)
        out.append((code, name, sector))
    return out


def load_us_pool(cache_path, rebuild: bool = False) -> list[tuple[str, str, str]]:
    """缓存优先; rebuild=True 或缓存缺失则重建并写 CSV。"""
    from pathlib import Path
    cache_path = Path(cache_path)
    if not rebuild and cache_path.exists():
        df = pd.read_csv(cache_path)
        return [(str(r.code).strip(), str(r.name).strip(), str(r.sector).strip())
                for r in df.itertuples()]
    pool = build_pool()
    pd.DataFrame(pool, columns=["code", "name", "sector"]).to_csv(
        cache_path, index=False, encoding="utf-8-sig")
    print(f"[Pool] 重建 {len(pool)} 只 -> {cache_path}")
    return pool


if __name__ == "__main__":
    import sys, argparse
    ap = argparse.ArgumentParser(description="构建美股 universe (S&P500∩NDX100 + ADR)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--cache", default="output/us_pool.csv")
    args = ap.parse_args()
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = load_us_pool(args.cache, rebuild=args.rebuild)
    print(f"[Pool] {len(p)} 只")
    for c, n, s in p[:8]:
        print(f"  {c:10s} {n:24s} {s}")
    print("  ...")
