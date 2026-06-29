"""美股 ETF 流动性 Top300 + 按基准去重 + passive/active 标注.

数据源:
  - EODHD: exchange-symbol-list/US (Type==ETF) + eod-bulk-last-day/US (单日OHLCV, 算流动性)
  - yfinance: 每只 ETF 的 name/category/fundFamily/totalAssets (AUM), 限流需 sleep+缓存
  - 基准/passive-active: EODHD fundamentals 403, yfinance 无直接字段 → 名称关键词解析 + 知名 ticker 白名单

流程:
  1. EODHD ETF 列表 + bulk 成交量 → 成交额=close×volume, 取 top300
  2. yfinance 逐个补 name/category/fundFamily/AUM (缓存断点续跑)
  3. benchmark_for(ticker,name,category): 白名单 + 名称正则, 匹配不上用 category 兜底
  4. is_active(ticker,name,fundFamily): 主动家族白名单 + 名称含"Active"/"ARK" → Active, 否则 Passive
  5. 按 benchmark 去重 (同基准只留流动性最高的一只; benchmark 为 category 兜底或 '—' 的不去重)
  6. 输出 output/us_etf_top300.md (表格) + .csv

用法: python scripts/etf_screener.py [--top 300] [--asof 2026-06-26]
"""
from __future__ import annotations
import os, sys, io, re, time, argparse, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR, "output")
EODHD_BASE = "https://eodhd.com/api"


def _load_eodhd_key():
    k = os.getenv("EODHD_KEY")
    if k:
        return k
    local = os.path.join(PROJECT_DIR, "configs", "eodhd_key.local")
    if os.path.exists(local):
        return open(local, encoding="utf-8").read().strip()
    raise RuntimeError("EODHD_KEY 未设置")


EODHD_KEY = _load_eodhd_key()


# ── 知名 ETF → 基准 白名单 (名称不直接含指数名的, 如 QQQ) ──
TICKER_BENCH = {
    "SPY": "S&P 500", "VOO": "S&P 500", "IVV": "S&P 500", "SPLG": "S&P 500", "SXPB": "S&P 500",
    "RSP": "S&P 500 Equal Weight", "EQAL": "S&P 500 Equal Weight",
    "QQQ": "NASDAQ-100", "QQQM": "NASDAQ-100", "TQQQ": "NASDAQ-100 3x",
    "QLD": "NASDAQ-100 2x", "PSQ": "NASDAQ-100 Inverse",
    "DIA": "Dow Jones Industrial Average",
    "IWM": "Russell 2000", "IWO": "Russell 2000 Growth", "IWN": "Russell 2000 Value",
    "IWV": "Russell 3000", "IWB": "Russell 1000", "IWD": "Russell 1000 Value", "IWF": "Russell 1000 Growth",
    "VTI": "CRSP US Total Market", "SCHB": "Dow Jones US Broad Market", "ITOT": "S&P Total Market",
    "VEA": "FTSE Developed", "IEFA": "MSCI EAFE", "EFA": "MSCI EAFE",
    "VWO": "FTSE Emerging", "EEM": "MSCI Emerging Markets", "IEMG": "MSCI Emerging Markets",
    "ACWI": "MSCI ACWI", "ACWV": "MSCI ACWI Low Volatility",
    "AGG": "Bloomberg US Aggregate Bond", "BND": "Bloomberg US Aggregate Bond", "SCHZ": "Bloomberg US Aggregate Bond",
    "TLT": "US Treasury 20+ Year", "SCHQ": "US Treasury 20+ Year",
    "IEF": "US Treasury 7-10 Year", "SHY": "US Treasury 1-3 Year", "SHV": "US Treasury 0-1 Year",
    "LQD": "Bloomberg US Corporate Bond", "HYG": "Bloomberg US High Yield", "JNK": "Bloomberg US High Yield",
    "TIP": "Bloomberg US TIPS",
    "GLD": "Gold", "IAU": "Gold", "GLDM": "Gold", "SGOL": "Gold",
    "SLV": "Silver", "SIVR": "Silver",
    "USO": "WTI Crude Oil", "UNG": "Natural Gas",
    "UUP": "US Dollar Index",
    # sector SPDRs
    "XLF": "S&P Financials", "XLK": "S&P Information Technology", "XLE": "S&P Energy",
    "XLV": "S&P Health Care", "XLI": "S&P Industrials", "XLY": "S&P Consumer Discretionary",
    "XLP": "S&P Consumer Staples", "XLU": "S&P Utilities", "XLB": "S&P Materials",
    "XLRE": "S&P Real Estate", "XLC": "S&P Communication Services",
    # 其它常见
    "MTUM": "MSCI USA Momentum", "VLUE": "MSCI USA Value", "QUAL": "MSCI USA Quality",
    "USMV": "MSCI USA Min Volatility", "EFG": "MSCI EAFE Growth", "EFV": "MSCI EAFE Value",
    "EPP": "MSCI Pacific", "ILF": "MSCI Latin America", "EPHE": "MSCI Philippines",
    "PFF": "Preferred Stock", "SCHD": "Dow Jones US Dividend 100", "VYM": "FTSE High Dividend Yield",
    "DVY": "Dow Jones US Select Dividend", "HDV": "Morningstar Dividend Yield Focus",
}

# ── 名称 → 基准 正则 (按顺序匹配, 先命中先用) ──
NAME_BENCH = [
    (r"\b[23]x\b|ultrapro|daily\s+(bull|bear)|inverse\s+etf|\bshort\b.*etf|leveraged", "Leveraged/Inverse"),
    (r"semiconductor|semis?\b", "Semiconductor"),
    (r"s&p\s*500|sp\s*500|sp500", "S&P 500"),
    (r"s&p\s*100", "S&P 100"),
    (r"nasdaq-?100|ndx-?100", "NASDAQ-100"),
    (r"nasdaq\s*composite", "NASDAQ Composite"),
    (r"russell\s*2000", "Russell 2000"),
    (r"russell\s*1000", "Russell 1000"),
    (r"russell\s*3000", "Russell 3000"),
    (r"dow\s*jones|djia|industrial\s*average", "Dow Jones"),
    (r"msci\s*eafe|developed\s*markets", "MSCI EAFE"),
    (r"msci\s*emerging|emerging\s*markets", "MSCI Emerging Markets"),
    (r"msci\s*acwi|all\s*country\s*world", "MSCI ACWI"),
    (r"bloomberg|barclays.*(aggregate|agg)|us\s*aggregate", "Bloomberg US Aggregate Bond"),
    (r"20\+?\s*year\s*treasury", "US Treasury 20+ Year"),
    (r"7-?10\s*year\s*treasury", "US Treasury 7-10 Year"),
    (r"tips|inflation.?protected", "TIPS"),
    (r"high\s*yield|junk", "US High Yield Bond"),
    (r"corporate\s*bond", "US Corporate Bond"),
    (r"gold\b", "Gold"),
    (r"silver\b", "Silver"),
    (r"crude\s*oil|wti|brent", "Crude Oil"),
    (r"natural\s*gas", "Natural Gas"),
    (r"u\.?s\.?\s*dollar|dollar\s*index", "US Dollar"),
    (r"preferred\s*stock|preferreds", "Preferred Stock"),
    (r"dividend", "Dividend"),
    (r"reit|real\s*estate", "Real Estate / REIT"),
    (r"7-?10\s*year|intermediate\s*treasury", "US Treasury 7-10 Year"),
]

# ── 主动 ETF 家族/关键词 ──
ACTIVE_TICKERS = {"ARKK", "ARKW", "ARKG", "ARKQ", "ARKF", "ARKX", "ARKZ"}
ACTIVE_FAMILIES = ("ARK", "Capital Group", "Capital Crescendo")  # fundFamily 含这些 → 主动
ACTIVE_NAME_KW = ("active", "ark ", "arknxt", "stocktopus")


def benchmark_for(ticker, name, category):
    if ticker in TICKER_BENCH:
        return TICKER_BENCH[ticker]
    n = (name or "").lower() if isinstance(name, str) else ""
    for pat, b in NAME_BENCH:
        if re.search(pat, n):
            return b
    # 兜底: 用 Morningstar category 做粗分组 (同名 category 的 ETF 仍各自保留, 不去重)
    if isinstance(category, str) and category:
        return category
    return "—"


def is_active(ticker, name, fund_family):
    if ticker in ACTIVE_TICKERS:
        return "Active"
    ff = fund_family if isinstance(fund_family, str) else ""
    nm = name if isinstance(name, str) else ""
    s = f"{nm} {ff}".lower()
    if any(kw in s for kw in ACTIVE_NAME_KW):
        return "Active"
    if ff and any(fam.lower() in ff.lower() for fam in ACTIVE_FAMILIES):
        return "Active"
    return "Passive"


# ── EODHD 取数 ──
def fetch_etf_list():
    r = requests.get(f"{EODHD_BASE}/exchange-symbol-list/US",
                     params={"api_token": EODHD_KEY, "fmt": "json"}, timeout=60)
    r.raise_for_status()
    d = r.json()
    etfs = [{"code": x["Code"], "name": x["Name"]} for x in d if x.get("Type") == "ETF"]
    return etfs


def fetch_bulk(date):
    r = requests.get(f"{EODHD_BASE}/eod-bulk-last-day/US",
                     params={"api_token": EODHD_KEY, "fmt": "json", "date": date}, timeout=120)
    r.raise_for_status()
    d = r.json()
    if not isinstance(d, list) or len(d) < 1000:
        return {}
    return {x["code"]: x for x in d if "code" in x}


# ── yfinance 补充 (缓存断点续跑) ──
def yf_infos(tickers, asof):
    import yfinance as yf
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"etf_yf_info_{asof}.pkl")
    info = {}
    if os.path.exists(cache):
        info = pickle.load(open(cache, "rb"))
    print(f"[YF] 已缓存 {len(info)}/{len(tickers)}, 继续拉其余 ...")
    for i, t in enumerate(tickers):
        if t in info:
            continue
        try:
            inf = yf.Ticker(t).info or {}
            info[t] = {
                "name": inf.get("shortName") or inf.get("longName"),
                "category": inf.get("category"),
                "fundFamily": inf.get("fundFamily"),
                "totalAssets": inf.get("totalAssets"),
            }
        except Exception:
            info[t] = {}
        if (i + 1) % 25 == 0:
            print(f"  yf {i+1}/{len(tickers)}")
            pickle.dump(info, open(cache, "wb"))
        time.sleep(0.25)
    pickle.dump(info, open(cache, "wb"))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=300)
    ap.add_argument("--asof", default="2026-06-26")
    args = ap.parse_args()

    print(f"=== US ETF Top {args.top} (asof={args.asof}) ===")
    etfs = fetch_etf_list()
    print(f"[EODHD] ETF 列表 {len(etfs)} 只")
    bulk = fetch_bulk(args.asof)
    print(f"[EODHD] bulk 成交量 {len(bulk)} 条")

    rows = []
    for e in etfs:
        b = bulk.get(e["code"])
        if not b or not b.get("volume") or not b.get("close"):
            continue
        amt = float(b["close"]) * float(b["volume"])   # 单日成交额 USD
        rows.append({"ticker": e["code"], "name_eodhd": e["name"],
                     "close": float(b["close"]), "volume": float(b["volume"]), "amt": amt})
    df = pd.DataFrame(rows).sort_values("amt", ascending=False).reset_index(drop=True)
    top = df.head(args.top).copy()
    print(f"[Liquidity] 取 top {len(top)}, 成交额 {top['amt'].min()/1e6:.0f}M ~ {top['amt'].max()/1e9:.1f}B USD")

    infos = yf_infos(top["ticker"].tolist(), args.asof)
    top["yf_name"] = top["ticker"].map(lambda t: (infos.get(t) or {}).get("name"))
    top["category"] = top["ticker"].map(lambda t: (infos.get(t) or {}).get("category"))
    top["fundFamily"] = top["ticker"].map(lambda t: (infos.get(t) or {}).get("fundFamily"))
    top["aum"] = top["ticker"].map(lambda t: (infos.get(t) or {}).get("totalAssets"))
    # 用 EODHD 完整名称 (yfinance shortName 被截断~30字符, 会丢失 "3X/2X" 等关键词)
    top["name"] = top["name_eodhd"]

    top["benchmark"] = top.apply(lambda r: benchmark_for(r["ticker"], r["name"], r["category"]), axis=1)
    top["type"] = top.apply(lambda r: is_active(r["ticker"], r["name"], r["fundFamily"]), axis=1)

    # 去重: 同 benchmark 只留流动性最高一只; benchmark 来自 category 兜底(含"—"或category原文)的不去重
    dedup_keep = set()
    deduped = []
    for bench, g in top.groupby("benchmark"):
        g = g.sort_values("amt", ascending=False)
        first = g.iloc[0]
        if bench in ("—",) or bench == first.get("category"):  # 兜底组不去重
            for _, r in g.iterrows():
                deduped.append(r)
        else:
            deduped.append(first)
    out = pd.DataFrame(deduped).sort_values("amt", ascending=False).reset_index(drop=True)

    # 输出
    os.makedirs(OUT_DIR, exist_ok=True)
    out["amt_yi"] = (out["amt"] / 1e8).round(2)   # 亿USD
    out["aum_yi"] = (out["aum"].fillna(0) / 1e8).round(0).astype(int)
    tbl = out[["ticker", "name", "benchmark", "amt_yi", "aum_yi", "type"]].rename(
        columns={"ticker": "Ticker", "name": "Name", "benchmark": "Benchmark",
                 "amt_yi": "Liquidity(亿USD)", "aum_yi": "AUM(亿USD)", "type": "Passive/Active"})

    md = os.path.join(OUT_DIR, f"us_etf_top{args.top}.md")
    csv = os.path.join(OUT_DIR, f"us_etf_top{args.top}.csv")
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# US ETF Top {args.top} (asof {args.asof})\n\n")
        f.write(f"按基准去重后 {len(tbl)} 只 (同基准只留流动性最高一只; 兜底组保留全部)。\n\n")
        f.write(tbl.to_markdown(index=False))
        f.write("\n")
    tbl.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"\n[Done] {len(tbl)} 只 → {md}\n{csv}")
    print(tbl.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
