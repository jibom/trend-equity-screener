"""合并 142 只 ETF + 用户跟踪清单 → 我在跟踪的 ETF (去债券).
复用 etf_screener 的 benchmark_for/is_active (扩展白名单), EODHD bulk (US流动性), yfinance (HK + 补充).
输出: output/us_etf_tracking.csv / .md
"""
from __future__ import annotations
import os, sys, io, re, time, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import requests, pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUT_DIR = os.path.join(PROJECT_DIR, "output")
EODHD_BASE = "https://eodhd.com/api"


def _key():
    k = os.getenv("EODHD_KEY")
    if k:
        return k
    local = os.path.join(PROJECT_DIR, "configs", "eodhd_key.local")
    return open(local, encoding="utf-8").read().strip()
EODHD_KEY = _key()

# 用户跟踪清单 (US + HK)
USER_US = ["KWEB","MCHI","CWEB","YINN","YANG","SRTY","IWM","UPRO","SDS","SPXU",
           "RSPU","RSPG","RSPC","RSPF","RSPR","RSPS","RSPN","RSPH","RSPM","RSPD","RSPT",
           "XBI","GLD","SLV","USO","GDX","NUGT","INDA","ITA","FNGS","ARKK","IBIT","VXX","UNG",
           "IFRA","DBA","AGIX","METV","URA","COPX","BOTZ","TAN","EWY","EWT","EWZ","EEM","IGV",
           "IBAT","IBLC","ICOP","ILF","ISTM","XT","VEGI"]
USER_HK = ["7552.HK", "7226.HK"]

# 扩展白名单 (新增 ticker → 基准)
TICKER_BENCH = {
    "CWEB": "China Internet 2x (Leveraged)", "YINN": "China 3x Leveraged", "YANG": "China 3x Inverse",
    "SRTY": "Russell 2000 3x Inverse", "UPRO": "S&P 500 3x", "SDS": "S&P 500 2x Inverse", "SPXU": "S&P 500 3x Inverse",
    "RSPU": "S&P 500 EW Utilities", "RSPG": "S&P 500 EW Energy", "RSPC": "S&P 500 EW Communication",
    "RSPF": "S&P 500 EW Financials", "RSPR": "S&P 500 EW Real Estate", "RSPS": "S&P 500 EW Consumer Staples",
    "RSPN": "S&P 500 EW Industrials", "RSPH": "S&P 500 EW Health Care", "RSPM": "S&P 500 EW Materials",
    "RSPD": "S&P 500 EW Consumer Disc", "RSPT": "S&P 500 EW Technology",
    "GDX": "Gold Miners", "NUGT": "Gold Miners 3x", "FNGS": "NYSE FANG+",
    "VXX": "Volatility (VIX)", "IFRA": "Infrastructure", "DBA": "Agriculture", "VEGI": "Agriculture",
    "AGIX": "AI (Public-Private)", "BOTZ": "Robotics/AI", "XT": "Exponential Tech",
    "METV": "Metaverse", "IBLC": "Blockchain", "TAN": "Solar", "IBAT": "Energy Storage",
    "ICOP": "Copper/Metals Mining", "ISTM": "Strategic Metals", "ILF": "Latin America",
    "7552.HK": "Hang Seng Tech", "7226.HK": "Hang Seng Tech",
}
NAME_BENCH = [
    (r"\b[23]x\b|ultrapro|daily\s+(bull|bear)|inverse\s+etf|\bshort\b.*etf|leveraged", "Leveraged/Inverse"),
    (r"semiconductor|semis?\b", "Semiconductor"),
    (r"s&p\s*500|sp\s*500|sp500", "S&P 500"),
    (r"nasdaq-?100|ndx-?100", "NASDAQ-100"), (r"russell\s*2000", "Russell 2000"),
    (r"russell\s*1000", "Russell 1000"), (r"russell\s*3000", "Russell 3000"),
    (r"dow\s*jones|djia|industrial\s*average", "Dow Jones"),
    (r"msci\s*eafe|developed\s*markets", "MSCI EAFE"), (r"msci\s*emerging|emerging\s*markets", "MSCI Emerging Markets"),
    (r"msci\s*acwi|all\s*country\s*world", "MSCI ACWI"),
    (r"gold\s*miner", "Gold Miners"), (r"gold\b", "Gold"), (r"silver\b", "Silver"),
    (r"crude\s*oil|wti|brent", "Crude Oil"), (r"natural\s*gas", "Natural Gas"),
    (r"agricultur", "Agriculture"), (r"infrastructure", "Infrastructure"),
    (r"solar|clean\s*energy", "Solar/Clean Energy"), (r"robotic|artificial\s*intelligence|\bai\b", "Robotics/AI"),
    (r"metaverse", "Metaverse"), (r"blockchain", "Blockchain"),
    (r"copper|metals\s*mining", "Copper/Metals Mining"), (r"strategic\s*metals", "Strategic Metals"),
    (r"latin\s*america", "Latin America"), (r"fang", "NYSE FANG+"),
    (r"vix|volatility", "Volatility (VIX)"),
    (r"china|csi", "China"), (r"taiwan", "Taiwan"), (r"brazil", "Brazil"), (r"korea", "Korea"),
    (r"india", "India"), (r"japan", "Japan"),
    (r"reit|real\s*estate", "Real Estate / REIT"), (r"dividend", "Dividend"),
]
ACTIVE_TICKERS = {"ARKK","ARKW","ARKG","ARKQ","ARKF","ARKX"}
ACTIVE_FAMILIES = ("ARK", "Capital Group")
ACTIVE_NAME_KW = ("active", "ark ", "arknxt")


def benchmark_for(ticker, name, category):
    if ticker in TICKER_BENCH:
        return TICKER_BENCH[ticker]
    n = (name or "").lower() if isinstance(name, str) else ""
    for pat, b in NAME_BENCH:
        if re.search(pat, n):
            return b
    return category if (isinstance(category, str) and category) else "—"


def is_active(ticker, name, ff):
    if ticker in ACTIVE_TICKERS:
        return "Active"
    ff = ff if isinstance(ff, str) else ""
    nm = name if isinstance(name, str) else ""
    s = f"{nm} {ff}".lower()
    if any(kw in s for kw in ACTIVE_NAME_KW):
        return "Active"
    if ff and any(fam.lower() in ff.lower() for fam in ACTIVE_FAMILIES):
        return "Active"
    return "Passive"


BOND_KW = ['bond','treasury','government','muni national','municipal','mbs','securitized',
           'bank loan','aggregate','high yield','tips','global bond','ultrashort','intermediate','long government']
SINGLE = {'Trading--Inverse Equity', 'Trading--Leveraged Equity'}


def is_bond(b):
    return any(k in str(b).lower() for k in BOND_KW)


def fetch_bulk(date):
    r = requests.get(f"{EODHD_BASE}/eod-bulk-last-day/US",
                     params={"api_token": EODHD_KEY, "fmt": "json", "date": date}, timeout=120)
    r.raise_for_status()
    d = r.json()
    return {x["code"]: x for x in d if "code" in x} if isinstance(d, list) and len(d) > 1000 else {}


def yf_batch(tickers):
    import yfinance as yf
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, "etf_yf_info_2026-06-26.pkl")  # 复用已有缓存
    info = {}
    if os.path.exists(cache):
        info = pickle.load(open(cache, "rb"))
    for t in tickers:
        if t in info:
            continue
        try:
            inf = yf.Ticker(t).info or {}
            info[t] = {"name": inf.get("shortName") or inf.get("longName"),
                       "category": inf.get("category"), "fundFamily": inf.get("fundFamily"),
                       "totalAssets": inf.get("totalAssets"),
                       "avgVolume": inf.get("averageVolume"), "price": inf.get("regularMarketPrice")}
        except Exception:
            info[t] = {}
        time.sleep(0.25)
    pickle.dump(info, open(cache, "wb"))
    return info


def main():
    asof = "2026-06-26"
    base = pd.read_csv(os.path.join(OUT_DIR, "us_etf_equity.csv"))   # 142 只
    print(f"[基础] {len(base)} 只")

    # 用户清单中已在 base 的
    have = set(base["Ticker"])
    new_us = [t for t in USER_US if t not in have]
    new_hk = [t for t in USER_HK if t not in have]
    print(f"[新增] US {len(new_us)} + HK {len(new_hk)}")

    bulk = fetch_bulk(asof)
    infos = yf_batch(new_us + new_hk)

    rows = []
    for t in new_us:
        b = bulk.get(t) or {}
        amt = (float(b["close"]) * float(b["volume"])) if b.get("close") and b.get("volume") else None
        if amt is None:  # EODHD bulk 没有 → 用 yfinance avgVol×price
            iv = (infos.get(t) or {}).get("avgVolume"); ip = (infos.get(t) or {}).get("price")
            amt = (iv * ip) if iv and ip else 0
        rows.append(_row(t, infos.get(t) or {}, amt, ccy="USD"))
    for t in new_hk:
        iv = (infos.get(t) or {}).get("avgVolume"); ip = (infos.get(t) or {}).get("price")
        amt_hkd = (iv * ip) if iv and ip else 0
        rows.append(_row(t, infos.get(t) or {}, amt_hkd / 7.8, ccy="HKD→USD"))   # HKD→USD
    newdf = pd.DataFrame(rows)
    allf = pd.concat([base, newdf], ignore_index=True).fillna(0)
    # 去债券 + 去 single-stock + 去 '—'
    allf = allf[~allf["Benchmark"].apply(is_bond) & ~allf["Benchmark"].isin(SINGLE) & (allf["Benchmark"] != "—")]
    allf = allf.sort_values(["Benchmark", "Liquidity(亿USD)"], ascending=[True, False]).reset_index(drop=True)
    allf.to_csv(os.path.join(OUT_DIR, "us_etf_tracking.csv"), index=False, encoding="utf-8-sig")
    print(f"[合并去债券后] {len(allf)} 只 | Active {int((allf['Passive/Active']=='Active').sum())}")
    print(allf[["Ticker","Name","Benchmark","Liquidity(亿USD)","Passive/Active"]].head(50).to_string(index=False))


def _row(t, inf, amt_usd, ccy):
    name = inf.get("name") or t
    return {"Ticker": t, "Name": name[:40],
            "Benchmark": benchmark_for(t, name, inf.get("category")),
            "Liquidity(亿USD)": round(amt_usd / 1e8, 2) if amt_usd else 0,
            "AUM(亿USD)": round((inf.get("totalAssets") or 0) / 1e8 / (1 if ccy == "USD" else 7.8), 0),
            "Passive/Active": is_active(t, name, inf.get("fundFamily"))}


if __name__ == "__main__":
    main()
