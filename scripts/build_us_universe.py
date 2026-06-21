"""构建美股趋势分析基准名单 (top 600 by turnover, 含ADR 剔ETF, yfinance 两级GICS).

数据源:
  - EODHD: exchange-symbol-list/US (Type==Common Stock, 含ADR, 自然排除ETF/FUND)
  - EODHD: eod-bulk-last-day/US?date=...  (近N交易日全市场 OHLCV; 取中位成交量, 成交量干净)
  - yfinance: regularMarketPrice + sector + industry + marketCap  (fundamentals 403, 改用 yfinance)

成交金额 = yfinance真实价 × EODHD中位成交量  (EODHD对个别仙股有价小数点错位, 如NMDX,
  用yf价校正并剔除|eod/yf|发散>3x的脏票). 排名按此成交金额降序取前600.

输出:
  - configs/us_universe.csv   主名单 (引擎 load_pool 用)
  - public/data/us_universe.json  前端用

yfinance 取数支持断点续跑 (每50只落盘). 用法: python scripts/build_us_universe.py [--top 600] [--days 5] [--pool 900]
"""
from __future__ import annotations
import os, sys, json, time, argparse, csv, statistics
from datetime import date, timedelta
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

EODHD_BASE = "https://eodhd.com/api"


def _load_eodhd_key():
    k = os.getenv("EODHD_KEY")
    if k:
        return k
    local = os.path.join(os.path.dirname(__file__), "..", "configs", "eodhd_key.local")
    if os.path.exists(local):
        return open(local, encoding="utf-8").read().strip()
    raise RuntimeError("EODHD_KEY 未设置: 设环境变量 EODHD_KEY, 或写入 configs/eodhd_key.local (gitignored)")


EODHD_KEY = _load_eodhd_key()

OUT_CSV = os.path.join(PROJECT_DIR, "configs", "us_universe.csv")
OUT_JSON = os.path.join(PROJECT_DIR, "public", "data", "us_universe.json")
CACHE_CSV = os.path.join(PROJECT_DIR, "configs", "us_universe_partial.csv")


def eodhd(path, **params):
    params = {"api_token": EODHD_KEY, "fmt": "json", **params}
    r = requests.get(f"{EODHD_BASE}/{path}", params=params, timeout=180)
    r.raise_for_status()
    return r.json()


def fetch_common_stocks():
    d = eodhd("exchange-symbol-list/US")
    rows = []
    for x in d:
        if x.get("Type") != "Common Stock":
            continue
        code = x.get("Code", "")
        if not code or code.startswith("^"):
            continue
        rows.append({"code": code, "name": x.get("Name", ""), "exchange": x.get("Exchange", "")})
    return rows


def fetch_bulk(days=5):
    """近N交易日的全市场OHLCV, 每只票取中位成交量 + 中位成交金额(用EODHD价, 仅候选用) + 最近日close."""
    today = date.today()
    valid = []
    seen = 0
    for back in range(0, 20):
        ds = (today - timedelta(days=back)).strftime("%Y-%m-%d")
        try:
            rows = eodhd("eod-bulk-last-day/US", date=ds)
        except Exception as e:
            print(f"  bulk {ds} err: {e}"); continue
        if len(rows) < 20000:
            print(f"  bulk {ds} skip (rows={len(rows)})"); continue
        print(f"  bulk {ds} ok (rows={len(rows)})")
        valid.append((ds, rows)); seen += 1
        if seen >= days:
            break
    if not valid:
        raise RuntimeError("no valid bulk trading day in last 20 days")
    valid = valid[:days]

    agg = {}  # code -> dict(vol_list, to_list)
    for ds, rows in valid:
        for x in rows:
            code = x.get("code"); vol = x.get("volume") or 0; close = x.get("close")
            if not code or not close or vol <= 0:
                continue
            a = agg.setdefault(code, {"vol": [], "to": []})
            a["vol"].append(float(vol)); a["to"].append(float(close) * float(vol))
    # 最近一日 = valid[0] (valid 按最近→最旧排序), 用于 latest_close/latest_date
    latest_rows = {x["code"]: x for x in valid[0][1]}
    latest_date = valid[0][0]
    recs = []
    for code, a in agg.items():
        lr = latest_rows.get(code)
        recs.append({"code": code, "med_volume": statistics.median(a["vol"]),
                     "med_turnover_eod": statistics.median(a["to"]),
                     "latest_close": float(lr["close"]) if lr and lr.get("close") else None,
                     "latest_date": latest_date})
    return recs, latest_date


def is_adr(name):
    n = (name or "").lower()
    return "adr" in n or "american depositary" in n


def _load_cache():
    if not os.path.exists(CACHE_CSV):
        return {}
    done = {}
    with open(CACHE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done[row["code"]] = {"sector": row.get("sector", ""), "industry": row.get("industry", ""),
                                 "market_cap": row.get("market_cap", "") or "",
                                 "yf_name": row.get("yf_name", ""),
                                 "yf_price": row.get("yf_price", "") or ""}
    return done


def _save_cache(done):
    os.makedirs(os.path.dirname(CACHE_CSV), exist_ok=True)
    with open(CACHE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "sector", "industry", "market_cap", "yf_name", "yf_price"])
        for c, r in done.items():
            w.writerow([c, r["sector"], r["industry"], r["market_cap"], r["yf_name"], r["yf_price"]])


def yf_classify(codes, done):
    """取 yf_price(fast_info, 快) + sector/industry/marketCap(get_info). 缺啥补啥, 断点续跑."""
    import yfinance as yf
    out = dict(done)
    # 需要补: 缺 yf_price 的 (用于校正/排名)
    need_price = [c for c in codes if not out.get(c, {}).get("yf_price")]
    need_info = [c for c in codes if not (out.get(c, {}).get("sector") or out.get(c, {}).get("industry"))]
    print(f"[yfinance] price to fetch: {len(need_price)}; info(sector/industry) to fetch: {len(need_info)}")

    ok = 0
    for i, code in enumerate(need_price):
        row = out.get(code, {"sector": "", "industry": "", "market_cap": "", "yf_name": "", "yf_price": ""})
        p = None; mc = None
        # fast_info (轻量, 属性访问); 失败回退 get_info
        for attempt in range(2):
            try:
                fi = yf.Ticker(code).fast_info
                p = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
                mc = getattr(fi, "market_cap", None)
                if p:
                    break
                time.sleep(0.4)
            except Exception:
                time.sleep(0.6)
        if not p:  # 回退 get_info
            try:
                info = yf.Ticker(code).get_info()
                p = info.get("regularMarketPrice") or info.get("currentPrice")
                mc = mc or info.get("marketCap")
                if info and not row["sector"]:
                    row["sector"] = info.get("sector") or ""
                    row["industry"] = info.get("industry") or ""
                    row["yf_name"] = info.get("shortName") or info.get("longName") or ""
            except Exception:
                pass
        if p:
            row["yf_price"] = round(float(p), 6); ok += 1
        if mc and not row["market_cap"]:
            row["market_cap"] = int(mc)
        out[code] = row
        time.sleep(0.25)  # 防 yfinance 限流
        if (i + 1) % 100 == 0:
            print(f"  price {i+1}/{len(need_price)} (ok={ok})"); _save_cache(out)
    _save_cache(out)
    print(f"  price done: {ok}/{len(need_price)} got price")

    for i, code in enumerate(need_info):
        row = out.get(code, {"sector": "", "industry": "", "market_cap": "", "yf_name": "", "yf_price": ""})
        if not row["yf_price"]:  # fast_info 都拿不到的票, get_info 多半也拿不到, 跳过省时
            continue
        if row["sector"] or row["industry"]:  # price回退时已取到, 跳过
            continue
        for attempt in range(2):
            try:
                info = yf.Ticker(code).get_info()
                if info:
                    row["sector"] = info.get("sector") or ""
                    row["industry"] = info.get("industry") or ""
                    mc = info.get("marketCap")
                    if mc:
                        row["market_cap"] = int(mc)
                    row["yf_name"] = info.get("shortName") or info.get("longName") or ""
                break
            except Exception:
                if attempt == 0:
                    time.sleep(0.8)
        out[code] = row
        if (i + 1) % 50 == 0:
            print(f"  info {i+1}/{len(need_info)}"); _save_cache(out)
    _save_cache(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--pool", type=int, default=900, help="候选池大小(清洗前)")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    print(f"=== build US universe (top {args.top}, {args.days}-day median vol, pool {args.pool}) ===")

    stocks = fetch_common_stocks()
    smap = {s["code"]: s for s in stocks}
    print(f"[Symbols] {len(stocks)} Common Stock (含ADR, 排除ETF)")

    bulks, latest_date = fetch_bulk(args.days)
    print(f"[Bulk] {len(bulks)} 只有成交, latest_date={latest_date}")

    rows = []
    for b in bulks:
        s = smap.get(b["code"])
        if not s:
            continue
        rows.append({"code": b["code"], "name": s["name"], "exchange": s["exchange"],
                     "is_adr": is_adr(s["name"]), "med_volume": b["med_volume"],
                     "eod_close": b["latest_close"], "eod_turnover": b["med_turnover_eod"],
                     "latest_date": b["latest_date"]})
    # 候选池: 按EODHD中位成交金额降序取 pool (含可能的脏票)
    rows.sort(key=lambda x: x["eod_turnover"], reverse=True)
    cand = rows[:args.pool]
    print(f"[Pool] {len(cand)} candidates by EOD turnover; "
          f"#1 {cand[0]['code']} ${cand[0]['eod_turnover']/1e9:.1f}B")

    done = {} if args.refresh_cache else _load_cache()
    codes = [r["code"] for r in cand]
    done = yf_classify(codes, done)

    # 清洗 + 重排: 用 yf_price × med_volume
    clean = []
    for r in cand:
        c = done.get(r["code"], {})
        yp = c.get("yf_price")
        if not yp or float(yp) <= 0:
            continue  # yfinance 无价 (多半仙股/退市/脏票), 弃
        yp = float(yp)
        ratio = r["eod_close"] / yp if yp else 0
        if ratio > 3.0 or ratio < 0.333:  # EODHD价与yf价发散>3x → EODHD小数点错位等脏数据, 弃
            continue
        clean.append({"code": r["code"], "name": r["name"], "exchange": r["exchange"],
                      "is_adr": r["is_adr"], "sector": c.get("sector", ""),
                      "industry": c.get("industry", ""), "market_cap": c.get("market_cap", ""),
                      "yf_name": c.get("yf_name", ""), "yf_price": yp,
                      "med_volume": r["med_volume"],
                      "turnover_usd": yp * r["med_volume"], "latest_date": r["latest_date"]})
    clean.sort(key=lambda x: x["turnover_usd"], reverse=True)
    top = clean[:args.top]
    print(f"[Clean] {len(clean)} valid after price-check; top {len(top)} by yf_price×vol; "
          f"#1 {top[0]['code']} ${top[0]['turnover_usd']/1e9:.1f}B, "
          f"#{args.top} {top[-1]['code']} ${top[-1]['turnover_usd']/1e6:.0f}M")

    df = pd.DataFrame([{"Ticker": r["code"] + ".US", "Code": r["code"], "Name": r["name"],
                        "Exchange": r["exchange"], "ADR": r["is_adr"], "Sector": r["sector"],
                        "Industry": r["industry"], "MarketCap": r["market_cap"],
                        "Price": r["yf_price"], "AvgVolume": int(r["med_volume"]),
                        "AvgTurnoverUSD": round(r["turnover_usd"], 0),
                        "LatestDate": r["latest_date"]} for r in top])
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[CSV] -> {OUT_CSV} ({len(df)} rows)")

    n_sec = df["Sector"].replace("", pd.NA).dropna().nunique()
    n_ind = df["Industry"].replace("", pd.NA).dropna().nunique()
    n_miss = int((df["Sector"] == "").sum())
    print(f"[GICS] sector={n_sec} industry={n_ind} 未分类={n_miss} ADR={int(df['ADR'].sum())}")
    print("[Sector分布]"); print(df.groupby("Sector").size().sort_values(ascending=False).to_string())

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"date": latest_date, "top": args.top, "days": args.days, "count": len(df),
                   "stocks": df.to_dict(orient="records")}, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[JSON] -> {OUT_JSON}")


if __name__ == "__main__":
    main()
