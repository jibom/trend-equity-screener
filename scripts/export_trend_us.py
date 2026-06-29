"""导出美股趋势五部曲到 public/data/trend_hotspot_us.json, 供网页「美股-趋势」tab 使用。

复用 src/us_trend.py:
  hotspot_scores(pool, raw) -> (df_full, ind_rank, sec_rank)
  analyze_pullback(g, ...)  -> Part4 回调买点候选
  screen_surge(raw, pool, exclude) -> Part3 异动放量
  compute_share_rotation(raw, pool, "Industry") -> Part5 资金轮动

JSON 字段名刻意与港股 trend_hotspot_hk.json 对齐 (sub_industry=Industry 名, hottest_sub=最热行业,
金额单位亿USD), 前端 renderUs 可复用 renderTrend 的表格/钻取/快捷键逻辑, 仅按 .HK/.US 路由。

用法: python scripts/export_trend_us.py --asof 2026-06-18
"""
from __future__ import annotations
import os, sys, json, argparse, math, re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import us_trend as ust   # 模块级已设 utf-8 stdout

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_PATH = os.path.join(PROJECT_DIR, "public", "data", "trend_hotspot_us.json")
FUND_CSV = os.path.join(PROJECT_DIR, "configs", "us_universe_fundamentals.csv")
_FUND = {}   # ticker -> {sales_growth, earn_growth, roe} (fraction), main() 载入


def _load_fundamentals():
    if not os.path.exists(FUND_CSV):
        print("[Fund] 未找到 us_universe_fundamentals.csv (基本面列将为空; 跑 build_us_fundamentals.py 生成)")
        return
    import pandas as pd
    fdf = pd.read_csv(FUND_CSV).fillna("")

    def _f(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return None
    for _, fr in fdf.iterrows():
        _FUND[fr["Ticker"]] = {"sales_growth": _f(fr["SalesGrowth"]),
                               "earn_growth": _f(fr["EarnGrowth"]),
                               "roe": _f(fr["ROE"])}
    print(f"[Fund] 载入 {len(_FUND)} 只基本面")


# ── 名称/行业缩写 (前端表格可读性) ──────────────────────────
_ABBR = {
    'Semiconductors': 'Semis', 'Semiconductor': 'Semi', 'Manufacturing': 'Mfg', 'Manufacturers': 'Mfr',
    'Technologies': 'Tech', 'Technology': 'Tech', 'Equipment': 'Equip', 'Materials': 'Mat', 'Material': 'Mat',
    'Communication': 'Comm', 'Communications': 'Comm', 'Services': 'Svc', 'Service': 'Svc',
    'Diversified': 'Div', 'Integrated': 'Intg', 'Renewable': 'Ren', 'Construction': 'Constr',
    'Engineering': 'Eng', 'Entertainment': 'Entertain', 'Biotechnology': 'Biotech', 'Pharmaceuticals': 'Pharma',
    'Automotive': 'Auto', 'Logistics': 'Log', 'Financial': 'Fin', 'Insurance': 'Ins',
    'Instruments': 'Instr', 'Components': 'Comp', 'Technical': 'Tech', 'Scientific': 'Sci',
    'Aerospace': 'Aero', 'Defense': 'Def', 'Industrial': 'Indl', 'International': 'Intl',
    'Exploration': 'Expl', 'Development': 'Dev', 'Solutions': 'Sol', 'Networks': 'Net', 'Systems': 'Sys',
    'Refining': 'Refg', 'Marketing': 'Mktg', 'Information': 'Info', 'Infrastructure': 'Infra',
    'Application': 'Appl', 'Machinery': 'Mach', 'Cyclical': 'Cyc', 'Defensive': 'Def',
    'Holdings': 'Hldg', 'Corporation': 'Corp', 'Incorporated': 'Inc',
}
_NAME_SUFFIX = re.compile(
    r'\s*,?\s*(?:'
    r'Inc\.?|Corp\.?|Corporation|Incorporated|Co\.?|Ltd\.?|Limited|plc|PLC|S\.A\.|S\.p\.A\.|N\.V\.|SE|'
    r'Group|Holdings|Holding|Company|Worldwide|'
    r'American Depositary (?:Shares|Receipt)|ADR|'
    r'Class [A-Z]\s+(?:Common Stock|Ordinary Shares)?|Common Stock|Ordinary Shares'
    r')\.?\s*$', re.I)


def _abbr(s):
    if not s:
        return s
    def rep(m):
        w = m.group(0)
        short = _ABBR.get(w) or _ABBR.get(w.capitalize()) or _ABBR.get(w.lower(), )
        if not short:
            return w
        return short if (not w[0].isupper()) else (short[0].upper() + short[1:])
    return re.sub(r'[A-Za-z]+', rep, s)


def short_industry(s):
    s = _abbr(s)
    return (s[:24] + '…') if s and len(s) > 24 else s


def short_sector(s):
    return _abbr(s) if s else s


def short_name(name):
    if not name:
        return name
    n = name.strip()
    for _ in range(5):
        new = _NAME_SUFFIX.sub('', n).strip()
        if new == n or not new:
            break
        n = new
    n = _abbr(n)
    return (n[:20] + '…') if len(n) > 20 else n



def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return v
    if isinstance(v, float):
        return round(v, 3)
    return v


def rec(d):
    return {k: clean(v) for k, v in d.items()}


def part1_records(rank_df, name_key, sector_map=None):
    name_fn = short_industry if name_key == "sub_industry" else short_sector
    out = []
    for idx, row in rank_df.iterrows():
        d = {
            name_key: name_fn(idx),
            "n": int(row["n"]),
            "pct_high_250": row.get("mean_pct_high_250"),
            "breadth_250": row.get("breadth_250%"),
            "breadth_60": row.get("breadth_60%"),
            "amt_surge": row.get("mean_amt_surge"),
            "amt_rank": row.get("mean_amt_rank"),
            "amt_1d_yi": row.get("总成交金额_1日_亿USD"),
            "amt_5d_yi": row.get("总成交金额_5日均值_亿USD"),
            "share_1d": row.get("占比_1日%"),
            "share_5d": row.get("占比_5日%"),
            "share5_pctile200": row.get("share5_pctile200"),
            "ret_60": row.get("mean_ret_60"),
            "composite": row.get("composite"),
        }
        if sector_map is not None:
            d["sector"] = short_sector(sector_map.get(idx))
        out.append(rec(d))
    return out


def stock_record(r, with_nh=False):
    f = _FUND.get(r["Ticker"], {})
    d = {
        "ticker": r["Ticker"], "name": short_name(r["Name"]),
        "sub_industry": short_industry(r["Industry"]), "sector": short_sector(r.get("Sector")),
        "close": r.get("close"), "pct_high_250": r.get("pct_high_250"),
        "amt_surge": r.get("amt_surge"), "amt_rank": r.get("amt_rank_pct"),
        "ret_60": r.get("ret_60"),
        "ma_aligned": int(r["ma_aligned"]) if r.get("ma_aligned") is not None else 0,
        "hotness": r.get("hotness"),
        "sales_growth": f.get("sales_growth"), "earn_growth": f.get("earn_growth"), "roe": f.get("roe"),
    }
    if with_nh:
        d["pct_high_126"] = r.get("pct_high_126")
        d["nh_ratio_126"] = r.get("nh_ratio_126")
    return rec(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="截止日 YYYY-MM-DD; 默认 universe LatestDate")
    args = ap.parse_args()
    asof = args.asof or str(pd_read_latest())

    pool = ust.load_pool()
    print(f"=== export_trend_us (asof={asof}) ===\n[Pool] {len(pool)} 只")
    _load_fundamentals()
    raw = ust.fetch_all(pool, asof)
    print(f"[Fetch] {raw['code'].nunique()} 只有数据")
    asof = str(raw["date"].max())   # 用实际最新数据日, 避免 asof=今天 但数据 T-1 的不符

    df_full, ind_rank, sec_rank = ust.hotspot_scores(pool, raw)
    hottest_ind = ind_rank.index[0] if len(ind_rank) else "N/A"
    top10_ind = set(ind_rank.head(10).index)   # Top10 行业 (综合分), Part2/3/4/all_stocks 限定其内
    print(f"[Hotspot] {len(df_full)} 只个股, 最热行业={hottest_ind}; Top10行业限定后续筛选")

    ind_to_sector = df_full.groupby("Industry")["Sector"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    part1 = {
        "sub_industries": part1_records(ind_rank.head(10), "sub_industry", sector_map=ind_to_sector),
        "sectors": part1_records(sec_rank, "sector"),
    }

    # Part1c: 6月新高 (全市场, pct_high_126≥0.98, 126交易日≈6个月) — 拆首次/持续, 互斥
    nh6m = df_full[df_full["pct_high_126"] >= 0.98]
    # 1c1 首次新高: 近6月新高天数占比<4%(刚突破), 不要求多头(早期突破/低位N倍股候选), 低位优先
    first_nh = nh6m[nh6m["nh_ratio_126"] < 0.04].sort_values("pct_high_250", ascending=True).head(20)
    # 1c2 持续新高: 占比≥4% + 多头排列(ma_stack), 排除已在1c1的, hotness 排序
    sust_nh = nh6m[(nh6m["nh_ratio_126"] >= 0.04) & (nh6m["ma_stack"] == 1)
                   & (~nh6m["Ticker"].isin(first_nh["Ticker"]))]
    sust_nh = sust_nh.sort_values("hotness", ascending=False).head(20)
    part1c1 = [stock_record(r, with_nh=True) for _, r in first_nh.iterrows()]
    part1c2 = [stock_record(r, with_nh=True) for _, r in sust_nh.iterrows()]

    # Part2: 趋势个股 (60日>10%, 且在 Top10 行业内) hotness 排序 ≤20
    top = df_full[(df_full["ret_60"] > 0.10) & (df_full["Industry"].isin(top10_ind))]
    top = top.sort_values("hotness", ascending=False).head(20)
    part2 = [stock_record(r) for _, r in top.iterrows()]

    # Part4: 回调买点 (先算, 用于 Part3 排除)
    hotness_map = df_full.set_index("Ticker")["hotness"]
    ind_comp = ind_rank["composite"]
    part4 = []
    for code, g in raw.groupby("code"):
        gg = ust.forward_adjust_group(g)
        meta = pool[pool["Ticker"] == code]
        if meta.empty:
            continue
        meta = meta.iloc[0]
        if meta["Industry"] not in top10_ind:
            continue
        r = ust.analyze_pullback(gg, code, meta["Name"], meta["Industry"], meta["Sector"])
        if r:
            f = _FUND.get(r["Ticker"], {})
            part4.append(rec({
                "ticker": r["Ticker"], "name": short_name(r["Name"]), "sub_industry": short_industry(r["Industry"]),
                "close": r["Close"], "retrace": r["Retrace"],
                "ind_comp": round(float(ind_comp.get(r["Industry"], 0)), 3),
                "hotness": round(float(hotness_map.get(r["Ticker"], 0)), 3),
                "doji": r["Doji"], "shrink": r["Shrink"], "entangle": r["Entangle"],
                "kdj_div": r["KDJdiv"], "at_support": r["AtSupport"], "nsig": r["NSig"],
                "sales_growth": f.get("sales_growth"), "earn_growth": f.get("earn_growth"), "roe": f.get("roe"),
            }))
    part4.sort(key=lambda x: (x["ind_comp"], x["hotness"]), reverse=True)

    # Part3: 异动放量, 排除 Part2/4
    exclude = set(p["ticker"] for p in part2) | set(p["ticker"] for p in part4)
    surge = ust.screen_surge(raw, pool, exclude=exclude)
    surge = [r for r in surge if r["Industry"] in top10_ind]
    part3 = [rec({
        "ticker": r["Ticker"], "name": short_name(r["Name"]), "sub_industry": short_industry(r["Industry"]),
        "close": r["Close"], "ret_1d": r["Ret1d"], "ret_3d": r["Ret3d"],
        "vol_ratio": r["VolRatio"], "trigger": r["Trigger"],
        "hotness": round(float(hotness_map.get(r["Ticker"], 0)), 3),
    }) for r in surge]

    # Part5: 资金轮动 (Industry 维度)
    rot = ust.compute_share_rotation(raw, pool, "Industry", lookback=10)
    part5 = [rec({
        "sub_industry": short_industry(r["Industry"]), "n": int(r["n"]),
        "pctile_now": r["pctile_now"], "pctile_ago": r["pctile_ago"],
        "delta10": r["delta10"], "amt5_now_yi": r["amt5_now_yi"],
        "amt5_chg_pct": r["amt5_chg_pct"], "flag": r["flag"],
    }) for _, r in rot.iterrows()]
    rotation_alerts = {
        "lose": [p for p in part5 if p["flag"] == "lose"],
        "gain": [p for p in part5 if p["flag"] == "gain"],
    }

    all_stocks = [rec({
        "ticker": r["Ticker"], "name": short_name(r["Name"]), "sub_industry": short_industry(r["Industry"]),
        "sector": short_sector(r["Sector"]), "close": r.get("close"), "hotness": r.get("hotness"),
        "ret_60": r.get("ret_60"), "pct_high_250": r.get("pct_high_250"),
        "ma_aligned": int(r["ma_aligned"]) if r.get("ma_aligned") is not None else 0,
    }) for _, r in df_full[df_full["Industry"].isin(top10_ind)].sort_values("hotness", ascending=False).iterrows()]

    payload = {
        "date": asof, "market": "US",
        "hottest_sub": hottest_ind,
        "part1": part1, "part1c1": part1c1, "part1c2": part1c2, "part2": part2, "part3": part3, "part4": part4,
        "part5": part5, "rotation_alerts": rotation_alerts, "all_stocks": all_stocks,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[Export] -> {OUT_PATH}")
    print(f"  part1: {len(part1['sub_industries'])} 行业, {len(part1['sectors'])} GICS板块")
    print(f"  part1c1: {len(part1c1)} 首次新高 | part1c2: {len(part1c2)} 持续新高 | part2: {len(part2)} 趋势个股 | part3: {len(part3)} 异动 | part4: {len(part4)} 回调买点")
    print(f"  part5: {len(part5)} 行业轮动 (🔻lose={len(rotation_alerts['lose'])} 🔺gain={len(rotation_alerts['gain'])})")
    print(f"  all_stocks: {len(all_stocks)}")


def pd_read_latest():
    import pandas as pd
    return pd.read_csv(ust.UNIVERSE_CSV)["LatestDate"].iloc[0]


if __name__ == "__main__":
    main()
