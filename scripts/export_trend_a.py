"""导出A股趋势四部曲到 public/data/trend_hotspot_a.json, 供网页「A股-趋势」tab 使用。

复用 src/a_trend.py (其透出 pullback_buypoint.hotspot_scores/analyze/screen_surge +
sector_cluster.compute_share_rotation; 取数 jianxin ashareeodprices, Wind schema, 单位千元).
行业: 申万三级=SubIndustry, 申万一级=Sector. Part1 各取 Top10; Part2/3/4 限定 Top10 申万三级内.
Part2 趋势个股 30 只. 金额单位亿元(人民币, 千元/1e5).

JSON 字段名与港股/美股对齐 (sub_industry/sector/hottest_sub), 前端 renderA 复用 renderUs 逻辑.
用法: python scripts/export_trend_a.py [--asof 2026-06-18]
"""
from __future__ import annotations
import os, sys, json, argparse, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import a_trend as at   # 透传 utf-8 stdout (经 sector_cluster)

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "public", "data", "trend_hotspot_a.json")
TOP_SUB = 10   # Part1 申万三级 Top N
TOP_SEC = 10   # Part1 申万一级 Top N
TOP_STOCKS = 30  # Part2 趋势个股数


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
    out = []
    for idx, row in rank_df.iterrows():
        d = {
            name_key: idx, "n": int(row["n"]),
            "pct_high_250": row.get("mean_pct_high_250"),
            "breadth_250": row.get("breadth_250%"), "breadth_60": row.get("breadth_60%"),
            "amt_surge": row.get("mean_amt_surge"), "amt_rank": row.get("mean_amt_rank"),
            "amt_1d_yi": row.get("总成交金额_1日_亿"), "amt_5d_yi": row.get("总成交金额_5日均值_亿"),
            "share_1d": row.get("占比_1日%"), "share_5d": row.get("占比_5日%"),
            "share5_pctile200": row.get("share5_pctile200"), "ret_60": row.get("mean_ret_60"),
            "composite": row.get("composite"),
        }
        if sector_map is not None:
            d["sector"] = sector_map.get(idx)
        out.append(rec(d))
    return out


def stock_record(r, with_nh=False):
    d = {
        "ticker": r["Ticker"], "name": r["Name"],
        "sub_industry": r["SubIndustry"], "sector": r.get("Sector"),
        "close": r.get("close"), "pct_high_250": r.get("pct_high_250"),
        "amt_surge": r.get("amt_surge"), "amt_rank": r.get("amt_rank_pct"),
        "ret_60": r.get("ret_60"),
        "ma_aligned": int(r["ma_aligned"]) if r.get("ma_aligned") is not None else 0,
        "hotness": r.get("hotness"),
    }
    if with_nh:
        d["pct_high_126"] = r.get("pct_high_126")
        d["nh_ratio_126"] = r.get("nh_ratio_126")
    return rec(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="截止日; 默认 DB 最新交易日")
    args = ap.parse_args()
    asof = args.asof or at.latest_asof()

    pool = at.load_pool()
    print(f"=== export_trend_a (asof={asof}) ===\n[Pool] {len(pool)} 只")
    raw = at.fetch_all(pool["Ticker"].tolist(), asof)
    print(f"[Fetch] {raw['code'].nunique()} 只有数据")

    df_full, sub_rank, sec_rank = at.hotspot_scores(pool, raw)
    hottest_sub = sub_rank.index[0] if len(sub_rank) else "N/A"
    top10_sub = set(sub_rank.head(TOP_SUB).index)
    print(f"[Hotspot] {len(df_full)} 只个股, 最热申万三级={hottest_sub}; 后续限定 Top{TOP_SUB} 申万三级内")

    sub_to_sector = df_full.groupby("SubIndustry")["Sector"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    part1 = {
        "sub_industries": part1_records(sub_rank.head(TOP_SUB), "sub_industry", sector_map=sub_to_sector),
        "sectors": part1_records(sec_rank.head(TOP_SEC), "sector"),
    }

    # Part2: 趋势个股 (60日>10%, Top10申万三级内) hotness 排序, 30只
    top = df_full[(df_full["ret_60"] > 0.10) & (df_full["SubIndustry"].isin(top10_sub))]
    top = top.sort_values("hotness", ascending=False).head(TOP_STOCKS)
    part2 = [stock_record(r) for _, r in top.iterrows()]

    # Part1c: 6月新高 (pct_high_126≥0.98) — 拆首次/持续, 互斥 (全市场, 不限Top10申万三级)
    nh6m = df_full[df_full["pct_high_126"] >= 0.98]
    first_nh = nh6m[nh6m["nh_ratio_126"] < 0.04].sort_values("hotness", ascending=False).head(20)
    sust_nh = nh6m[(nh6m["nh_ratio_126"] >= 0.04) & (nh6m["ma_stack"] == 1)
                   & (~nh6m["Ticker"].isin(first_nh["Ticker"]))]
    sust_nh = sust_nh.sort_values("hotness", ascending=False).head(20)
    part1c1 = [stock_record(r, with_nh=True) for _, r in first_nh.iterrows()]
    part1c2 = [stock_record(r, with_nh=True) for _, r in sust_nh.iterrows()]

    # Part4: 回调买点 (Top10申万三级内)
    hotness_map = df_full.set_index("Ticker")["hotness"]
    sub_comp = sub_rank["composite"]
    part4 = []
    for code, g in raw.groupby("code"):
        gg = at.forward_adjust_group(g)
        meta = pool[pool["Ticker"] == code]
        if meta.empty:
            continue
        meta = meta.iloc[0]
        if meta["SubIndustry"] not in top10_sub:
            continue
        r = at.analyze_pullback(gg, code, meta["Name"], meta["SubIndustry"], meta["Sector"])
        if r:
            part4.append(rec({
                "ticker": r["Ticker"], "name": r["Name"], "sub_industry": r["SubIndustry"],
                "close": r["Close"], "retrace": r["Retrace"],
                "ind_comp": round(float(sub_comp.get(r["SubIndustry"], 0)), 3),
                "hotness": round(float(hotness_map.get(r["Ticker"], 0)), 3),
                "doji": r["Doji"], "shrink": r["Shrink"], "entangle": r["Entangle"],
                "kdj_div": r["KDJdiv"], "at_support": r["AtSupport"], "nsig": r["NSig"],
            }))
    part4.sort(key=lambda x: (x["ind_comp"], x["hotness"]), reverse=True)

    # Part3: 异动放量, 排除 Part2/4, Top10申万三级内
    exclude = set(p["ticker"] for p in part2) | set(p["ticker"] for p in part4)
    surge = at.screen_surge(raw, pool, exclude=exclude)
    surge = [r for r in surge if r["SubIndustry"] in top10_sub]
    part3 = [rec({
        "ticker": r["Ticker"], "name": r["Name"], "sub_industry": r["SubIndustry"],
        "close": r["Close"], "ret_1d": r["Ret1d"], "ret_3d": r["Ret3d"],
        "vol_ratio": r["VolRatio"], "trigger": r["Trigger"],
        "hotness": round(float(hotness_map.get(r["Ticker"], 0)), 3),
    }) for r in surge]

    all_stocks = [rec({
        "ticker": r["Ticker"], "name": r["Name"], "sub_industry": r["SubIndustry"],
        "sector": r["Sector"], "close": r.get("close"), "hotness": r.get("hotness"),
        "ret_60": r.get("ret_60"), "pct_high_250": r.get("pct_high_250"),
        "ma_aligned": int(r["ma_aligned"]) if r.get("ma_aligned") is not None else 0,
    }) for _, r in df_full[df_full["SubIndustry"].isin(top10_sub)].sort_values("hotness", ascending=False).iterrows()]

    payload = {
        "date": asof, "market": "A", "hottest_sub": hottest_sub,
        "part1": part1, "part1c1": part1c1, "part1c2": part1c2, "part2": part2, "part3": part3, "part4": part4,
        "all_stocks": all_stocks,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[Export] -> {OUT_PATH}")
    print(f"  part1: Top{TOP_SUB} 申万三级, Top{TOP_SEC} 申万一级")
    print(f"  part1c1: {len(part1c1)} 首次新高 | part1c2: {len(part1c2)} 持续新高 | part2: {len(part2)} 趋势个股 | part3: {len(part3)} 异动 | part4: {len(part4)} 回调买点")
    print(f"  all_stocks: {len(all_stocks)} (Top{TOP_SUB}申万三级内)")


if __name__ == "__main__":
    main()
