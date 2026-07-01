"""
导出港股趋势四部曲 (Part1-4) 到 public/data/trend_hotspot_hk.json, 供网页「港股-趋势」tab 使用。

复用 src/pullback_buypoint.py:
  hotspot_scores(pool, raw) -> (df_full, sub_rank, sector_rank)   # Part1-3 打分
  analyze(g, ...) -> Part4 回调买点候选

用法: python scripts/export_trend_hotspot.py --asof 2026-06-20
输出: public/data/trend_hotspot_hk.json
"""
from __future__ import annotations
import os, sys, json, argparse
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import sector_cluster as sc  # 其模块级已设 utf-8 stdout, 本脚本不再单独 wrap
import pullback_buypoint as pb

OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "public", "data", "trend_hotspot_hk.json")
# asof 默认取 DB 最新交易日 (避免 asof=今天 但数据滞后导致日期与数据不符)


def clean(v):
    """NaN/inf → None; float → round 3 位。"""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return v
    if isinstance(v, float):
        return round(v, 3)
    return v


def rec(d: dict) -> dict:
    return {k: clean(v) for k, v in d.items()}


def part1_records(rank_df, name_key, sector_map=None):
    """sub_rank/sector_rank → list[dict], 列名转 snake_case。sector_map: 子行业→GICS板块(可选)。"""
    out = []
    for idx, row in rank_df.iterrows():
        d = {
            name_key: idx,
            "n": int(row["n"]),
            "pct_high_250": row.get("mean_pct_high_250"),
            "breadth_250": row.get("breadth_250%"),
            "breadth_60": row.get("breadth_60%"),
            "amt_surge": row.get("mean_amt_surge"),
            "amt_rank": row.get("mean_amt_rank"),
            "amt_1d_yi": row.get("总成交金额_1日_亿"),
            "amt_5d_yi": row.get("总成交金额_5日均值_亿"),
            "share_1d": row.get("占比_1日%"),
            "share_5d": row.get("占比_5日%"),
            "share5_pctile200": row.get("share5_pctile200"),
            "ret_60": row.get("mean_ret_60"),
            "composite": row.get("composite"),
        }
        if sector_map is not None:
            d["sector"] = sector_map.get(idx)
        out.append(rec(d))
    return out


def stock_record(r, with_nh=False):
    d = {
        "ticker": r["Ticker"],
        "name": r["Name"],
        "sub_industry": r["SubIndustry"],
        "sector": r.get("Sector"),
        "close": r.get("close"),
        "pct_high_250": r.get("pct_high_250"),
        "amt_surge": r.get("amt_surge"),
        "amt_rank": r.get("amt_rank_pct"),
        "ret_60": r.get("ret_60"),
        "ma_aligned": int(r["ma_aligned"]) if r.get("ma_aligned") is not None else 0,
        "days_below_ma10": int(r.get("days_below_ma10", 0)),
        "hotness": r.get("hotness"),
    }
    if with_nh:
        d["pct_high_126"] = r.get("pct_high_126")
        d["nh_ratio_126"] = r.get("nh_ratio_126")
    return rec(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="截止日期 YYYY-MM-DD; 默认取 DB 最新交易日")
    args = ap.parse_args()
    asof = args.asof or sc.latest_asof()

    pool = sc.load_pool()
    print(f"=== export_trend_hotspot (asof={asof}) ===\n[Pool] {len(pool)} 只")
    raw = sc.fetch_all(pool["Ticker"].tolist(), asof)
    print(f"[Fetch] {raw['code'].nunique()} 只有数据")

    # Part1-3 打分
    df_full, sub_rank, sector_rank = pb.hotspot_scores(pool, raw)
    hottest_sub = sub_rank.index[0]
    print(f"[Hotspot] {len(df_full)} 只个股, 最热细分行业={hottest_sub}")

    # 子行业 → 所属 GICS 板块 (成员股票的众数), 用于 GICS 钻取弹窗
    sub_to_sector = df_full.groupby("SubIndustry")["Sector"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    part1 = {
        "sub_industries": part1_records(sub_rank, "sub_industry", sector_map=sub_to_sector),
        "sectors": part1_records(sector_rank, "sector"),
    }

    # Part2: 趋势个股 (60日涨幅>10%) 按 hotness 排序, 最多20只
    top = df_full[df_full["ret_60"] > 0.10].sort_values("hotness", ascending=False).head(20)
    part2 = [stock_record(r) for _, r in top.iterrows()]

    # Part1c: 6月新高 (pct_high_126≥0.98, 126交易日≈6个月) — 拆首次/持续, 互斥
    nh6m = df_full[df_full["pct_high_126"] >= 0.98]
    # 1c1 首次新高: 近6月新高天数占比<4%(刚突破), 不要求多头, 低位优先(距250高升序)
    cond_6m = (df_full["pct_high_126"] >= 0.98) & (df_full["nh_ratio_126"] < 0.04)
    cond_3m = (df_full["pct_high_60"] >= 1.0) & (df_full["nh_ratio_60"] < 0.04)
    first_nh = df_full[cond_6m | cond_3m].sort_values("hotness", ascending=False).head(20)
    # 1c2 持续新高: 占比≥4% + 多头排列(ma_stack), 排除已在1c1的, hotness 排序
    sust_nh = nh6m[(nh6m["nh_ratio_126"] >= 0.04)
                   & (~nh6m["Ticker"].isin(first_nh["Ticker"]))]
    sust_nh = sust_nh.sort_values("hotness", ascending=False).head(20)
    part1c1 = [stock_record(r, with_nh=True) for _, r in first_nh.iterrows()]
    part1c2 = [stock_record(r, with_nh=True) for _, r in sust_nh.iterrows()]

    # Part4: 回调买点候选 (先算, 用于 Part3 排除)
    hotness_map = df_full.set_index("Ticker")["hotness"]
    ind_comp = sub_rank["composite"]
    rows = []
    for code, g in raw.groupby("code"):
        gg = sc.forward_adjust_group(g)
        meta = pool[pool["Ticker"] == code]
        if meta.empty:
            continue
        meta = meta.iloc[0]
        r = pb.analyze(gg, code, meta["Name"], meta["SubIndustry"], meta["Sector"])
        if r:
            rows.append(r)
    part4 = []
    for r in rows:
        part4.append(rec({
            "ticker": r["Ticker"], "name": r["Name"], "sub_industry": r["SubIndustry"],
            "close": r["Close"], "retrace": r["Retrace"],
            "ind_comp": round(float(ind_comp.get(r["SubIndustry"], 0)), 3),
            "hotness": round(float(hotness_map.get(r["Ticker"], 0)), 3),
            "doji": r["Doji"], "shrink": r["Shrink"], "entangle": r["Entangle"],
            "kdj_div": r["KDJdiv"], "at_support": r["AtSupport"], "nsig": r["NSig"],
        }))
    part4.sort(key=lambda x: (x["ind_comp"], x["hotness"]), reverse=True)

    # Part3: 异动放量个股 (单日>4%+放量 或 3日>10%+放量), 排除已在 Part2/Part4 的
    exclude = set(p["ticker"] for p in part2) | set(p["ticker"] for p in part4)
    surge = pb.screen_surge(raw, pool, exclude=exclude)
    part3 = []
    for r in surge:
        part3.append(rec({
            "ticker": r["Ticker"], "name": r["Name"], "sub_industry": r["SubIndustry"],
            "close": r["Close"], "ret_1d": r["Ret1d"], "ret_3d": r["Ret3d"],
            "vol_ratio": r["VolRatio"], "trigger": r["Trigger"],
            "hotness": round(float(hotness_map.get(r["Ticker"], 0)), 3),
        }))

    # Part5: 资金轮动 (5日占比200日分位·10日变化)
    rot = sc.compute_share_rotation(raw, pool, "SubIndustry", lookback=10)
    part5 = []
    for _, r in rot.iterrows():
        part5.append(rec({
            "sub_industry": r["SubIndustry"], "n": int(r["n"]),
            "pctile_now": r["pctile_now"], "pctile_ago": r["pctile_ago"],
            "delta10": r["delta10"], "amt5_now_yi": r["amt5_now_yi"],
            "amt5_chg_pct": r["amt5_chg_pct"], "flag": r["flag"],
        }))
    rotation_alerts = {
        "lose": [p for p in part5 if p["flag"] == "lose"],
        "gain": [p for p in part5 if p["flag"] == "gain"],
    }

    # all_stocks: 全池个股 (供子行业钻取显示成分股, 按热度分排序)
    all_stocks = []
    for _, r in df_full.sort_values("hotness", ascending=False).iterrows():
        all_stocks.append(rec({
            "ticker": r["Ticker"], "name": r["Name"], "sub_industry": r["SubIndustry"],
            "sector": r["Sector"], "close": r.get("close"), "hotness": r.get("hotness"),
            "ret_60": r.get("ret_60"), "pct_high_250": r.get("pct_high_250"),
            "ma_aligned": int(r["ma_aligned"]) if r.get("ma_aligned") is not None else 0,
        "days_below_ma10": int(r.get("days_below_ma10", 0)),
        }))

    payload = {
        "date": asof,
        "hottest_sub": hottest_sub,
        "part1": part1,
        "part1c1": part1c1,
        "part1c2": part1c2,
        "part2": part2,
        "part3": part3,
        "part4": part4,
        "part5": part5,
        "rotation_alerts": rotation_alerts,
        "all_stocks": all_stocks,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[Export] -> {OUT_PATH}")
    print(f"  part1: {len(part1['sub_industries'])} 细分行业, {len(part1['sectors'])} GICS板块")
    print(f"  part1c1: {len(part1c1)} 首次新高 | part1c2: {len(part1c2)} 持续新高 | part2: {len(part2)} 只趋势个股 (60日>10%, ≤20)")
    print(f"  part3: {len(part3)} 只异动放量个股 (已排除Part2/4)")
    print(f"  part4: {len(part4)} 只回调买点候选")
    print(f"  part5: {len(part5)} 个行业轮动 (🔻lose={len(rotation_alerts['lose'])} 🔺gain={len(rotation_alerts['gain'])})")


if __name__ == "__main__":
    main()
