"""
趋势股回调买点筛选器 (Part 4)

在 305 港股池中找: 处于中期上升趋势、近期深回调(>=10%)到支撑位、且出现>=2个技术买点信号的股票。
复用 sector_cluster 的取数/前复权/特征 + 热点打分(行业 composite + 个股 hotness), kdj_div_basic 的 KDJ 底背离。

趋势股: 价格>MA200 且 MA200年化>0 (长期向上) 且 近120日涨幅>10% (趋势曾确立)
回调:   距近60日高点回撤 10%~25% (深回调, 趋势未破)
剔除:   最新大阴线(实体>=60%+收位<=35%)跌破近5日低点 → 破位下行, 等新均衡
买点信号 (满足>=2):
  doji      近5日有十字星/锤子线 (多空平衡)
  shrink    近5日量中位/30日均量 <= 0.8 (缩量)
  entangle  MA5/10/20 离散度 <= 2% (短期均线纠缠)
  kdj_div   日线或周线 KDJ 底背离
  at_support收盘在 MA20/50/60 ±2% 内 (触及支撑)

排序: 先按【行业综合分(细分行业composite)】降序, 同行业按【个股综合分(hotness)】降序。

用法: python src/pullback_buypoint.py --asof 2026-06-20
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# 复用 sector_cluster (其模块级已设 utf-8 stdout), 不再单独 wrap
import sector_cluster as sc
from kdj_div_basic import calc_kdj, detect_divergence


def analyze(g, ticker, name, sub, sector):
    """单股回调买点判定。返回候选 dict 或 None。"""
    if g is None or len(g) < 210:          # 需 MA200 + 30日斜率
        return None
    g = g.copy()
    g["date"] = pd.to_datetime(g["TRADE_DT"], format="%Y%m%d")
    c = g["fwd_close"]
    close = float(c.iloc[-1])

    ma = {w: c.rolling(w).mean().iloc[-1] for w in (5, 10, 20, 50, 60, 200)}
    if any(pd.isna(v) for v in [ma[5], ma[10], ma[20], ma[50], ma[60], ma[200]]):
        return None
    ma200_30ago = c.rolling(200).mean().iloc[-31]
    if pd.isna(ma200_30ago) or ma200_30ago == 0:
        return None
    ma200_ann = (ma[200] / ma200_30ago - 1) * (250 / 30)
    ret_120 = close / float(c.iloc[-121]) - 1 if len(c) > 120 else None

    # --- 趋势 ---
    if not ((close > ma[200]) and (ma200_ann > 0) and ret_120 is not None and ret_120 > 0.10):
        return None

    # --- 深回调 10%~25% ---
    peak_60 = float(c.iloc[-60:].max())
    retrace = (peak_60 - close) / peak_60 if peak_60 > 0 else 0.0
    if not (0.10 <= retrace <= 0.25):
        return None

    # --- 排除: 最新大阴线跌破前期均衡(近5日低点) → 已破位下行, 非买点, 等新均衡 ---
    oo = float(g["S_DQ_ADJOPEN"].iloc[-1]); oh = float(g["S_DQ_ADJHIGH"].iloc[-1])
    ol = float(g["S_DQ_ADJLOW"].iloc[-1]); oc = float(g["S_DQ_ADJCLOSE"].iloc[-1])
    rng0 = oh - ol
    if rng0 > 0:
        body0 = abs(oc - oo) / rng0
        loc0 = (oc - ol) / rng0
        prior5_low = float(g["S_DQ_ADJLOW"].iloc[-6:-1].min())
        if oc < oo and body0 >= 0.60 and loc0 <= 0.35 and oc < prior5_low:
            return None

    # --- 买点信号 ---
    # 十字星/锤子线用后复权 OHLC (同日同因子, 比值不变)
    o = g["S_DQ_ADJOPEN"].iloc[-5:].values; h = g["S_DQ_ADJHIGH"].iloc[-5:].values
    lo = g["S_DQ_ADJLOW"].iloc[-5:].values; cl = g["S_DQ_ADJCLOSE"].iloc[-5:].values
    doji = False
    for i in range(len(o)):
        rng = h[i] - lo[i]
        if rng <= 0:
            continue
        body = abs(cl[i] - o[i]); lsh = min(o[i], cl[i]) - lo[i]
        if body / rng <= 0.10:
            doji = True; break
        if body > 0 and lsh >= 2 * body and lsh >= 0.6 * rng:
            doji = True; break

    vol = g["vol"].values
    v30 = np.nanmean(vol[-30:])
    shrink = bool(v30 > 0 and np.nanmedian(vol[-5:]) / v30 <= 0.8)

    vals = np.array([ma[5], ma[10], ma[20]])
    entangle = bool((vals.max() - vals.min()) / np.median(vals) <= 0.02)

    try:
        kres = detect_divergence(calc_kdj(g))
        kdj_div = (kres["daily_divergence"] == "底背离") or (kres["weekly_divergence"] == "底背离")
    except Exception:
        kdj_div = False

    at_support = any(abs(close / ma[w] - 1) <= 0.02 for w in (20, 50, 60))

    n_sig = sum([doji, shrink, entangle, kdj_div, at_support])
    if n_sig < 2:
        return None

    return dict(Ticker=ticker, Name=name, SubIndustry=sub, Sector=sector,
                Close=round(close, 2), Retrace=round(retrace * 100, 1),
                MA200ann=round(ma200_ann * 100, 1), Ret120=round(ret_120 * 100, 1),
                Doji=int(doji), Shrink=int(shrink), Entangle=int(entangle),
                KDJdiv=int(kdj_div), AtSupport=int(at_support), NSig=n_sig,
                MA20=round(ma[20], 2), MA50=round(ma[50], 2), MA60=round(ma[60], 2))


def screen_surge(raw, pool, exclude=None):
    """异动放量个股: 近5日单日涨幅>4%+放量, 或 近3日累计涨幅>10%+放量(3日均量)。
    exclude: 跳过的 ticker 集合(已在其它 screen 中)。返回 list[dict], 按异动强度降序。"""
    exclude = exclude or set()
    out = []
    for code, g in raw.groupby("code"):
        gg = sc.forward_adjust_group(g)
        if gg is None or len(gg) < 35:
            continue
        c = gg["fwd_close"].values
        v = gg["vol"].values
        vol_ma = np.nanmean(v[-30:])          # 30日均量(放量基准)
        if not vol_ma or vol_ma <= 0:
            continue
        ret = np.diff(c) / c[:-1] * 100.0      # 日涨幅% (ret[i] 对应 v[i+1] 当日)

        # 单日: 近3日某日 >4% 且 当日量 >=1.5×30日均量, 且涨幅守住(最新收盘≥起涨前收盘, 排除冲高回落假突破)
        single = False
        single_ret = single_vr = None
        for i in range(max(0, len(ret) - 3), len(ret)):
            if ret[i] > 4 and v[i + 1] >= 1.5 * vol_ma and c[-1] >= c[i]:
                single = True
                single_ret = float(ret[i])
                single_vr = float(v[i + 1] / vol_ma)
                break
        # 3日: 近3日累计>10% 且 3日均量>=1.5×30日均量
        ret3 = (c[-1] / c[-4] - 1) * 100.0 if len(c) >= 4 else None
        three_vr = float(np.nanmean(v[-3:]) / vol_ma) if vol_ma else None
        three = ret3 is not None and ret3 > 10 and three_vr is not None and three_vr >= 1.5
        if not (single or three) or code in exclude:
            continue
        meta = pool[pool["Ticker"] == code]
        if meta.empty:
            continue
        meta = meta.iloc[0]
        trig = "单日+3日" if single and three else ("单日" if single else "3日")
        out.append(dict(Ticker=code, Name=meta["Name"], SubIndustry=meta["SubIndustry"],
                        Sector=meta["Sector"], Close=round(float(c[-1]), 2),
                        Ret1d=round(single_ret, 1) if single else None,           # 单日异动当天涨幅
                        Ret3d=round(ret3, 1) if ret3 is not None else None,       # 3日累计涨幅
                        VolRatio=round(single_vr, 2) if single
                                 else (round(three_vr, 2) if three_vr is not None else None),
                        Trigger=trig))
    out.sort(key=lambda x: max(x.get("Ret3d") or 0, x.get("Ret1d") or 0), reverse=True)
    return out


def hotspot_scores(pool, raw):
    """复用 sector_cluster 逻辑: 算全池个股特征+hotness, 细分行业/GICS板块 composite。

    返回 (df_full, sub_rank, sector_rank):
      df_full   — 全池个股 DataFrame (含 Ticker/Name/SubIndustry/Sector + 特征 + hotness)
      sub_rank  — 细分行业 composite 排名 (n>=3), 含 breadth/成交金额/占比/composite 等
      sector_rank — GICS 板块 composite 排名 (n>=5)
    """
    rows = []
    for code, g in raw.groupby("code"):
        gg = sc.forward_adjust_group(g)
        f = sc.compute_features(gg)
        if f is None:
            continue
        m = pool[pool["Ticker"] == code]
        if m.empty:
            continue
        m = m.iloc[0]
        rows.append({"Ticker": code, "Name": m["Name"], "SubIndustry": m["SubIndustry"],
                     "Sector": m["Sector"], **f})
    df = pd.DataFrame(rows).dropna(subset=["pct_high_250", "amt_surge", "ret_60"]).reset_index(drop=True)

    # 个股 hotness
    df["amt_rank_pct"] = df["amt_5d"].rank(pct=True)
    for col in ["pct_high_250", "nh_ratio_60", "amt_surge", "amt_rank_pct"]:
        df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
    df["hotness"] = df[["z_pct_high_250", "z_nh_ratio_60", "z_amt_surge", "z_amt_rank_pct"]].mean(axis=1)

    # 分组打分 (与 sector_cluster 同公式同列); composite 用 5日占比200日百分位
    SCORE_COLS = ["breadth_250%", "breadth_60%", "mean_amt_surge", "share5_pctile200", "mean_ret_60"]
    pa1 = df["amt_1d"].sum(); pa5 = df["amt_5d"].sum()
    pctile_sub = sc.compute_share_pctile200(raw, pool, "SubIndustry")
    pctile_sec = sc.compute_share_pctile200(raw, pool, "Sector")

    def group_score(d):
        g1 = d["amt_1d"].sum(); g5 = d["amt_5d"].sum()
        return pd.Series({
            "n": len(d),
            "mean_pct_high_250": d["pct_high_250"].mean(),
            "breadth_250%": (d["pct_high_250"] >= 0.95).mean() * 100,
            "breadth_60%": (d["pct_high_60"] >= 0.98).mean() * 100,
            "mean_amt_surge": d["amt_surge"].mean(),
            "mean_amt_rank": d["amt_rank_pct"].mean(),
            "总成交金额_1日_亿": round(g1 / 1e5, 2),
            "总成交金额_5日均值_亿": round(g5 / 1e5, 2),
            "占比_1日%": round(g1 / pa1 * 100, 2) if pa1 else 0.0,
            "占比_5日%": round(g5 / pa5 * 100, 2) if pa5 else 0.0,
            "mean_ret_60": d["ret_60"].mean(),
            "mean_hotness": d["hotness"].mean(),
        })

    def rank_group(col, min_n, pctile):
        g = df.groupby(col).apply(group_score, include_groups=False)
        g = g[g["n"] >= min_n].copy()
        g["share5_pctile200"] = g.index.map(pctile).astype(float).fillna(50.0)
        z = g.copy()
        for c in SCORE_COLS:
            s = z[c].std()
            z[c] = (z[c] - z[c].mean()) / s if s and s > 0 else 0.0
        g["composite"] = (0.25 * z["breadth_250%"] + 0.15 * z["breadth_60%"]
                          + 0.15 * z["mean_amt_surge"] + 0.30 * z["share5_pctile200"]
                          + 0.15 * z["mean_ret_60"])
        return g.sort_values("composite", ascending=False).round(3)

    sub_rank = rank_group("SubIndustry", min_n=3, pctile=pctile_sub)
    sector_rank = rank_group("Sector", min_n=5, pctile=pctile_sec)
    return df, sub_rank, sector_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="截止日期; 默认取 DB 最新交易日")
    args = ap.parse_args()
    asof = args.asof or sc.latest_asof()

    pool = sc.load_pool()
    print(f"=== 趋势股回调买点筛选 (asof={asof}) ===\n[Pool] {len(pool)} 只")
    raw = sc.fetch_all(pool["Ticker"].tolist(), asof)
    print(f"[Fetch] {raw['code'].nunique()} 只有数据")

    # 1) 全池热点打分 (行业 composite + 个股 hotness)
    df_full, sub_rank, sector_rank = hotspot_scores(pool, raw)
    hotness_map = df_full.set_index("Ticker")["hotness"]
    ind_comp = sub_rank["composite"]
    print(f"[Hotspot] {len(df_full)} 只有个股分, {len(sub_rank)} 个细分行业有综合分")

    # 2) 回调买点筛选
    rows = []
    for code, g in raw.groupby("code"):
        gg = sc.forward_adjust_group(g)
        meta = pool[pool["Ticker"] == code]
        if meta.empty:
            continue
        meta = meta.iloc[0]
        r = analyze(gg, code, meta["Name"], meta["SubIndustry"], meta["Sector"])
        if r:
            rows.append(r)
    if not rows:
        print("\n[结果] 无符合条件的回调买点标的。")
        return

    df = pd.DataFrame(rows)
    # 3) 附行业综合分 + 个股综合分
    df["IndComp"] = df["SubIndustry"].map(ind_comp).round(3)
    df["Hotness"] = df["Ticker"].map(hotness_map).round(3)
    # 4) 排序: 行业综合分降序, 同行业个股综合分降序
    df = df.sort_values(["IndComp", "Hotness"], ascending=[False, False]).reset_index(drop=True)

    print(f"\n[结果] 命中 {len(df)} 只趋势回调买点标的 (按行业综合分→个股综合分排序)")

    sig_cols = ["Doji", "Shrink", "Entangle", "KDJdiv", "AtSupport"]
    print("\n--- 信号分布 ---")
    print(pd.DataFrame({s: df[s].sum() for s in sig_cols}, index=["命中数"]).T.to_string())

    print("\n--- 候选 (行业综合分→个股综合分排序) ---")
    show = ["Ticker", "Name", "SubIndustry", "Close", "Retrace", "IndComp", "Hotness",
            "Doji", "Shrink", "Entangle", "KDJdiv", "AtSupport", "NSig"]
    print(df[show].to_string(index=False))

    out_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    out_path = os.path.join(out_dir, f"pullback_buypoint_{asof}.xlsx")
    df.to_excel(out_path, index=False, sheet_name="回调买点")
    print(f"\n[Excel] -> {out_path}")


if __name__ == "__main__":
    main()
