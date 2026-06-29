"""中国ETF 持仓重叠聚类: 按 Wind 实际持仓(非名称)合并高度重叠的ETF, 同簇只留最流动的一只.
解决 "科创半导体设备 vs 半导体设备 vs 半导体" 名称略异但持仓高度重叠的问题.
Wind: chinamutualfundstockportfolio (F_PRT_STKVALUETONAV = 占NAV权重), 最新报告期.
用法: python scripts/etf_a_cluster.py [--threshold 0.5]
"""
from __future__ import annotations
import os, sys, io, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pymysql
import pandas as pd
from db_config import DB_CONFIG

DEDUP_CSV = os.path.join(os.path.dirname(__file__), "..", "output", "china_a_etf_dedup.csv")
OUT_CSV = os.path.join(os.path.dirname(__file__), "..", "output", "china_a_etf_clustered.csv")
BROAD_CAT = {"规模指数ETF"}   # 宽基(规模指数)不聚类; 只对行业/主题/策略类聚类
# 宽基白名单: 只保留这10个核心宽基(增强/Smart-beta版流动性都不达标, 全部剔除; 其余宽基不看)
BROAD_KEEP = {"沪深300", "中证500", "中证1000", "中证2000", "上证50",
              "科创50", "科创100", "科创200", "创业板", "创业板50"}


def _issuers(names):
    """从名称抽发行人 (ETF/LOF 之后的尾部), 数据驱动."""
    import re
    out = set()
    for n in names:
        m = re.search(r'(ETF|LOF)(.*)$', str(n))
        if m and m.group(2).strip():
            out.add(m.group(2).strip())
    return sorted(out, key=len, reverse=True)


def broad_theme(name, issuers):
    """宽基主题: 只剥发行人+ETF/LOF, 不剥中证/上证前缀 (保留 上证50/中证500 完整)."""
    n = str(name).replace('ETF', '').replace('LOF', '')
    for iss in issuers:
        if iss in n:
            n = n.replace(iss, '')
    return n.strip()


def fetch_holdings(tickers):
    conn = pymysql.connect(**DB_CONFIG); cur = conn.cursor()
    in_sql = ",".join(f"'{c}'" for c in tickers)
    # 取 2025-12-31 之后的持仓(含 2026Q1 top15 + 季报全量), 再在 pandas 里取每只最新报告期
    cur.execute(f"""SELECT S_INFO_WINDCODE, F_PRT_ENDDATE, S_INFO_STOCKWINDCODE, F_PRT_STKVALUETONAV
                    FROM chinamutualfundstockportfolio
                    WHERE S_INFO_WINDCODE IN ({in_sql}) AND F_PRT_ENDDATE >= '20251231'
                      AND F_PRT_STKVALUETONAV IS NOT NULL AND F_PRT_STKVALUETONAV > 0""")
    rows = cur.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=["etf", "rpt", "stock", "w"])
    # 每只 ETF 取最新报告期
    latest = df.groupby("etf")["rpt"].max().to_dict()
    df = df[df.apply(lambda r: r["rpt"] == latest.get(r["etf"]), axis=1)]
    holdings = {}
    for etf, g in df.groupby("etf"):
        holdings[etf] = dict(zip(g["stock"], g["w"].astype(float)))
    return holdings


def overlap(a, b):
    """加权 Jaccard = sum(min(wA,wB)) / sum(max(wA,wB)) over union. 对称, 0-1.
    两只集中且持仓相同的行业ETF→高; 宽基(分散)vs行业(集中)→低 (宽基大量权重在行业没持的票上)."""
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    num = sum(min(a.get(s, 0), b.get(s, 0)) for s in keys)
    den = sum(max(a.get(s, 0), b.get(s, 0)) for s in keys)
    return num / den if den > 0 else 0.0


def cluster(tickers, holdings, thr):
    """并查集: overlap>thr 的 ETF 合并同簇"""
    parent = {t: t for t in tickers}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            if a not in holdings or b not in holdings:
                continue
            if overlap(holdings[a], holdings[b]) >= thr:
                union(a, b)
    clusters = {}
    for t in tickers:
        clusters.setdefault(find(t), []).append(t)
    return list(clusters.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.5, help="持仓重叠阈值, ≥则合并")
    args = ap.parse_args()
    df = pd.read_csv(DEDUP_CSV)
    print(f"[输入] {len(df)} 只 ETF")
    issuers = _issuers(df["Name"])
    # 宽基: 按名称主题匹配白名单(10个核心宽基); 规模指数ETF不在白名单的(A500/上证180/双创50等)直接丢, 不进聚类
    is_keep = df["Name"].apply(lambda n: broad_theme(n, issuers) in BROAD_KEEP)
    broad = df[is_keep].copy()
    broad_drop = df[df["Category"].isin(BROAD_CAT) & ~is_keep]
    theme = df[~is_keep & ~df["Category"].isin(BROAD_CAT)].copy()
    print(f"[拆分] 宽基白名单 {len(broad)} 只 | 剔除非白名单规模指数宽基 {len(broad_drop)} 只 | 行业/主题 {len(theme)} 只待聚类")
    # 取持仓算集中度, 进一步剔除非白名单的"分散型宽基"(如 A500 被分在主题类, 持仓分散 top权重和<30)
    holdings = fetch_holdings(theme["Ticker"].tolist())
    conc = {t: sum(holdings.get(t, {}).values()) for t in theme["Ticker"]}
    spread_broad = theme[theme["Ticker"].apply(lambda t: 0 < conc.get(t, 999) < 30)]
    print(f"[集中度] 再剔除分散型非白名单宽基 {len(spread_broad)} 只 (top权重和<30, 如A500)")
    theme = theme[~theme["Ticker"].isin(spread_broad["Ticker"])].copy()
    print(f"[Wind] 持仓命中 {len(holdings)}/{len(theme)} (最新报告期)")
    tickers = theme["Ticker"].tolist()

    clusters = cluster(tickers, holdings, args.threshold)
    liq = dict(zip(df["Ticker"], df["Liquidity(亿元)"]))
    name = dict(zip(df["Ticker"], df["Name"]))
    cat = dict(zip(df["Ticker"], df["Category"]))
    rows = []
    merged_count = 0
    # 宽基: 全部保留 (ClusterSize=1)
    for _, r in broad.iterrows():
        rows.append({"Ticker": r["Ticker"], "Name": r["Name"], "Category": r["Category"],
                     "Liquidity(亿元)": r["Liquidity(亿元)"], "ClusterSize": 1, "ClusterMembers": ""})
    # 行业/主题: 每簇留最流动
    for cl in clusters:
        cl_sorted = sorted(cl, key=lambda t: liq.get(t, 0), reverse=True)
        rep = cl_sorted[0]
        rows.append({"Ticker": rep, "Name": name.get(rep, ""), "Category": cat.get(rep, ""),
                     "Liquidity(亿元)": liq.get(rep, 0), "ClusterSize": len(cl),
                     "ClusterMembers": ";".join(cl_sorted[1:]) if len(cl_sorted) > 1 else ""})
        merged_count += len(cl_sorted) - 1
    out = pd.DataFrame(rows).sort_values("Liquidity(亿元)", ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[Done] 阈值 {args.threshold}: {len(df)} → {len(out)} 只 (行业/主题合并 {merged_count}) → {OUT_CSV}")
    multi = out[(out["ClusterSize"] > 1) & (~out["Category"].isin(BROAD_CAT))]
    print(f"\n=== 合并簇 ({len(multi)} 个, 代表←被合并成员) ===")
    for _, r in multi.iterrows():
        print(f"  [留] {r['Ticker']} {r['Name']} ({r['Liquidity(亿元)']}亿)  ←  {r['ClusterMembers']}")


if __name__ == "__main__":
    main()
