"""A股板块热度聚类 + 趋势四部曲 (对标港股, 同库 jianxin, 行业用申万一级+三级).

universe: configs/a_universe.csv (top600 by 成交金额, 申万一级=Sector, 申万三级=SubIndustry).
取数: jianxin MySQL ashareeodprices (Wind schema, S_DQ_AMOUNT 单位千元, 同港股).
复用: pullback_buypoint.hotspot_scores/analyze/screen_surge + sector_cluster.compute_share_rotation
      (均为泛型: 吃 S_DQ_* 列 + pool 的 SubIndustry/Sector; A股EOD同Wind schema, 单位同千元).

独立运行打印 Part1-3; 完整四部曲 JSON 由 scripts/export_trend_a.py 组装.
用法: python src/a_trend.py [--asof 2026-06-18]
"""
from __future__ import annotations
import os, sys, argparse
import pandas as pd
import pymysql

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
from db_config import DB_CONFIG  # noqa: E402
import sector_cluster as sc       # 模块级已设 utf-8 stdout; 提供 forward_adjust_group/compute_features/compute_share_pctile200/compute_share_rotation
import pullback_buypoint as pb    # hotspot_scores/analyze/screen_surge (泛型, 复用)

UNIVERSE_CSV = os.path.join(PROJECT_DIR, "configs", "a_universe.csv")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOOKBACK_DAYS = 1200         # 日历日, 覆盖周线3年(~780交易日) + 日线热身


def _cache_path(asof):
    return os.path.join(DATA_DIR, f"a_eod_raw_{asof}.pkl")


def latest_asof() -> str:
    conn = pymysql.connect(**DB_CONFIG)
    try:
        df = pd.read_sql("SELECT MAX(TRADE_DT) AS d FROM ashareeodprices", conn)
    finally:
        conn.close()
    return pd.to_datetime(str(df.iloc[0, 0]), format="%Y%m%d").strftime("%Y-%m-%d")


def load_pool() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    df = df.dropna(subset=["Ticker"]).copy()
    df["SubIndustry"] = df["SubIndustry"].fillna("未分类").replace("", "未分类")
    df["Sector"] = df["Sector"].fillna("未分类").replace("", "未分类")
    return df[["Ticker", "Name", "SubIndustry", "Sector"]]


def fetch_all(pool_codes: list[str], asof: str) -> pd.DataFrame:
    """全池A股EOD (Wind schema). 带 asof pkl 缓存 (远程DB行传输慢, 首次~8-10min, 后续秒级)."""
    import pickle
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = _cache_path(asof)
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            raw = pickle.load(f)
        print(f"[Cache] 命中 {cache} ({raw['code'].nunique()} 只)")
        return raw
    end = asof.replace("-", "")
    start = (pd.to_datetime(asof) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    codes_sql = ",".join(f"'{c}'" for c in pool_codes)
    sql = f"""
        SELECT S_INFO_WINDCODE AS code, TRADE_DT,
               S_DQ_CLOSE, S_DQ_ADJOPEN, S_DQ_ADJHIGH, S_DQ_ADJLOW, S_DQ_ADJCLOSE,
               S_DQ_VOLUME, S_DQ_AMOUNT
        FROM ashareeodprices
        WHERE TRADE_DT BETWEEN '{start}' AND '{end}'
          AND S_INFO_WINDCODE IN ({codes_sql})
        ORDER BY S_INFO_WINDCODE, TRADE_DT
    """
    print(f"[DB] 拉取 {len(pool_codes)} 只A股EOD ({start}~{end}), 远程库较慢请耐心...")
    conn = pymysql.connect(**DB_CONFIG)
    try:
        raw = pd.read_sql(sql, conn)
    finally:
        conn.close()
    raw = raw[raw["TRADE_DT"] <= end].copy()
    with open(cache, "wb") as f:
        pickle.dump(raw, f)
    print(f"[Cache] 写入 {cache} ({raw['code'].nunique()} 只, {len(raw)} 行)")
    return raw


# 透出复用的打分/信号函数 (export 脚本用)
hotspot_scores = pb.hotspot_scores
analyze_pullback = pb.analyze
screen_surge = pb.screen_surge
compute_share_rotation = sc.compute_share_rotation
forward_adjust_group = sc.forward_adjust_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    ap.add_argument("--top-stocks", type=int, default=20)
    args = ap.parse_args()
    asof = args.asof or latest_asof()

    pool = load_pool()
    print(f"=== A股板块热度聚类 (asof={asof}) ===\n[Pool] {len(pool)} 只")
    raw = fetch_all(pool["Ticker"].tolist(), asof)
    print(f"[Fetch] {len(raw)} 行, {raw['code'].nunique()} 只")

    df, sub_rank, sec_rank = pb.hotspot_scores(pool, raw)
    hottest_sub = sub_rank.index[0] if len(sub_rank) else "N/A"
    print(f"[Feat] 有效 {len(df)} 只; 最热申万三级 = 【{hottest_sub}】 "
          f"composite={sub_rank.loc[hottest_sub,'composite']}" if len(sub_rank) else "[Feat] 无")

    top_stocks = df.sort_values("hotness", ascending=False).head(args.top_stocks)
    hot_sub_stocks = df[df["SubIndustry"] == hottest_sub].sort_values("hotness", ascending=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 22)
    print("\n## Part 1  最热申万三级 (top 15, n>=3)")
    show = ["n", "breadth_250%", "breadth_60%", "mean_amt_surge",
            "总成交金额_5日均值_亿", "占比_5日%", "share5_pctile200", "mean_ret_60", "composite"]
    print(sub_rank[show].head(15).to_string())
    print("\n## Part 1b  申万一级 热度 (n>=5)")
    print(sec_rank[show].to_string())
    print(f"\n## Part 2  最热个股 (hotness top {args.top_stocks})")
    c2 = ["Ticker", "Name", "SubIndustry", "close", "pct_high_250", "amt_surge",
          "amt_rank_pct", "ret_60", "ma_aligned", "hotness"]
    print(top_stocks[c2].round(3).to_string(index=False))
    print(f"\n## Part 3  最热申万三级【{hottest_sub}】个股热度排序")
    c3 = ["Ticker", "Name", "close", "pct_high_250", "nh_ratio_60",
          "amt_surge", "amt_rank_pct", "ret_60", "ma_aligned", "hotness"]
    print(hot_sub_stocks[c3].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
