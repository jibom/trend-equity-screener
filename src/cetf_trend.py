"""中国ETF (A股+港股通行业) 板块热度 + 趋势 (对标美股ETF, Wind取行情).
universe: configs/cetf_universe.csv (485只, Industry=跟踪基准).
取数: jianxin MySQL chinaclosedfundeodprice (Wind schema, S_DQ_AMOUNT 单位千元).
复用: pullback_buypoint.hotspot_scores/analyze/screen_surge + sector_cluster.compute_features
      (含 pct_high_126/nh_ratio_126/ma_stack) + compute_share_rotation.
独立运行打印 Part1-3; 完整 JSON 由 scripts/export_trend_cetf.py 组装.
用法: python src/cetf_trend.py [--asof 2026-06-26]
"""
from __future__ import annotations
import os, sys, argparse
import pandas as pd
import pymysql

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
from db_config import DB_CONFIG
import sector_cluster as sc       # forward_adjust_group/compute_features/compute_share_rotation
import pullback_buypoint as pb    # hotspot_scores/analyze/screen_surge (泛型)

UNIVERSE_CSV = os.path.join(PROJECT_DIR, "configs", "cetf_universe.csv")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LOOKBACK_DAYS = 1200         # 日历日, 覆盖周线3年(~780交易日) + 日线热身
TABLE = "chinaclosedfundeodprice"   # Wind ETF EOD (含 S_DQ_ADJOPEN/HIGH/LOW/CLOSE/AMOUNT/VOLUME)


def _cache_path(asof):
    return os.path.join(DATA_DIR, f"cetf_eod_raw_{asof}.pkl")


def latest_asof() -> str:
    conn = pymysql.connect(**DB_CONFIG)
    try:
        df = pd.read_sql(f"SELECT MAX(TRADE_DT) AS d FROM {TABLE}", conn)
    finally:
        conn.close()
    return pd.to_datetime(str(df.iloc[0, 0]), format="%Y%m%d").strftime("%Y-%m-%d")


def load_pool() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    df = df.dropna(subset=["Ticker"]).copy()
    df["SubIndustry"] = df["Industry"].fillna("未分类").replace("", "未分类")
    df["Sector"] = df["Sector"].fillna("未分类").replace("", "未分类")
    return df[["Ticker", "Name", "SubIndustry", "Sector"]]


def fetch_all(pool_codes: list[str], asof: str) -> pd.DataFrame:
    """全池ETF EOD (Wind schema). 带 asof pkl 缓存."""
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
        FROM {TABLE}
        WHERE TRADE_DT BETWEEN '{start}' AND '{end}'
          AND S_INFO_WINDCODE IN ({codes_sql})
        ORDER BY S_INFO_WINDCODE, TRADE_DT
    """
    print(f"[DB] 拉取 {len(pool_codes)} 只ETF EOD ({start}~{end}) ...")
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


hotspot_scores = pb.hotspot_scores
analyze_pullback = pb.analyze
screen_surge = pb.screen_surge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()
    asof = args.asof or latest_asof()
    pool = load_pool()
    print(f"=== 中国ETF (asof={asof}) ===\n[Pool] {len(pool)} 只")
    raw = fetch_all(pool["Ticker"].tolist(), asof)
    print(f"[Fetch] {raw['code'].nunique()} 只有数据")
    df, sub_rank, sec_rank = pb.hotspot_scores(pool, raw)
    hottest = sub_rank.index[0] if len(sub_rank) else "N/A"
    print(f"[Hotspot] {len(df)} 只有效, 最热基准={hottest}")


if __name__ == "__main__":
    main()
