"""A股 建仓策略回测 (强势/弱势, 移植自港股 backtest_entry.py)

universe: 中证800(沪深300∪中证500) + 科创100 + 创业板指(=创业板100) ≈ 879 只
  - 沪深300=000300.SH, 中证500=000905.SH (中证800无独立代码, 取两者并集)
  - 科创100=000698.SH, 创业板指=399006.SZ (100只成分, 即"创业板100")
基准: 沪深300 000300.SH (RS_Mansfield) | 数据: jianxin MySQL (Wind镜像 ashareeodprices)
退出: 固定horizon[5,10,20,60,120,250] + -8%固定止损 + 12%移动止损
报告: 胜率/盈亏比(payoff)/期望(expect)/avg_win·avg_loss 分项

用法:
  python backtest_entry_a.py --mode weak --asof 2026-07-25
  python backtest_entry_a.py --mode strong
"""
from __future__ import annotations
import os, sys, time, argparse
from pathlib import Path
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

from data_provider import forward_adjust  # noqa: E402  Wind schema 通用, A股ashareeodprices同列名
from backtest_entry import (              # noqa: E402  复用引擎
    HORIZONS, STOP_LOSS, TRAIL_STOP, LOOKBACK_DAYS, compute_signals, run_backtest,
)

BENCH = "000300.SH"
INDEX_CODES = ["000300.SH", "000905.SH", "000698.SH", "399006.SZ"]
DB_CONFIG = {
    "host": os.environ["DB_HOST"], "user": os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"], "database": os.environ.get("DB_NAME", "jianxin"),
    "port": int(os.environ.get("DB_PORT", "3306")), "charset": "utf8mb4",
}


def load_pool(cache_path: Path) -> list:
    """universe = 四指数当前成分(CUR_SIGN=1)并集 -> [(code, name, sector)]。缓存到 CSV。"""
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        return [(r.code, r.name, r.sector) for r in df.itertuples()]
    idx_sql = ",".join(f"'{i}'" for i in INDEX_CODES)
    conn = pymysql.connect(**DB_CONFIG)
    try:
        m = pd.read_sql(
            f"SELECT DISTINCT S_CON_WINDCODE AS code FROM aindexmembers "
            f"WHERE S_INFO_WINDCODE IN ({idx_sql}) AND CUR_SIGN=1", conn)
        codes = m["code"].tolist()
        cs = ",".join(f"'{c}'" for c in codes)
        n = pd.read_sql(
            f"SELECT S_INFO_WINDCODE AS code, S_INFO_NAME AS name FROM asharedescription "
            f"WHERE S_INFO_WINDCODE IN ({cs})", conn)
    finally:
        conn.close()
    n["sector"] = ""
    df = m.merge(n, on="code", how="left").fillna("")
    df.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return [(r.code, r.name, r.sector) for r in df.itertuples()]


def fetch_data_a(pool, asof, cache_path: Path) -> dict:
    """全池 A股 EOD (Wind schema) -> forward_adjust -> {code: daily_df}。缓存 parquet。"""
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return {c: g.drop(columns=["code"]).reset_index(drop=True) for c, g in df.groupby("code")}
    codes = [c for c, _, _ in pool] + [BENCH]
    end = asof.replace("-", "")
    start = (pd.to_datetime(asof) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    cs = ",".join(f"'{c}'" for c in codes)
    sql = (f"SELECT S_INFO_WINDCODE AS code, TRADE_DT, S_DQ_CLOSE, "
           f"S_DQ_ADJOPEN, S_DQ_ADJHIGH, S_DQ_ADJLOW, S_DQ_ADJCLOSE, S_DQ_VOLUME "
           f"FROM ashareeodprices WHERE TRADE_DT BETWEEN '{start}' AND '{end}' "
           f"AND S_INFO_WINDCODE IN ({cs}) ORDER BY S_INFO_WINDCODE, TRADE_DT")
    print(f"[DB] 拉取 {len(codes)} 只 A股 EOD ({start}~{end}), 远程库较慢请耐心...")
    conn = pymysql.connect(**DB_CONFIG)
    try:
        raw = pd.read_sql(sql, conn)
    finally:
        conn.close()
    raw = raw[raw["TRADE_DT"] <= end].copy()
    bag = {}
    for code, g in raw.groupby("code"):
        fa = forward_adjust(g)
        if fa is not None and not fa.empty:
            bag[code] = fa
    frames = [d.assign(code=c) for c, d in bag.items()]
    pd.concat(frames, ignore_index=True).to_parquet(cache_path)
    return bag


def main():
    ap = argparse.ArgumentParser(description="A股建仓策略回测 (强势/弱势)")
    ap.add_argument("--asof", default="2026-07-25")
    ap.add_argument("--mode", choices=["strong", "weak"], default="weak",
                    help="strong=RS或年线斜率前25%%; weak=不在前25%% (默认)")
    ap.add_argument("--inverse", action="store_true", help="反向做空: 同一信号对称止损")
    ap.add_argument("--top", action="store_true", help="筛选镜像: 见顶信号(超买/顶背离/死叉)代替抄底")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    out_dir = ROOT / cfg["output"]["dir"]; out_dir.mkdir(exist_ok=True)

    pool = load_pool(out_dir / "a_universe_idx.csv")
    print(f"[Pool] {len(pool)} 只 | asof={args.asof} | mode={args.mode} | inverse={args.inverse} | top={args.top}")

    bag = fetch_data_a(pool, args.asof, out_dir / "backtest_entry_a_bag.parquet")
    bench = bag.get(BENCH)
    print(f"[Data] {len(bag)} 只有数据 (含基准)")

    suffix = ("_top" if args.top else "") + ("_inverse" if args.inverse else "")
    run_backtest(pool, bag, bench, args.asof, args.mode,
                 out_dir / f"backtest_entry_a_{args.mode}{suffix}_results.csv",
                 label="A股建仓策略",
                 direction="short" if args.inverse else "long",
                 screen="top" if args.top else "bottom")


if __name__ == "__main__":
    main()
