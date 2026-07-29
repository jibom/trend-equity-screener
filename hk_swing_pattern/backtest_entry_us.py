"""美股 建仓策略回测 (强势/弱势股方法论, 移植自港股 backtest_entry.py)

  池筛选(每日截面): strong = OR(RS_SPY > 75百分位, 年线斜率 > 75百分位)
                    weak   = NOT (上述)
  买入条件(全部AND):
    1. 周线J极值 <= 25 AND 至少1个反转信号(周度背离/周度DeMark9或13/日度背离=1/climax=-1)
    2. SOS = 1
    3. 至少1个趋势建立(日度DeMark/日MACD=1/5_10 Cross=1/10_50 Cross=1)
  退出: 固定持有期 5/10/20/60/120/250 交易日; 期间触及 -8% 固定止损 或 峰值回撤12%移动止损 则提前平仓

universe: src/us_pool.py (S&P500∪NDX100 + EU/JP/KR ADR), 基准 SPY, 数据 EODHD。
用法:
  python backtest_entry_us.py                         # weak (默认)
  python backtest_entry_us.py --mode strong --asof 2026-07-25
  python backtest_entry_us.py --rebuild               # 重建美股 universe 缓存
"""
from __future__ import annotations
import sys, time, argparse, datetime as dt
from pathlib import Path
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

import eodhd                       # noqa: E402
import us_pool                     # noqa: E402
from data_provider import forward_adjust  # noqa: E402
from backtest_entry import (        # noqa: E402  复用引擎
    HORIZONS, STOP_LOSS, TRAIL_STOP, LOOKBACK_DAYS, compute_signals, run_backtest,
)

BENCH = "SPY.US"


def fetch_data_us(pool, asof, cache_path):
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return {c: g.drop(columns=["code"]).reset_index(drop=True) for c, g in df.groupby("code")}
    codes = [c for c, _, _ in pool] + [BENCH]
    raw = eodhd.fetch_all_eodhd(codes, asof, LOOKBACK_DAYS, workers=5)
    bag = {}
    for code, df in raw.items():
        if df is None or df.empty:
            continue
        fa = forward_adjust(df)
        if fa is not None and not fa.empty:
            bag[code] = fa
    frames = [df.assign(code=c) for c, df in bag.items()]
    pd.concat(frames, ignore_index=True).to_parquet(cache_path)
    return bag


def main():
    ap = argparse.ArgumentParser(description="美股建仓策略回测 (强势/弱势)")
    ap.add_argument("--asof", default="2026-07-25")
    ap.add_argument("--mode", choices=["strong", "weak"], default="weak",
                    help="strong=RS或年线斜率前25%%; weak=不在前25%% (默认)")
    ap.add_argument("--rebuild", action="store_true", help="重建美股 universe 缓存")
    ap.add_argument("--inverse", action="store_true", help="反向做空: 同一信号对称止损")
    ap.add_argument("--top", action="store_true", help="筛选镜像: 见顶信号(超买/顶背离/死叉)代替抄底")
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    out_dir = ROOT / cfg["output"]["dir"]; out_dir.mkdir(exist_ok=True)

    pool = us_pool.load_us_pool(ROOT / "us_pool.csv", rebuild=args.rebuild)
    print(f"[Pool] {len(pool)} 只 | asof={args.asof} | mode={args.mode} | inverse={args.inverse} | top={args.top}")

    cache = out_dir / "backtest_entry_us_bag.parquet"
    bag = fetch_data_us(pool, args.asof, cache)
    bench = bag.get(BENCH)
    print(f"[Data] {len(bag)} 只有数据 (含基准)")

    suffix = ("_top" if args.top else "") + ("_inverse" if args.inverse else "")
    run_backtest(pool, bag, bench, args.asof, args.mode,
                 out_dir / f"backtest_entry_us_{args.mode}{suffix}_results.csv",
                 label="美股建仓策略",
                 direction="short" if args.inverse else "long",
                 screen="top" if args.top else "bottom")


if __name__ == "__main__":
    main()
