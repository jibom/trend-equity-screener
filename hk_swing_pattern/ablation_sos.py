"""入场信号消融: cond_sos 在港/A/美 三市场的胜率/盈亏比/期望。
每市场算一次信号, 评估4种入场(均做多, 8%止损+12%trailing):
  全量    = universe & cond_j & trend>=3 & (SOS | (reversal&trend))
  SOS支   = universe & cond_j & trend>=3 & SOS
  非SOS支 = universe & cond_j & trend>=3 & reversal&trend & ~SOS
  SOS裸   = universe & SOS   (去掉cond_j/trend门, 看SOS原始力度)
用法: python ablation_sos.py [--markets hk,us,a] [--asof 2026-07-24]
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
if sys.platform == "win32":
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

import pandas as pd
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

import backtest_entry as BE
import backtest_entry_us as BU
import backtest_entry_a as BA
from backtest_entry import compute_all_signals, build_conds, evaluate_entries


def load_market(market, asof):
    out_dir = ROOT / "output"
    if market == "hk":
        import provider as P, yaml
        cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
        pool = P.load_pool(cfg["data"]["pool_csv"])
        bag = BE.fetch_data(pool, asof, out_dir / "backtest_entry_bag.parquet")
        bench = bag.get("2800.HK")
    elif market == "us":
        import us_pool
        pool = us_pool.load_us_pool(ROOT / "us_pool.csv")
        bag = BU.fetch_data_us(pool, asof, out_dir / "backtest_entry_us_bag.parquet")
        bench = bag.get("SPY.US")
    else:  # a
        pool = BA.load_pool(out_dir / "a_universe_idx.csv")
        bag = BA.fetch_data_a(pool, asof, out_dir / "backtest_entry_a_bag.parquet")
        bench = bag.get("000300.SH")
    return pool, bag, bench


VARIANTS = [
    ("全量",    lambda d: d["entry"]),
    ("SOS支",   lambda d: d["universe"] & d["cond_j"] & (d["short_trend"] >= 3) & d["cond_sos"]),
    ("非SOS支", lambda d: d["universe"] & d["cond_j"] & (d["short_trend"] >= 3) & d["cond_reversal"] & d["cond_trend"] & ~d["cond_sos"]),
    ("SOS裸",   lambda d: d["universe"] & d["cond_sos"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="hk,us,a")
    ap.add_argument("--asof", default="2026-07-24")
    ap.add_argument("--modes", default="strong,weak")
    ap.add_argument("--pure-sos", action="store_true",
                    help="只跑纯SOS: 全池, 无universe, 无止损(收盘对收盘)")
    args = ap.parse_args()

    for market in args.markets.split(","):
        m = market.strip()
        pool, bag, bench = load_market(m, args.asof)
        print(f"\n{'#'*70}\n# 市场={m.upper()} | asof={args.asof} | 池={len(pool)} 只有数据={len(bag)}\n{'#'*70}")
        df = compute_all_signals(pool, bag, bench)
        if df is None:
            print("无信号"); continue

        if args.pure_sos:
            # 纯SOS: 全池, 不分strong/weak, 无止损
            print(f"\n========= {m.upper()} 纯SOS (全池, 无止损) =========")
            evaluate_entries(df, df["sos"], direction="long", use_stops=False,
                             label=f"{m.upper()} 纯SOS", verbose=False)
            continue

        for mode in args.modes.split(","):
            mode = mode.strip()
            d = build_conds(df, mode, screen="bottom")
            print(f"\n========= {m.upper()} mode={mode} =========")
            for name, fn in VARIANTS:
                mask = fn(d)
                evaluate_entries(d, mask, direction="long",
                                 label=f"{m.upper()} {mode} {name}", verbose=False)


if __name__ == "__main__":
    main()
