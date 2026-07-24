"""Climax 信号回测: 全池 4 年历史, 扫参数, 统计 forward return + 胜率。
climax bottom(-1) 后应反弹(fwd正), climax top(+1) 后应回落(fwd负)。
用法: python backtest_climax.py --asof 2026-07-20
"""
from __future__ import annotations
import sys, io, argparse, datetime as dt, time, itertools
from pathlib import Path
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, r"D:\equity-trend-screener\src")
import provider as P
import climax as CX

HORIZONS = (5, 10, 20, 60)
LOOKBACK_DAYS = 1500  # ~4 年


def collect_events(fetcher, pool, asof, cache_path):
    """拉每只股票长历史, 返回 {code: daily_df}。带 parquet 缓存。"""
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return {code: g.drop(columns=["code"]).reset_index(drop=True)
                for code, g in df.groupby("code")}
    bag = {}
    t0 = time.time()
    for idx, (code, name, sector) in enumerate(pool):
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(pool)}] {time.time()-t0:.0f}s ...")
        daily = P.fetch_daily(fetcher, code, asof)
        if daily is not None and len(daily) > 300:
            bag[code] = daily
    # 缓存
    frames = [df.assign(code=c) for c, df in bag.items()]
    pd.concat(frames, ignore_index=True).to_parquet(cache_path)
    return bag


def evaluate(bag, params):
    """对一组参数, 汇总全池 climax 事件 + forward return。"""
    frames = []
    for code, daily in bag.items():
        ev = CX.events_with_fwd(daily, horizons=HORIZONS, **params)
        if not ev.empty:
            ev["code"] = code
            frames.append(ev)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def summarize(ev: pd.DataFrame) -> dict:
    """按 type 统计 n/胜率/avg/median forward return。"""
    out = {}
    for t, label in ((1, "top"), (-1, "bottom")):
        sub = ev[ev["type"] == t]
        d = {"n": len(sub)}
        for h in HORIZONS:
            col = sub[f"fwd{h}"].dropna()
            if len(col) == 0:
                d[f"fwd{h}_avg"] = np.nan; d[f"fwd{h}_win"] = np.nan; continue
            # top 后应跌(win=负); bottom 后应涨(win=正)
            win = (col < 0).mean() if t == 1 else (col > 0).mean()
            d[f"fwd{h}_avg"] = col.mean()
            d[f"fwd{h}_win"] = win
        out[label] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2026-07-20")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    out_dir = ROOT / cfg["output"]["dir"]; out_dir.mkdir(exist_ok=True)

    pool = P.load_pool(cfg["data"]["pool_csv"])
    print(f"[Pool] {len(pool)} 只 | asof={args.asof} | 回测窗口~{LOOKBACK_DAYS}日")
    fetcher = P.make_fetcher(LOOKBACK_DAYS)
    bag = collect_events(fetcher, pool, args.asof, out_dir / "climax_bag.parquet")
    fetcher.close()
    print(f"[Data] {len(bag)} 只有足够历史")

    # 参数扫描
    param_grid = [
        dict(k_atr=k, v_mult=v, pos_lo=lo, pos_hi=hi)
        for k in (2.5, 3.0, 4.0)
        for v in (1.5, 2.0, 3.0)
        for (lo, hi) in ((0.20, 0.80), (0.15, 0.85))
    ]
    print(f"\n[扫描] {len(param_grid)} 组参数\n")
    rows = []
    for params in param_grid:
        ev = evaluate(bag, params)
        if ev is None or ev.empty:
            rows.append({**params, "n_top": 0, "n_bot": 0})
            continue
        s = summarize(ev)
        rows.append({
            **params,
            "n_top": s["top"]["n"], "n_bot": s["bottom"]["n"],
            "top_fwd10_avg": s["top"]["fwd10_avg"], "top_fwd10_win": s["top"]["fwd10_win"],
            "top_fwd20_avg": s["top"]["fwd20_avg"], "top_fwd20_win": s["top"]["fwd20_win"],
            "bot_fwd10_avg": s["bottom"]["fwd10_avg"], "bot_fwd10_win": s["bottom"]["fwd10_win"],
            "bot_fwd20_avg": s["bottom"]["fwd20_avg"], "bot_fwd20_win": s["bottom"]["fwd20_win"],
        })
    sweep = pd.DataFrame(rows)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
    print("=== 参数扫描 (top 后应跌 win↑/avg负; bottom 后应涨 win↑/avg正) ===")
    print(sweep.round(3).to_string(index=False))
    sweep.to_csv(out_dir / "climax_param_sweep.csv", index=False, encoding="utf-8-sig")

    # 默认参数详细分布
    default = dict(k_atr=3.0, v_mult=2.0, pos_lo=0.20, pos_hi=0.80)
    ev = evaluate(bag, default)
    if ev is not None and not ev.empty:
        ev.to_csv(out_dir / "climax_events_default.csv", index=False, encoding="utf-8-sig")
        s = summarize(ev)
        print(f"\n=== 默认参数 {default} 详细 ===")
        print(f"top 事件 n={s['top']['n']}, bottom 事件 n={s['bottom']['n']}")
        for t, label in ((1, "top(后应跌)"), (-1, "bottom(后应涨)")):
            key = "top" if t == 1 else "bottom"
            print(f"  {label}:")
            for h in HORIZONS:
                a = s[key][f"fwd{h}_avg"]; w = s[key][f"fwd{h}_win"]
                print(f"    fwd{h}: avg={a:+.3f}  win={w:.1%}")
        print(f"\n[CSV] {out_dir/'climax_events_default.csv'}")
    print(f"[CSV] {out_dir/'climax_param_sweep.csv'}")


if __name__ == "__main__":
    main()
