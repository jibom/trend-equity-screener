"""扫描: 遍历池 → 三层合成 → 收集信号"""
from __future__ import annotations
import time
import pandas as pd
import provider as P
import synth as sig


def run(asof: str, cfg: dict, fetcher) -> tuple[pd.DataFrame, dict, dict]:
    pool = P.load_pool(cfg["data"]["pool_csv"])
    codes = [c for c, _, _ in pool]
    print(f"[Pool] {len(pool)} 只 | asof={asof}")

    today_iso = asof.replace("-", "")
    # yfinance 缓存按自然日; 用 asof 当天作 key (回测时也按 asof 日)
    float_map = P.fetch_float_shares(codes, today_iso)
    n_float = sum(1 for v in float_map.values() if v)
    print(f"[Float] 流通股本有效 {n_float}/{len(codes)} (换手率模型覆盖)")

    rows, gmap = [], {}
    t0 = time.time()
    for idx, (code, name, sector) in enumerate(pool):
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(pool)}] {time.time()-t0:.0f}s ...")
        daily = P.fetch_daily(fetcher, code, asof)
        if daily is None:
            continue
        try:
            r = sig.analyze(daily, float_map.get(code), cfg)
        except Exception as e:
            print(f"  [ERR] {code}: {e}")
            r = None
        if r is None:
            continue
        r["Ticker"] = code
        r["Name"] = name
        r["Sector"] = sector
        rows.append(r)
        # 保留所有信号股的 daily 供前端绘图
        gmap[code] = daily

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Side", "Score", "WeeksSinceDiv"],
                            ascending=[True, False, True]).reset_index(drop=True)
    print(f"\n[完成] {len(df)} 个信号, 耗时 {time.time()-t0:.0f}s")
    return df, gmap, float_map
