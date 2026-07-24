"""美股 Swing 宽表扫描入口: S&P500∪NDX100 + EU/JP/KR ADR
  - 数据: EODHD (src/eodhd.py) 并行拉取 -> forward_adjust -> swing.analyze
  - universe: src/us_pool.py (us_pool.csv 缓存, 根目录)
  - 输出: US_Swing_Pattern.xlsx + 重建两 tab 首页 (gen_site)

用法:
  python run_swing_us.py                  # 截至今天
  python run_swing_us.py --asof 2026-07-24
  python run_swing_us.py --rebuild        # 重建 pool (需联网 Wikipedia)
"""
from __future__ import annotations
import sys, argparse, datetime as dt, time, shutil
from pathlib import Path
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import yaml
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")  # EODHD_TOKEN
sys.path.insert(0, str(ROOT / "src"))

import eodhd            # noqa: E402
import swing as sw      # noqa: E402
import us_pool          # noqa: E402
from data_provider import forward_adjust  # noqa: E402
import run_swing        # noqa: E402  复用 COL_ORDER / write_excel
import gen_site         # noqa: E402

OUT_XLSX = "US_Swing_Pattern.xlsx"


def to_bloomberg(us_code: str) -> str:
    """EODHD 代码 -> Bloomberg: BRK-B.US -> BRK/B US; AAPL.US -> AAPL US。"""
    sym, suf = us_code.split(".")
    return f"{sym.replace('-', '/')} {suf}"


def main():
    ap = argparse.ArgumentParser(description="美股 Swing 宽表")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--rebuild", action="store_true", help="重建美股 universe 缓存")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    asof = args.asof or cfg["data"]["asof"] or dt.date.today().strftime("%Y-%m-%d")
    out_dir = ROOT / cfg["output"]["dir"]; out_dir.mkdir(exist_ok=True)

    pool = us_pool.load_us_pool(ROOT / "us_pool.csv", rebuild=args.rebuild)
    codes = [c for c, _, _ in pool]
    print(f"[Pool] {len(pool)} 只 | asof={asof}")

    lookback = cfg["data"]["lookback_days"]
    raw_map = eodhd.fetch_all_eodhd(codes, asof, lookback_days=lookback, workers=5)
    print(f"[EODHD] 取到 {len(raw_map)}/{len(codes)} 只日线")

    rows, t0 = [], time.time()
    for idx, (code, name, sector) in enumerate(pool):
        raw = raw_map.get(code)
        if raw is None or raw.empty:
            continue
        try:
            daily = forward_adjust(raw)
            if daily is None or daily.empty or len(daily) < 60:
                continue
            daily = daily[daily["date"] <= pd.to_datetime(asof)].reset_index(drop=True)
            if len(daily) < 60:
                continue
            r = sw.analyze(daily)
        except Exception as e:
            print(f"  [ERR] {code}: {e}"); r = None
        if r is None:
            continue
        r["Ticker"] = to_bloomberg(code)
        r["Name"] = name
        r["Sector"] = sector
        rows.append(r)
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(pool)}] {time.time()-t0:.0f}s ...")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[run_swing.COL_ORDER]
        def _has_sig(r):
            return any(str(r.get(c)) not in ("", "nan", "None") for c in
                       ["周度背离", "日度背离", "周度DeMark", "日度DeMark",
                        "KDJ Cross", "日MACD Cross", "5_10 Cross", "10_50 Cross"])
        df["_sig"] = df.apply(_has_sig, axis=1)
        df = df.sort_values(["_sig", "Ticker"], ascending=[False, True]).drop(columns="_sig").reset_index(drop=True)
    print(f"\n[完成] {len(df)} 只, 耗时 {time.time()-t0:.0f}s")
    if df.empty:
        print("无数据"); return
    out_path = out_dir / OUT_XLSX
    try:
        run_swing.write_excel(df, asof, out_path)
    except PermissionError:
        out_path = out_dir / f"swing_us_{asof}_alt.xlsx"
        print(f"[WARN] 主文件被占用, 写到 {out_path.name}")
        run_swing.write_excel(df, asof, out_path)
    # 复制到根目录 + 重建两 tab 首页
    try:
        shutil.copy(out_path, ROOT / OUT_XLSX)
        gen_site.build(ROOT / "HK_Swing_Pattern.xlsx", ROOT / OUT_XLSX, ROOT / "index.html")
        print(f"[Site] -> {ROOT / 'index.html'}")
    except Exception as e:
        print(f"[Site] 生成失败: {e}")


if __name__ == "__main__":
    main()
