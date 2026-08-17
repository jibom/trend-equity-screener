"""中国 ETF Swing 宽表扫描入口: 宽基+行业 curated 30 只
  - 数据: jianxin Wind 库 chinaclosedfundeodprice (WindFetcher) -> forward_adjust -> swing.analyze
  - universe: src/etf_pool.py (静态清单)
  - RS: src/a_rs.py (MRS vs 510300.SH 沪深300; 中证500列不算, ETF 对宽基基准)
  - 输出: ETF_Swing_Pattern.xlsx + 重建四 tab 首页 (gen_site)

用法:
  python run_swing_etf.py                  # 截至今天
  python run_swing_etf.py --asof 2026-08-14
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
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

import etf_pool          # noqa: E402
import swing as sw       # noqa: E402
import a_rs              # noqa: E402
from data_provider import WindFetcher, forward_adjust  # noqa: E402
import run_swing         # noqa: E402  复用 COL_ORDER / write_excel
import gen_site          # noqa: E402

OUT_XLSX = "ETF_Swing_Pattern.xlsx"
# ETF 专属 4 列 (年线斜率 / MRS_沪深300 + 1W + 4W); 无中证500
COL_ORDER_ETF = run_swing.COL_ORDER + [
    "年线斜率", "MRS_沪深300", "MRS_沪深300_1W", "MRS_沪深300_4W",
]


def to_bloomberg_etf(code: str) -> str:
    """ETF代码 -> 展示: 510300.SH -> 510300 SH。"""
    return code.replace(".", " ")


def main():
    ap = argparse.ArgumentParser(description="中国 ETF Swing 宽表")
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    asof = args.asof or cfg["data"]["asof"] or dt.date.today().strftime("%Y-%m-%d")
    out_dir = ROOT / cfg["output"]["dir"]; out_dir.mkdir(exist_ok=True)

    pool = etf_pool.load_etf_pool()
    print(f"[Pool] {len(pool)} 只 ETF | asof={asof}")

    lookback = cfg["data"]["lookback_days"]
    fetcher = WindFetcher(lookback_days=lookback, table="chinaclosedfundeodprice")
    benchmarks = a_rs.fetch_benchmarks(asof, lookback)
    print(f"[Bench] 510300 {'OK' if '510300.SH' in benchmarks else '缺失'}")

    rows, t0 = [], time.time()
    latest_dt = None
    for idx, (code, name, sector) in enumerate(pool):
        try:
            raw = fetcher.fetch(code, asof)
        except Exception as e:
            print(f"  [ERR] {code}: {e}"); continue
        if raw is None or raw.empty:
            print(f"  [SKIP] {code} 无数据"); continue
        daily = forward_adjust(raw)
        if daily is None or daily.empty or len(daily) < 60:
            continue
        daily = daily[daily["date"] <= pd.to_datetime(asof)].reset_index(drop=True)
        if len(daily) < 60:
            continue
        d_max = daily["date"].max()
        if latest_dt is None or d_max > latest_dt:
            latest_dt = d_max
        try:
            r = sw.analyze(daily)
            if r:
                r.update(a_rs.compute_extras(daily, benchmarks))
        except Exception as e:
            print(f"  [ERR] {code}: {e}"); r = None
        if r is None:
            continue
        r["Ticker"] = to_bloomberg_etf(code)
        r["Name"] = name
        r["Sector"] = sector
        rows.append(r)
    fetcher.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[COL_ORDER_ETF]
        def _has_sig(r):
            return any(str(r.get(c)) not in ("", "nan", "None") for c in
                       ["周度背离", "日度背离", "周度DeMark", "日度DeMark",
                        "KDJ Cross", "日MACD Cross", "5_10 Cross", "10_50 Cross"])
        df["_sig"] = df.apply(_has_sig, axis=1)
        df = df.sort_values(["_sig", "Ticker"], ascending=[False, True]).drop(columns="_sig").reset_index(drop=True)
    print(f"\n[完成] {len(df)} 只, 耗时 {time.time()-t0:.0f}s")
    if df.empty:
        print("无数据"); return
    if latest_dt is not None:
        real_asof = pd.Timestamp(latest_dt).strftime("%Y-%m-%d")
        if real_asof != asof:
            print(f"[asof] 运行日 {asof} 非交易日, 改用实际最新交易日 {real_asof}")
        asof = real_asof
    out_path = out_dir / OUT_XLSX
    try:
        run_swing.write_excel(df, asof, out_path, col_order=COL_ORDER_ETF)
    except PermissionError:
        out_path = out_dir / f"swing_etf_{asof}_alt.xlsx"
        print(f"[WARN] 主文件被占用, 写到 {out_path.name}")
        run_swing.write_excel(df, asof, out_path, col_order=COL_ORDER_ETF)
    try:
        shutil.copy(out_path, ROOT / OUT_XLSX)
        gen_site.build(gen_site.default_panels(ROOT), ROOT / "index.html")
        print(f"[Site] -> {ROOT / 'index.html'}")
    except Exception as e:
        print(f"[Site] 生成失败: {e}")


if __name__ == "__main__":
    main()
