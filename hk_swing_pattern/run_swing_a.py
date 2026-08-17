"""A股 Swing 宽表扫描入口: 全A top600 成交额池
  - 数据: jianxin Wind 库 ashareeodprices (WindFetcher, 含 S_DQ_TRADESTATUS='交易' 过滤) -> forward_adjust -> swing.analyze
  - universe: src/a_pool.py (a_pool.csv 缓存, 根目录), --rebuild 重建
  - RS: src/a_rs.py (MRS vs 510300.SH 沪深300 + 510500.SH 中证500)
  - 输出: A_Swing_Pattern.xlsx + 重建四 tab 首页 (gen_site)

用法:
  python run_swing_a.py                  # 截至今天
  python run_swing_a.py --asof 2026-08-14
  python run_swing_a.py --rebuild        # 重建 A股 top600 成交额池
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

import a_pool           # noqa: E402
import swing as sw      # noqa: E402
import a_rs             # noqa: E402
from data_provider import WindFetcher, forward_adjust  # noqa: E402
import run_swing        # noqa: E402  复用 COL_ORDER / write_excel
import gen_site         # noqa: E402

OUT_XLSX = "A_Swing_Pattern.xlsx"
# A股专属 7 列 (年线斜率 / MRS_沪深300+1W+4W / MRS_中证500+1W+4W)
COL_ORDER_A = run_swing.COL_ORDER + [
    "年线斜率", "MRS_沪深300", "MRS_沪深300_1W", "MRS_沪深300_4W",
    "MRS_中证500", "MRS_中证500_1W", "MRS_中证500_4W",
]


def to_bloomberg_a(code: str) -> str:
    """A股代码 -> 展示: 000001.SZ -> 000001 SZ (保留6位, 不去前导零)。"""
    return code.replace(".", " ")


def main():
    ap = argparse.ArgumentParser(description="A股 Swing 宽表")
    ap.add_argument("--asof", default=None)
    ap.add_argument("--rebuild", action="store_true", help="重建 A股 top600 成交额池缓存")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    asof = args.asof or cfg["data"]["asof"] or dt.date.today().strftime("%Y-%m-%d")
    out_dir = ROOT / cfg["output"]["dir"]; out_dir.mkdir(exist_ok=True)

    pool = a_pool.load_a_pool(ROOT / "a_pool.csv", asof, rebuild=args.rebuild)
    print(f"[Pool] {len(pool)} 只 | asof={asof}")

    lookback = cfg["data"]["lookback_days"]
    # WindFetcher: ashareeodprices 表, 仅取交易日
    fetcher = WindFetcher(lookback_days=lookback, table="ashareeodprices",
                          trade_status_filter="交易")
    # 基准: 510300 + 510500 (沪深300/中证500 ETF 代理)
    benchmarks = a_rs.fetch_benchmarks(asof, lookback)
    print(f"[Bench] 510300 {'OK' if '510300.SH' in benchmarks else '缺失'} | "
          f"510500 {'OK' if '510500.SH' in benchmarks else '缺失'}")

    rows, t0 = [], time.time()
    latest_dt = None
    for idx, (code, name, sector) in enumerate(pool):
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(pool)}] {time.time()-t0:.0f}s ...")
        try:
            raw = fetcher.fetch(code, asof)
        except Exception as e:
            print(f"  [ERR] {code}: {e}"); continue
        if raw is None or raw.empty:
            continue
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
        r["Ticker"] = to_bloomberg_a(code)
        r["Name"] = name
        r["Sector"] = sector
        rows.append(r)
    fetcher.close()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[COL_ORDER_A]
        def _has_sig(r):
            return any(str(r.get(c)) not in ("", "nan", "None") for c in
                       ["周度背离", "日度背离", "周度DeMark", "日度DeMark",
                        "KDJ Cross", "日MACD Cross", "5_10 Cross", "10_50 Cross"])
        df["_sig"] = df.apply(_has_sig, axis=1)
        df = df.sort_values(["_sig", "Ticker"], ascending=[False, True]).drop(columns="_sig").reset_index(drop=True)
    print(f"\n[完成] {len(df)} 只, 耗时 {time.time()-t0:.0f}s")
    if df.empty:
        print("无数据"); return
    # asof 修正为实际最新交易日
    if latest_dt is not None:
        real_asof = pd.Timestamp(latest_dt).strftime("%Y-%m-%d")
        if real_asof != asof:
            print(f"[asof] 运行日 {asof} 非交易日, 改用实际最新交易日 {real_asof}")
        asof = real_asof
    out_path = out_dir / OUT_XLSX
    try:
        run_swing.write_excel(df, asof, out_path, col_order=COL_ORDER_A)
    except PermissionError:
        out_path = out_dir / f"swing_a_{asof}_alt.xlsx"
        print(f"[WARN] 主文件被占用, 写到 {out_path.name}")
        run_swing.write_excel(df, asof, out_path, col_order=COL_ORDER_A)
    # 复制到根目录 + 重建四 tab 首页
    try:
        shutil.copy(out_path, ROOT / OUT_XLSX)
        gen_site.build(gen_site.default_panels(ROOT), ROOT / "index.html")
        print(f"[Site] -> {ROOT / 'index.html'}")
    except Exception as e:
        print(f"[Site] 生成失败: {e}")


if __name__ == "__main__":
    main()
