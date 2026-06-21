"""美股板块热度聚类 + 趋势五部曲 (对标港股 sector_cluster/pullback_buypoint, 取数换 EODHD, 聚类换 yfinance Industry).

universe: configs/us_universe.csv (top600 by turnover, 含ADR剔ETF, yfinance Sector+Industry, 已持久化).
取数: EODHD /eod/{CODE}.US 单股日K (adjusted_close 可靠, 前复权). EOD 缓存 data/us_eod_raw_<asof>.pkl.
特征/打分: 复用港股 compute_features + composite 公式
  composite = 0.25·z(新高广度250) + 0.15·z(新高广度60) + 0.15·z(量放倍数)
              + 0.30·z(5日占比200日分位) + 0.15·z(60日涨幅)
聚类粒度: yfinance Industry (对标港股 sub_industry); Sector 用 yfinance Sector.

五部曲:
  Part1 最热行业(Industry)/GICS Sector   Part2 趋势个股(hotness, 60日>10%)
  Part3 异动放量(单日>4%+量 或 3日>10%+量, 排除Part2/4)
  Part4 趋势股回调买点(深回调10-25% + ≥2信号, 剔除大阴破位, KDJ底背离用 kdj_div_basic)
  Part5 资金轮动(5日占比200日分位·10日变化, 高位失宠/低位放量突破)

独立运行只打印 Part1-3; 完整5部曲 JSON 由 scripts/export_trend_us.py 组装.
用法: python src/us_trend.py [--asof 2026-06-18] [--top-stocks 20]
"""
from __future__ import annotations
import os, sys, io, time, argparse, pickle
from datetime import date, timedelta
import requests
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
UNIVERSE_CSV = os.path.join(PROJECT_DIR, "configs", "us_universe.csv")
OUT_JSON = os.path.join(PROJECT_DIR, "public", "data", "trend_hotspot_us.json")

EODHD_BASE = "https://eodhd.com/api"


def _load_eodhd_key():
    k = os.getenv("EODHD_KEY")
    if k:
        return k
    local = os.path.join(os.path.dirname(__file__), "..", "configs", "eodhd_key.local")
    if os.path.exists(local):
        return open(local, encoding="utf-8").read().strip()
    raise RuntimeError("EODHD_KEY 未设置: 设环境变量 EODHD_KEY, 或写入 configs/eodhd_key.local (gitignored)")


EODHD_KEY = _load_eodhd_key()

LOOKBACK_DAYS = 600
MIN_DAYS = 60

sys.path.insert(0, BASE_DIR)
from kdj_div_basic import calc_kdj, detect_divergence  # noqa: E402


# ---------------- 取数 ----------------

def load_pool() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE_CSV)
    df["Industry"] = df["Industry"].fillna("未分类").replace("", "未分类")
    df["Sector"] = df["Sector"].fillna("未分类").replace("", "未分类")
    return df[["Ticker", "Code", "Name", "Sector", "Industry", "MarketCap", "Price", "AvgTurnoverUSD"]]


def fetch_eod(code: str, asof: str) -> pd.DataFrame | None:
    to = asof
    frm = (pd.to_datetime(asof) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    for attempt in range(3):
        try:
            r = requests.get(f"{EODHD_BASE}/eod/{code}.US",
                             params={"api_token": EODHD_KEY, "from": frm, "to": to,
                                     "period": "d", "fmt": "json"}, timeout=60)
            r.raise_for_status()
            d = r.json()
            if not d:
                return None
            df = pd.DataFrame(d).sort_values("date").reset_index(drop=True)
            for c in ("open", "high", "low", "close", "adjusted_close", "volume"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            return df
        except Exception as e:
            if attempt == 2:
                print(f"  [warn] {code} fetch err: {e}")
                return None
            time.sleep(1.0)


def fetch_all(pool: pd.DataFrame, asof: str) -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"us_eod_raw_{asof}.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            raw = pickle.load(f)
        print(f"[Cache] 命中 {cache} ({raw['code'].nunique()} 只)")
        return raw
    frames = []
    codes = pool["Code"].tolist()
    for i, code in enumerate(codes):
        df = fetch_eod(code, asof)
        if df is None or len(df) < MIN_DAYS:
            continue
        df["code"] = code + ".US"
        frames.append(df[["code", "date", "open", "high", "low", "close",
                          "adjusted_close", "volume"]])
        if (i + 1) % 50 == 0:
            print(f"  fetched {i+1}/{len(codes)}")
        time.sleep(0.12)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    with open(cache, "wb") as f:
        pickle.dump(raw, f)
    print(f"[Cache] 写入 {cache} ({raw['code'].nunique()} 只)")
    return raw


# ---------------- 前复权 + 特征 (复用港股逻辑) ----------------

def forward_adjust_group(g: pd.DataFrame) -> pd.DataFrame | None:
    """前复权: 最新日 fwd_close == raw close. 同时产出 fwd_open/high/low (供KDJ/形态).
    amt = fwd_close × volume (USD, 一致调整序列). 保留 volume/raw_close 供周线重采样."""
    g = g.sort_values("date").reset_index(drop=True)
    lr = g["close"].iloc[-1]; la = g["adjusted_close"].iloc[-1]
    if pd.isna(lr) or pd.isna(la) or lr == 0:
        return None
    factor = la / lr
    close = g["close"].replace(0, np.nan)
    adjf = g["adjusted_close"] / close            # 逐日调整因子 (含拆股+分红)
    g["fwd_close"] = g["adjusted_close"] / factor
    g["fwd_open"] = g["open"] * adjf / factor
    g["fwd_high"] = g["high"] * adjf / factor
    g["fwd_low"] = g["low"] * adjf / factor
    g["raw_close"] = g["close"]
    g["vol"] = g["volume"]
    g["amt"] = g["fwd_close"] * g["vol"]
    return g


def compute_features(g: pd.DataFrame) -> dict | None:
    if g is None or len(g) < MIN_DAYS:
        return None
    c = g["fwd_close"].values
    amt = g["amt"].values
    n = len(g)
    close = float(c[-1])

    w250 = min(n, 250)
    high_250 = np.max(c[-w250:]); high_60 = np.max(c[-60:])
    pct_high_250 = close / high_250 if high_250 > 0 else np.nan
    pct_high_60 = close / high_60 if high_60 > 0 else np.nan

    s = pd.Series(c)
    roll60_max = s.rolling(60, min_periods=60).max()
    nh_60 = int((s[-60:] == roll60_max[-60:]).sum())
    nh_ratio_60 = nh_60 / 60.0
    roll_long = s.rolling(w250, min_periods=w250).max()
    nh_long = int((s[-w250:] == roll_long[-w250:]).sum())
    nh_ratio_250 = nh_long / float(w250)

    amt_20 = np.nanmean(amt[-20:]); amt_long = np.nanmean(amt[-w250:])
    amt_surge = amt_20 / amt_long if amt_long and amt_long > 0 else np.nan
    amt_1d = float(amt[-1]) if not np.isnan(amt[-1]) else np.nan
    amt_5d = float(np.nanmean(amt[-5:]))

    ret = lambda p: (close / c[-p - 1] - 1) if n > p and c[-p - 1] else None
    ret_20 = ret(20) if ret(20) is not None else np.nan
    ret_60 = ret(60) if ret(60) is not None else np.nan
    ret_120 = ret(120) if ret(120) is not None else (close / c[0] - 1)

    ma_windows = (10, 20, 30, 50, 100)
    ma = {w: s.rolling(w).mean().iloc[-1] for w in ma_windows}
    avail = [w for w in ma_windows if not pd.isna(ma[w])]
    aligned = int(all(ma[avail[i]] > ma[avail[i + 1]] for i in range(len(avail) - 1))) if len(avail) >= 4 else 0

    return dict(pct_high_250=pct_high_250, pct_high_60=pct_high_60,
                nh_ratio_60=nh_ratio_60, nh_ratio_250=nh_ratio_250,
                amt_20=amt_20, amt_surge=amt_surge, amt_1d=amt_1d, amt_5d=amt_5d,
                ret_20=ret_20, ret_60=ret_60, ret_120=ret_120,
                ma_aligned=aligned, close=close, days=n)


def _amt_long(raw: pd.DataFrame, pool: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """(code, date, group, amt) 长表 — 前复权后 fwd_close×volume, 按 pool 的 group_col 关联."""
    parts = []
    for code, g in raw.groupby("code"):
        gg = forward_adjust_group(g)
        if gg is None:
            continue
        parts.append(gg[["code", "date", "amt"]])
    long = pd.concat(parts, ignore_index=True)
    return long.merge(pool[["Ticker", group_col]], left_on="code", right_on="Ticker", how="inner")


def compute_share_pctile200(raw, pool, group_col) -> pd.Series:
    long = _amt_long(raw, pool, group_col)
    gd = long.groupby(["date", group_col])["amt"].sum().reset_index()
    td = long.groupby("date")["amt"].sum().reset_index().rename(columns={"amt": "tot"})
    gd = gd.merge(td, on="date").sort_values([group_col, "date"])
    gd["sub5"] = gd.groupby(group_col)["amt"].transform(lambda x: x.rolling(5, min_periods=5).sum())
    gd["tot5"] = gd.groupby(group_col)["tot"].transform(lambda x: x.rolling(5, min_periods=5).sum())
    gd["share5"] = np.where(gd["tot5"] > 0, gd["sub5"] / gd["tot5"], np.nan)

    def _pct(s):
        s = s.dropna()
        if len(s) < 10:
            return np.nan
        return float((s.iloc[-200:] <= s.iloc[-1]).mean() * 100)
    return gd.groupby(group_col)["share5"].apply(_pct)


def compute_share_rotation(raw, pool, group_col, lookback: int = 10) -> pd.DataFrame:
    """5日占比200日分位·过去lookback天变化 + 量能 → 资金轮动. flag: lose/gain/''."""
    long = _amt_long(raw, pool, group_col)
    gd = long.groupby(["date", group_col])["amt"].sum().reset_index()
    td = long.groupby("date")["amt"].sum().reset_index().rename(columns={"amt": "tot"})
    gd = gd.merge(td, on="date").sort_values([group_col, "date"])
    gd["sub5"] = gd.groupby(group_col)["amt"].transform(lambda x: x.rolling(5, min_periods=5).sum())
    gd["tot5"] = gd.groupby(group_col)["tot"].transform(lambda x: x.rolling(5, min_periods=5).sum())
    gd["share5"] = np.where(gd["tot5"] > 0, gd["sub5"] / gd["tot5"], np.nan)
    nmap = pool.groupby(group_col)["Ticker"].count()

    rows = []
    for gname, d in gd.groupby(group_col):
        d = d.reset_index(drop=True)
        s = d["share5"].values; a = d["sub5"].values
        L = len(s)
        if L < 60:
            continue

        def _pct(i):
            w = s[max(0, i - 199):i + 1]
            w = w[~np.isnan(w)]
            return float((w <= s[i]).mean() * 100) if len(w) >= 10 else np.nan

        now_i, ago_i = L - 1, max(0, L - 1 - lookback)
        pnow, pago = _pct(now_i), _pct(ago_i)
        if np.isnan(pnow) or np.isnan(pago):
            continue
        amt_now, amt_ago = float(a[now_i]), float(a[ago_i])
        amt_chg = (amt_now / amt_ago - 1) if amt_ago > 0 else None
        delta = pnow - pago
        flag = ""
        if pago >= 80 and pnow >= 60 and delta <= -15 and (amt_chg or 0) < -0.10:
            flag = "lose"
        elif pago <= 40 and pnow >= 50 and delta >= 30 and (amt_chg or 0) > 0.30:
            flag = "gain"
        rows.append({group_col: gname, "n": int(nmap.get(gname, 0)),
                     "pctile_now": round(pnow, 1), "pctile_ago": round(pago, 1),
                     "delta10": round(delta, 1),
                     "amt5_now_yi": round(amt_now / 1e8, 2) if amt_now else None,   # USD → 亿
                     "amt5_chg_pct": round(amt_chg * 100, 1) if amt_chg is not None else None,
                     "flag": flag})
    res = pd.DataFrame(rows)
    if len(res):
        res = res.sort_values("delta10", ascending=False).reset_index(drop=True)
    return res


# ---------------- 打分 ----------------

SCORE_COLS = ["breadth_250%", "breadth_60%", "mean_amt_surge", "share5_pctile200", "mean_ret_60"]


def hotspot_scores(pool, raw):
    """全池个股特征+hotness, Industry/Sector composite. 返回 (df_full, ind_rank, sector_rank)."""
    rows = []
    for code, g in raw.groupby("code"):
        gg = forward_adjust_group(g)
        f = compute_features(gg)
        if f is None:
            continue
        m = pool[pool["Ticker"] == code]
        if m.empty:
            continue
        m = m.iloc[0]
        rows.append({"Ticker": code, "Name": m["Name"], "Sector": m["Sector"],
                     "Industry": m["Industry"], **f})
    df = pd.DataFrame(rows).dropna(subset=["pct_high_250", "amt_surge", "ret_60"]).reset_index(drop=True)
    df["amt_rank_pct"] = df["amt_5d"].rank(pct=True)
    for col in ["pct_high_250", "nh_ratio_60", "amt_surge", "amt_rank_pct"]:
        df[f"z_{col}"] = (df[col] - df[col].mean()) / df[col].std()
    df["hotness"] = df[["z_pct_high_250", "z_nh_ratio_60", "z_amt_surge", "z_amt_rank_pct"]].mean(axis=1)

    pa1, pa5 = df["amt_1d"].sum(), df["amt_5d"].sum()
    pctile_ind = compute_share_pctile200(raw, pool, "Industry")
    pctile_sec = compute_share_pctile200(raw, pool, "Sector")

    def group_score(d):
        g1, g5 = d["amt_1d"].sum(), d["amt_5d"].sum()
        return pd.Series({
            "n": len(d),
            "mean_pct_high_250": d["pct_high_250"].mean(),
            "breadth_250%": (d["pct_high_250"] >= 0.95).mean() * 100,
            "breadth_60%": (d["pct_high_60"] >= 0.98).mean() * 100,
            "mean_amt_surge": d["amt_surge"].mean(),
            "mean_amt_rank": d["amt_rank_pct"].mean(),
            "总成交金额_1日_亿USD": round(g1 / 1e8, 2),
            "总成交金额_5日均值_亿USD": round(g5 / 1e8, 2),
            "占比_1日%": round(g1 / pa1 * 100, 2) if pa1 else 0.0,
            "占比_5日%": round(g5 / pa5 * 100, 2) if pa5 else 0.0,
            "mean_ret_60": d["ret_60"].mean(),
            "mean_hotness": d["hotness"].mean(),
        })

    def rank_group(col, min_n, pctile):
        g = df.groupby(col).apply(group_score, include_groups=False)
        g = g[g["n"] >= min_n].copy()
        g["share5_pctile200"] = g.index.map(pctile).astype(float).fillna(50.0)
        z = g.copy()
        for c in SCORE_COLS:
            std = z[c].std()
            z[c] = (z[c] - z[c].mean()) / std if std and std > 0 else 0.0
        g["composite"] = (0.25 * z["breadth_250%"] + 0.15 * z["breadth_60%"]
                          + 0.15 * z["mean_amt_surge"] + 0.30 * z["share5_pctile200"]
                          + 0.15 * z["mean_ret_60"])
        return g.sort_values("composite", ascending=False).round(3)

    ind_rank = rank_group("Industry", min_n=3, pctile=pctile_ind)
    sec_rank = rank_group("Sector", min_n=5, pctile=pctile_sec)
    return df, ind_rank, sec_rank


# ---------------- Part4 回调买点 + Part3 异动 ----------------

def analyze_pullback(g, ticker, name, ind, sector):
    """趋势股回调买点判定 (对标 pullback_buypoint.analyze). 返回候选 dict 或 None."""
    if g is None or len(g) < 210:
        return None
    g = g.copy()
    c = g["fwd_close"]
    close = float(c.iloc[-1])
    ma = {w: c.rolling(w).mean().iloc[-1] for w in (5, 10, 20, 50, 60, 200)}
    if any(pd.isna(v) for v in ma.values()):
        return None
    ma200_30ago = c.rolling(200).mean().iloc[-31]
    if pd.isna(ma200_30ago) or ma200_30ago == 0:
        return None
    ma200_ann = (ma[200] / ma200_30ago - 1) * (250 / 30)
    ret_120 = close / float(c.iloc[-121]) - 1 if len(c) > 120 else None
    if not ((close > ma[200]) and (ma200_ann > 0) and ret_120 is not None and ret_120 > 0.10):
        return None

    peak_60 = float(c.iloc[-60:].max())
    retrace = (peak_60 - close) / peak_60 if peak_60 > 0 else 0.0
    if not (0.10 <= retrace <= 0.25):
        return None

    # 排除: 最新大阴线跌破前期均衡(近5日低点)
    oo = float(g["fwd_open"].iloc[-1]); oh = float(g["fwd_high"].iloc[-1])
    ol = float(g["fwd_low"].iloc[-1]); oc = float(g["fwd_close"].iloc[-1])
    rng0 = oh - ol
    if rng0 > 0:
        body0 = abs(oc - oo) / rng0; loc0 = (oc - ol) / rng0
        prior5_low = float(g["fwd_low"].iloc[-6:-1].min())
        if oc < oo and body0 >= 0.60 and loc0 <= 0.35 and oc < prior5_low:
            return None

    # 买点信号
    o = g["fwd_open"].iloc[-5:].values; h = g["fwd_high"].iloc[-5:].values
    lo = g["fwd_low"].iloc[-5:].values; cl = g["fwd_close"].iloc[-5:].values
    doji = False
    for i in range(len(o)):
        rng = h[i] - lo[i]
        if rng <= 0:
            continue
        body = abs(cl[i] - o[i]); lsh = min(o[i], cl[i]) - lo[i]
        if body / rng <= 0.10:
            doji = True; break
        if body > 0 and lsh >= 2 * body and lsh >= 0.6 * rng:
            doji = True; break

    vol = g["vol"].values
    v30 = np.nanmean(vol[-30:])
    shrink = bool(v30 > 0 and np.nanmedian(vol[-5:]) / v30 <= 0.8)
    vals = np.array([ma[5], ma[10], ma[20]])
    entangle = bool((vals.max() - vals.min()) / np.median(vals) <= 0.02)
    try:
        kres = detect_divergence(calc_kdj(g))
        kdj_div = (kres["daily_divergence"] == "底背离") or (kres["weekly_divergence"] == "底背离")
    except Exception:
        kdj_div = False
    at_support = any(abs(close / ma[w] - 1) <= 0.02 for w in (20, 50, 60))
    n_sig = sum([doji, shrink, entangle, kdj_div, at_support])
    if n_sig < 2:
        return None

    return dict(Ticker=ticker, Name=name, Industry=ind, Sector=sector,
                Close=round(close, 2), Retrace=round(retrace * 100, 1),
                Doji=int(doji), Shrink=int(shrink), Entangle=int(entangle),
                KDJdiv=int(kdj_div), AtSupport=int(at_support), NSig=n_sig)


def screen_surge(raw, pool, exclude=None):
    """异动放量: 近3日单日>4%+1.5×30均量(且涨幅守住) 或 3日>10%+1.5×量. 排除 exclude. 返回 list[dict]."""
    exclude = exclude or set()
    out = []
    for code, g in raw.groupby("code"):
        gg = forward_adjust_group(g)
        if gg is None or len(gg) < 35:
            continue
        c = gg["fwd_close"].values; v = gg["vol"].values
        vol_ma = np.nanmean(v[-30:])
        if not vol_ma or vol_ma <= 0:
            continue
        ret = np.diff(c) / c[:-1] * 100.0
        single = False; single_ret = single_vr = None
        for i in range(max(0, len(ret) - 3), len(ret)):
            if ret[i] > 4 and v[i + 1] >= 1.5 * vol_ma and c[-1] >= c[i]:
                single = True; single_ret = float(ret[i]); single_vr = float(v[i + 1] / vol_ma); break
        ret3 = (c[-1] / c[-4] - 1) * 100.0 if len(c) >= 4 else None
        three_vr = float(np.nanmean(v[-3:]) / vol_ma) if vol_ma else None
        three = ret3 is not None and ret3 > 10 and three_vr is not None and three_vr >= 1.5
        if not (single or three) or code in exclude:
            continue
        meta = pool[pool["Ticker"] == code]
        if meta.empty:
            continue
        meta = meta.iloc[0]
        trig = "单日+3日" if single and three else ("单日" if single else "3日")
        out.append(dict(Ticker=code, Name=meta["Name"], Industry=meta["Industry"],
                        Sector=meta["Sector"], Close=round(float(c[-1]), 2),
                        Ret1d=round(single_ret, 1) if single else None,
                        Ret3d=round(ret3, 1) if ret3 is not None else None,
                        VolRatio=round(single_vr, 2) if single
                                 else (round(three_vr, 2) if three_vr is not None else None),
                        Trigger=trig))
    out.sort(key=lambda x: max(x.get("Ret3d") or 0, x.get("Ret1d") or 0), reverse=True)
    return out


# ---------------- 主流程 (Part1-3 快照) ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    ap.add_argument("--top-stocks", type=int, default=20)
    args = ap.parse_args()

    pool = load_pool()
    asof = args.asof or str(pd.read_csv(UNIVERSE_CSV)["LatestDate"].iloc[0])
    print(f"=== 美股板块热度聚类 (asof={asof}) ===\n[Pool] {len(pool)} 只")
    raw = fetch_all(pool, asof)
    print(f"[Fetch] {raw['code'].nunique()} 只有数据")

    df, ind_rank, sec_rank = hotspot_scores(pool, raw)
    hottest_ind = ind_rank.index[0] if len(ind_rank) else "N/A"
    print(f"[Feat] 有效 {len(df)} 只; 最热行业 = 【{hottest_ind}】 composite={ind_rank.loc[hottest_ind,'composite']}"
          if len(ind_rank) else "[Feat] 无行业排名")

    top_stocks = df.sort_values("hotness", ascending=False).head(args.top_stocks)
    hot_ind_stocks = df[df["Industry"] == hottest_ind].sort_values("hotness", ascending=False)

    pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 22)
    print("\n## Part 1  最热行业 (top 15, n>=3)")
    show1 = ["n", "breadth_250%", "breadth_60%", "mean_amt_surge",
             "总成交金额_5日均值_亿USD", "占比_5日%", "share5_pctile200", "mean_ret_60", "composite"]
    print(ind_rank[show1].head(15).to_string())
    print("\n## Part 1b  GICS Sector 热度 (n>=5)")
    print(sec_rank[show1].to_string())
    print(f"\n## Part 2  最热个股 (hotness top {args.top_stocks})")
    c2 = ["Ticker", "Name", "Industry", "close", "pct_high_250", "amt_surge",
          "amt_rank_pct", "ret_60", "ma_aligned", "hotness"]
    print(top_stocks[c2].round(3).to_string(index=False))
    print(f"\n## Part 3  最热行业【{hottest_ind}】个股热度排序")
    c3 = ["Ticker", "Name", "close", "pct_high_250", "nh_ratio_60",
          "amt_surge", "amt_rank_pct", "ret_60", "ma_aligned", "hotness"]
    print(hot_ind_stocks[c3].round(3).to_string(index=False))

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = {"date": asof, "hottest_ind": hottest_ind,
               "part1": {"industries": ind_rank.reset_index().to_dict("records"),
                         "sectors": sec_rank.reset_index().to_dict("records")},
               "part2": top_stocks.round(3).to_dict("records"),
               "part3": hot_ind_stocks.round(3).to_dict("records")}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        import json
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"\n[JSON] -> {OUT_JSON}  (注: 完整5部曲由 export_trend_us.py 生成)")


if __name__ == "__main__":
    main()
