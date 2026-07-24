"""三层信号合成: 周线背离(context) + 日线doji(trigger) + 筹码峰(conviction) → 评分

底买: 周线底背离(≤10w) + 近N日doji + 筹码密集带命中
顶卖: 周线顶背离(≤10w) + 近N日doji + 筹码密集带命中  (对称)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from kdj_divergence import calc_kdj, resample_weekly
from kdj_strict_screener import detect_weekly_divergence
import chip as chipmod


def _atr14(daily: pd.DataFrame) -> float:
    h = daily["fwd_high"].values; l = daily["fwd_low"].values; c = daily["fwd_close"].values
    n = len(h)
    if n < 15:
        return np.nan
    tr = np.zeros(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return float(pd.Series(tr).rolling(14, min_periods=14).mean().iloc[-1])


def _is_doji_arr(daily: pd.DataFrame, range_min: float, body_to_rng: float) -> np.ndarray:
    o = daily["fwd_open"].values; h = daily["fwd_high"].values
    l = daily["fwd_low"].values; c = daily["fwd_close"].values
    body = np.abs(c - o)
    rng = h - l
    with np.errstate(invalid="ignore", divide="ignore"):
        b2r = np.where(rng > 0, body / rng, 1.0)
        r2o = np.where(o > 0, rng / o, 0.0)
    typeA = (b2r <= body_to_rng) & (r2o >= range_min)
    typeB = body <= np.where(c >= 5, 0.02, 0.01)
    return typeA | typeB


def _vol_div_signal(daily: pd.DataFrame, lookback: int = 10) -> tuple:
    """返回 (up_avg_vol, dn_avg_vol) 近 lookback 日"""
    d = daily.tail(lookback)
    pct = d["fwd_close"].pct_change()
    vol = d["volume"]
    up = vol[pct > 0]; dn = vol[pct < 0]
    return (float(up.mean()) if len(up) else np.nan,
            float(dn.mean()) if len(dn) else np.nan)


def analyze(daily: pd.DataFrame, float_shares: float | None, cfg: dict) -> dict | None:
    """返回信号 dict 或 None(无 context/trigger)"""
    if daily is None or len(daily) < 60:
        return None
    d = calc_kdj(daily.copy())

    # ---- 1. 周线背离 (context) ----
    div_cfg = cfg["divergence"]; dj = cfg["doji"]; ch = cfg["chip"]; sc = cfg["score"]
    r = detect_weekly_divergence(
        d, lookback_weeks=div_cfg["lookback_weeks"],
        j_oversold=div_cfg["j_oversold"], j_overbought=div_cfg["j_overbought"],
        max_weeks_since_div=div_cfg["max_weeks_since_div"],
    )
    div_type = r["divergence"]
    if not div_type or r["div_count"] == 0:
        return None
    if r["weeks_since_div"] is not None and r["weeks_since_div"] > div_cfg["max_weeks_since_div"]:
        return None

    side = "底买" if div_type == "底背离" else "顶卖"
    anchor = float(r["div_details"][-1]["price2"])  # 周线第二极值价

    # ---- 2. 日线 doji (trigger) ----
    is_doji = _is_doji_arr(d, dj["range_min"], dj["body_to_rng"])
    win = dj["recent_window"]
    recent_doji_idx = np.where(is_doji[-win:])[0]
    if len(recent_doji_idx) == 0:
        return None
    trig_i = len(d) - win + int(recent_doji_idx[-1])  # 最近一个 doji
    trig = d.iloc[trig_i]
    doji_lo = float(trig["fwd_low"]); doji_hi = float(trig["fwd_high"])
    doji_close = float(trig["fwd_close"])

    # ---- 3. 筹码峰 (conviction) ----
    exp = chipmod.compute_density_exp(d, tau=ch["exp_tau"], nbins=ch["nbins"],
                                      window=cfg["data"]["chip_window"])
    tur = chipmod.compute_density_turnover(d, float_shares, nbins=ch["nbins"],
                                           window=cfg["data"]["chip_window"],
                                           cap=ch["turnover_cap"])
    exp_bands = chipmod.find_bands(exp[0], exp[1], ch["peak_order"], ch["peak_band_ratio"],
                                   ch["sig_peak_ratio"], ch["smooth_win"]) if exp else []
    tur_bands = chipmod.find_bands(tur[0], tur[1], ch["peak_order"], ch["peak_band_ratio"],
                                   ch["sig_peak_ratio"], ch["smooth_win"]) if tur else []
    exp_hit, exp_band = chipmod.overlap_band(doji_lo, doji_hi, exp_bands)
    tur_hit, tur_band = chipmod.overlap_band(doji_lo, doji_hi, tur_bands)

    # ---- 4. 评分组件 ----
    atr = _atr14(d)
    near = (np.isfinite(atr) and abs(doji_close - anchor) <= dj["near_atr_k"] * atr)

    up_v, dn_v = _vol_div_signal(d, 10)
    if side == "底买":
        vol_div = np.isfinite(up_v) and np.isfinite(dn_v) and up_v > dn_v
    else:
        vol_div = np.isfinite(up_v) and np.isfinite(dn_v) and dn_v > up_v

    vol_ma20 = d["volume"].rolling(20, min_periods=1).mean().iloc[trig_i]
    trig_vol_ratio = float(trig["volume"] / vol_ma20) if vol_ma20 > 0 else np.nan
    quiet = np.isfinite(trig_vol_ratio) and trig_vol_ratio < sc["quiet_vol_ratio"]

    # 周线 J 极值
    wk = calc_kdj(resample_weekly(d))
    j40 = wk.tail(40)["j"]
    j_min = float(j40.min()); j_max = float(j40.max())
    if side == "底买":
        j_extreme = j_min <= sc["j_extreme_bottom"]
    else:
        j_extreme = j_max >= sc["j_extreme_top"]

    wk_signal = bool(r["signal"])

    dk = r["daily_kdj_k"]; dd_ = r["daily_kdj_d"]
    if side == "底买":
        daily_sync = dk is not None and dd_ is not None and dk > dd_ and dk < 80
    else:
        daily_sync = dk is not None and dd_ is not None and dk < dd_ and dk > 20

    # ---- 5. 汇总得分 ----
    score = sc["base"]
    score += int(exp_hit) * sc["chip_hit_each"]
    score += int(tur_hit) * sc["chip_hit_each"]
    score += int(near) * sc["near_anchor"]
    score += int(vol_div) * sc["vol_div"]
    score += int(quiet) * sc["quiet_doji"]
    score += int(j_extreme) * sc["j_extreme"]
    score += int(wk_signal) * sc["weekly_signal"]
    score += int(daily_sync) * sc["daily_sync"]

    if score >= sc["tier_high"]:
        tier = "高"
    elif score >= sc["tier_mid"]:
        tier = "中"
    else:
        tier = "低"

    return {
        "Side": side, "Score": score, "Tier": tier,
        "DivType": div_type, "DivDate": r["latest_div_date"], "WeeksSinceDiv": r["weeks_since_div"],
        "Anchor": round(anchor, 3),
        "TrigDate": str(trig["date"].date()), "TrigClose": round(doji_close, 3),
        "DojiLow": round(doji_lo, 3), "DojiHigh": round(doji_hi, 3),
        "周K": r["weekly_kdj_k"], "周D": r["weekly_kdj_d"], "周J": r["weekly_kdj_j"],
        "40wJmin": round(j_min, 2), "40wJmax": round(j_max, 2),
        "ChipExp": "✓" if exp_hit else "·",
        "ChipTur": "✓" if tur_hit else ("·" if tur else "N/A"),
        "ExpBand": f"{exp_band[0]:.2f}-{exp_band[1]:.2f}" if exp_band else "",
        "TurBand": f"{tur_band[0]:.2f}-{tur_band[1]:.2f}" if tur_band else "",
        "NearAnchor": int(near), "VolDiv": int(vol_div), "QuietDoji": int(quiet),
        "JExtreme": int(j_extreme), "WkSignal": int(wk_signal), "DailySync": int(daily_sync),
        "VolRatio": round(trig_vol_ratio, 2) if np.isfinite(trig_vol_ratio) else None,
        "FloatShares": float(float_shares) if float_shares else None,
    }
