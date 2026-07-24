"""筹码峰模块: 两种密度模型 + 峰检测 + 密集带命中

模型A 指数时间衰减:  w_t = exp(-(T-t)/tau), 每日量在[low,high]均匀分配, 累加
模型B 换手率衰减(经典通达信):  density_t = density_{t-1}*(1-turnover_t) + today_dist*turnover_t
                    turnover_t = min(volume_t / float_shares, cap)

两种模型输出归一化 density(份额分布, sum=1). 峰 = local max; 密集带 = density≥ratio×峰值的连续区间.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


def _bin_edges(daily: pd.DataFrame, nbins: int):
    lo = float(daily["fwd_low"].min())
    hi = float(daily["fwd_high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, None
    edges = np.linspace(lo, hi, nbins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return edges, centers


def _distribute(vol: float, low: float, high: float, edges: np.ndarray) -> np.ndarray:
    """单日成交量在 [low,high] 上均匀分配到 bin"""
    dens = np.zeros(len(edges) - 1)
    if vol <= 0 or not np.isfinite(low) or not np.isfinite(high):
        return dens
    if high <= low:
        idx = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, len(dens) - 1))
        dens[idx] += vol
        return dens
    i0 = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, len(dens) - 1))
    i1 = int(np.clip(np.searchsorted(edges, high, side="right") - 1, 0, len(dens) - 1))
    if i1 < i0:
        i1 = i0
    dens[i0:i1 + 1] += vol / (i1 - i0 + 1)
    return dens


def compute_density_exp(daily: pd.DataFrame, tau: float = 60, nbins: int = 200,
                        window: int = 120) -> tuple | None:
    """模型A: 指数时间衰减密度 (归一化)"""
    d = daily.tail(window).reset_index(drop=True)
    edges, centers = _bin_edges(d, nbins)
    if edges is None:
        return None
    n = len(d)
    density = np.zeros(len(centers))
    vol = d["volume"].values
    lo = d["fwd_low"].values
    hi = d["fwd_high"].values
    for t in range(n):
        w = np.exp(-(n - 1 - t) / tau)
        density += w * _distribute(float(vol[t]), float(lo[t]), float(hi[t]), edges)
    s = density.sum()
    if s <= 0:
        return None
    return centers, density / s


def compute_density_turnover(daily: pd.DataFrame, float_shares: float | None,
                             nbins: int = 200, window: int = 120,
                             cap: float = 0.99) -> tuple | None:
    """模型B: 换手率衰减密度 (归一化). float_shares 缺失返回 None"""
    if not float_shares or float_shares <= 0:
        return None
    d = daily.tail(window).reset_index(drop=True)
    edges, centers = _bin_edges(d, nbins)
    if edges is None:
        return None
    n = len(d)
    density = np.zeros(len(centers))
    vol = d["volume"].values
    lo = d["fwd_low"].values
    hi = d["fwd_high"].values
    for t in range(n):
        day_dist = _distribute(float(vol[t]), float(lo[t]), float(hi[t]), edges)
        s = day_dist.sum()
        if s <= 0:
            continue
        day_dist = day_dist / s
        turnover = min(float(vol[t]) / float_shares, cap)
        density = density * (1 - turnover) + day_dist * turnover
    s = density.sum()
    if s <= 0:
        return None
    return centers, density / s


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(np.pad(x, win // 2, mode="edge"), k, mode="same")[:len(x)]


def find_bands(centers: np.ndarray, density: np.ndarray,
               peak_order: int = 5, band_ratio: float = 0.50,
               sig_ratio: float = 0.30, smooth_win: int = 5) -> list:
    """返回显著密集带 [(lo, hi, peak_val, strength)]  strength=peak/全局max"""
    dens = _smooth(density, smooth_win)
    if dens.max() <= 0:
        return []
    gmax = dens.max()
    peaks = argrelextrema(dens, np.greater, order=peak_order)[0]
    # 末尾若为最大值, argrelextrema 漏掉, 补上
    if len(dens) > 0 and dens[-1] >= dens.max() * 0.999:
        if len(peaks) == 0 or peaks[-1] != len(dens) - 1:
            peaks = np.append(peaks, len(dens) - 1)
    bands = []
    for p in peaks:
        val = dens[p]
        if val < gmax * sig_ratio:
            continue
        lo_i, hi_i = int(p), int(p)
        while lo_i > 0 and dens[lo_i - 1] >= val * band_ratio:
            lo_i -= 1
        while hi_i < len(dens) - 1 and dens[hi_i + 1] >= val * band_ratio:
            hi_i += 1
        bands.append((float(centers[lo_i]), float(centers[hi_i]),
                      float(val), float(val / gmax)))
    # 合并重叠带
    bands.sort()
    merged = []
    for b in bands:
        if merged and b[0] <= merged[-1][1]:
            lo = min(merged[-1][0], b[0]); hi = max(merged[-1][1], b[1])
            val = max(merged[-1][2], b[2]); st = max(merged[-1][3], b[3])
            merged[-1] = (lo, hi, val, st)
        else:
            merged.append(b)
    return merged


def overlap_band(price_lo: float, price_hi: float, bands: list) -> tuple:
    """[price_lo,price_hi] 是否与某密集带重叠. 返回 (hit, (lo,hi,peak,strength))"""
    best = None
    for b in bands:
        if price_hi >= b[0] and price_lo <= b[1]:
            if best is None or b[3] > best[3]:
                best = b
    return (best is not None, best)
