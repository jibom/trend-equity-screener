"""Swing 特征组装: 把 patterns 各检测点 + 4 个综合判断列拼成一行 dict"""
from __future__ import annotations
import numpy as np
import pandas as pd
import patterns as P
import climax as CX

CLIMAX_PARAMS = dict(k_atr=4.0, v_mult=1.5, pos_lo=0.15, pos_hi=0.85)  # 回测胜率最高(bottom 20d 70.6%)


def analyze(daily: pd.DataFrame) -> dict | None:
    if daily is None or len(daily) < 60:
        return None
    d = daily.reset_index(drop=True)
    close = d["fwd_close"].values
    last_close = float(close[-1])

    je = P.weekly_j_extreme(daily)
    wk_rsi = P.weekly_rsi_swing(daily)
    div = P.all_divergences(daily)
    dm = P.demark_all(daily)
    cp = P.candlestick_patterns(daily)
    vp = P.volume_patterns(daily)
    co = P.crossovers(daily)

    # 背离汇总: +1 底 / -1 顶 / ±1 双向
    def _div_agg(keys, suppress_top=False, suppress_bot=False):
        bot = any(div[k] == "底背离" for k in keys) and not suppress_bot
        top = any(div[k] == "顶背离" for k in keys) and not suppress_top
        if bot and top: return "±1"
        if bot: return 1
        if top: return -1
        return ""
    # 周度: 周J(swing极值)超卖(<20)不显示顶背离; 超买(>80)不显示底背离
    sw_j = je["swing_extreme"]
    wk_oversold = sw_j is not None and sw_j < 20
    wk_overbought = sw_j is not None and sw_j > 80
    wk_div_agg = _div_agg(["周KDJ背离", "周MACD背离", "周RSI背离"],
                           suppress_top=wk_oversold, suppress_bot=wk_overbought)
    d_div_agg = _div_agg(["日KDJ背离", "日MACD背离", "日RSI背离"])
    # climax (ATR版, 取近5日最新): +1 最后一涨 / -1 最后一跌
    last5 = CX.climax_flags(daily, **CLIMAX_PARAMS)["flag"].tail(5).tolist()
    has_top = 1 in last5; has_bot = -1 in last5
    climax_val = "±1" if (has_top and has_bot) else (1 if has_top else (-1 if has_bot else ""))
    row = {
        "Close": round(last_close, 3),
        # ① 超卖超买 (周J/周RSI 仅输出近期 swing 极值)
        "周J": je["swing_extreme"],
        "周RSI": wk_rsi,
        # ② 趋势尾声
        "周KDJ背离": div["周KDJ背离"], "周MACD背离": div["周MACD背离"], "周RSI背离": div["周RSI背离"],
        "日KDJ背离": div["日KDJ背离"], "日MACD背离": div["日MACD背离"], "日RSI背离": div["日RSI背离"],
        "周度背离": wk_div_agg, "日度背离": d_div_agg,
        "周度DeMark": P.demark_col(dm, "周"), "日度DeMark": P.demark_col(dm, "日"),
        # ③ 多空平衡 (1=命中, 空=未命中)
        "十字星(5d)": cp["doji_count"],
        "涨放量跌缩量": 1 if vp["up_vol_dn_shrink"] else "",
        "climax": climax_val,
        # ④ 企稳上行 (近2日金叉=1/死叉=-1; 推算2日内将金叉=预1/将死叉=预-1)
        "KDJ Cross": P.cross_code(co["KDJ金叉"]),
        "日MACD Cross": P.cross_code(co["MACD金叉"]),
        "5_10 Cross": P.cross_code(co["5_10金叉"]),
        "10_50 Cross": P.cross_code(co["10_50金叉"]),
    }
    return row
