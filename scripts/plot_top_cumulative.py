"""可视化(逃顶): 镜像 plot_doji_cumulative。HSI(log) + RSI + 周KDJ + 顶部信号累计 + 宽度 + 百分位 + 投降式上涨Z + 放量/大涨.

8行: HSI(log) / RSI(14) / 周KDJ(J) / 顶部信号累计 / 百分位 / 宽度 / Blow-off Z / 放量·大涨%
6 项逃顶条件, 满足 ≥4 标注 Cap X/6 (向下三角, 标在 HSI 高点); 顶部表格 + 历史下跌概率/预期收益。
胜率 = 信号后 5/20/60 日 HSI 下跌概率 (fwd<0)。Crosshair 跨图联动。

用法: python scripts/plot_top_cumulative.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymysql
from db_config import DB_CONFIG
from kdj_div_basic import calc_kdj

CSV = os.path.join(os.path.dirname(__file__), '..', 'output', 'top_signal_daily.csv')
OUT = os.path.join(os.path.dirname(__file__), '..', 'output', 'top_escape_vs_hsi.html')


def build(csv=CSV, idx_code='HSI.HI', idx_table='hkindexeodprices', market='HSI'):
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)

    # 拉指数 OHLC (拉到今天, 不被 CSV 末日卡住 — breadth 可能滞后到上次 backtest)
    conn = pymysql.connect(**DB_CONFIG)
    start = df['date'].min().strftime('%Y%m%d')
    end = pd.Timestamp.today().strftime('%Y%m%d')
    hsi_raw = pd.read_sql(
        f"SELECT TRADE_DT, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE FROM {idx_table} "
        f"WHERE S_INFO_WINDCODE='{idx_code}' AND TRADE_DT BETWEEN '{start}' AND '{end}' ORDER BY TRADE_DT", conn)
    conn.close()
    hsi_raw['date'] = pd.to_datetime(hsi_raw['TRADE_DT'], format='%Y%m%d')
    hsi_raw = hsi_raw.sort_values('date').reset_index(drop=True)
    for c in ['S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']:
        hsi_raw[c] = hsi_raw[c].astype(float)

    # RSI (14)
    hsi_close = hsi_raw['S_DQ_CLOSE']
    delta = hsi_close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    hsi_raw['rsi'] = (100 - 100 / (1 + rs)).fillna(50)

    # Blow-off Z-score (20d): 当日 log 收益在【前 20 日】log 收益分布的 z-score。Z >= 2.5 = 投降式上涨/冲刺。
    hsi_raw['logret'] = np.log(hsi_raw['S_DQ_CLOSE'] / hsi_raw['S_DQ_CLOSE'].shift(1))
    _prior = hsi_raw['logret'].shift(1)
    hsi_raw['cap_z'] = (hsi_raw['logret'] - _prior.rolling(20).mean()) / _prior.rolling(20).std()

    # 用 hsi_raw 日期扩展 df: CSV 末日之后用新鲜 HSI 价格/RSI/cap_z, breadth 列留 NaN
    df = (hsi_raw[['date', 'S_DQ_CLOSE', 'rsi', 'cap_z', 'logret']]
          .rename(columns={'S_DQ_CLOSE': 'hsi_close'})
          .merge(df.drop(columns=['hsi_close']), on='date', how='left'))
    df['w_kdj_j'] = np.nan
    df['w_kdj_j_high4'] = np.nan

    # 周线 KDJ
    df_d = pd.DataFrame({
        'date': hsi_raw['date'],
        'high': hsi_raw['S_DQ_HIGH'].values,
        'low': hsi_raw['S_DQ_LOW'].values,
        'close': hsi_raw['S_DQ_CLOSE'].values,
    }).set_index('date')
    df_w = df_d.resample('W-FRI').agg({'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    w_kdj = calc_kdj(pd.DataFrame({
        'fwd_close': df_w['close'].values,
        'fwd_high': df_w['high'].values,
        'fwd_low': df_w['low'].values,
    }))
    w_j = w_kdj['j'].values
    w_j_high4 = pd.Series(w_j).rolling(4, min_periods=4).max().values   # 过去4周最高J
    w_records = pd.DataFrame({'date': df_w.index, 'w_kdj_j': w_j, 'w_kdj_j_high4': w_j_high4})
    w_records['date'] = pd.to_datetime(w_records['date'])
    for _, wr in w_records.iterrows():
        mask = df['date'] >= wr['date']
        df.loc[mask, 'w_kdj_j'] = wr['w_kdj_j']
        df.loc[mask, 'w_kdj_j_high4'] = wr['w_kdj_j_high4']

    # df 已含 rsi/cap_z/logret (上面 extend 时并入), 无需再 merge

    # Rolling 累计顶部信号
    for weeks, days in [(1, 5), (2, 10), (3, 15), (4, 20)]:
        df[f'cum_pct_{weeks}w'] = df['active_top_pct'].rolling(days, min_periods=days).sum()
    df['pctile_4w'] = df['cum_pct_4w'].expanding(min_periods=60).rank(pct=True) * 100

    # ---- 9 项逃顶条件逐日打分 (砍掉 c_kdj噪声 / c_union反向; c_div/c_bias放宽到95th) ----
    b10 = df['breadth_below_ma50'].expanding(min_periods=252).quantile(0.10).shift(1)
    b05 = df['breadth_below_ma50'].expanding(min_periods=252).quantile(0.05).shift(1)
    union_up99 = df["union_up_pct"].expanding(min_periods=252).quantile(0.99).shift(1)   # 仅作 row8 展示
    sky99   = df["sky_vol_pct"].expanding(min_periods=252).quantile(0.99).shift(1)
    div95   = df["vol_price_div_pct"].expanding(min_periods=252).quantile(0.95).shift(1)
    dist95  = df["dist_top_pct"].expanding(min_periods=252).quantile(0.95).shift(1)
    shrink95 = df["shrink_new_high_pct"].expanding(min_periods=252).quantile(0.95).shift(1)
    # 指数 BIAS20 = (close-MA20)/MA20, 情绪顶用 expanding 95th (99th 太严, 漏2018类窄基顶)
    hsi_ma20 = hsi_raw['S_DQ_CLOSE'].rolling(20, min_periods=20).mean()
    df['bias20'] = (hsi_raw['S_DQ_CLOSE'] / hsi_ma20 - 1) * 100
    bias95 = df['bias20'].expanding(min_periods=252).quantile(0.95).shift(1)
    rsi_lb = 10
    # "让子弹飞": 极度超买类信号回溯近 LOOK 日 (同步指标在转折日回落, 回溯让信号覆盖顶部区域而非掐尖)
    LOOK = 10
    def recent(s_bool):
        return s_bool.rolling(LOOK, min_periods=1).max().fillna(0).astype(int)
    df['c_rsi']    = (df['rsi'].rolling(rsi_lb, min_periods=1).max() > 70).astype(int)            # 近10日 RSI>70
    df['c_kdj']    = recent((df['w_kdj_j_high4'] > 100).fillna(False))                            # 近10日 4周高J曾>100 (极端超买)
    df['c_star']   = (df['pctile_4w'] >= 90).astype(int)                                          # 射击之星(高位)累计百分位 ≥ 90
    df['c_brd']    = (df['breadth_below_ma50'] <= b10).fillna(False).astype(int)                  # 宽度 ≤ 10th (today-only, 68% wr60)
    df['c_cap']    = (df['cap_z'] >= 2.5).astype(int)                                             # Blow-off Z ≥ 2.5 (today-only)
    df['c_sky']    = (df['sky_vol_pct'] >= sky99).fillna(False).astype(int)                       # 天量天价 ≥ 99th
    df['c_div']    = (df['vol_price_div_pct'] > 0).astype(int)                                    # 量价背离 ≥1股
    df['c_dist']   = (df['dist_top_pct'] >= dist95).fillna(False).astype(int)                     # 放量滞涨 ≥ 95th
    df['c_shrink'] = (df['shrink_new_high_pct'] >= shrink95).fillna(False).astype(int)            # 缩量新高 ≥ 95th
    df['c_bias']   = recent((df['bias20'] >= bias95).fillna(False))                               # 近10日 BIAS曾≥95th
    COND_COLS = ['c_rsi','c_kdj','c_star','c_brd','c_cap','c_sky','c_div','c_dist','c_shrink','c_bias']
    df['score']   = df[COND_COLS].sum(axis=1)
    # 前瞻 N 日 HSI 收益 (胜率=下跌概率; 末尾 N 天 NaN)
    for N in (5, 20, 60):
        df[f'fwd_{N}'] = df['hsi_close'].shift(-N) / df['hsi_close'] - 1

    fig = make_subplots(rows=10, cols=1, shared_xaxes='all', vertical_spacing=0.012,
                        row_heights=[0.14, 0.10, 0.07, 0.09, 0.08, 0.11, 0.07, 0.09, 0.10, 0.08],
                        subplot_titles=(
                            f'{market} Index (log scale)',
                            f'{market} RSI (14)',
                            f'{market} Weekly KDJ (J only)',
                            '顶部反转信号数',
                            '4-Week Cumulative % — Historical Percentile (50-100)',
                            'Breadth: % Stocks Below MA50 (越低越狂热)',
                            'Blow-off Z-score (20d)',
                            '放量∪大涨个股% (量比>2 或 日涨>3%)',
                            '量价见顶信号百分位 (0-100, 各detector expanding rank)',
                            f'{market} BIAS20 = (close/MA20 - 1)×100 (情绪顶)',
                        ))

    # Row 1: HSI (log scale)
    fig.add_trace(go.Scatter(x=df['date'], y=df['hsi_close'], name='HSI',
                             line=dict(color='#1f77b4', width=1.5), showlegend=False), row=1, col=1)

    # Row 1 叠加: Cap X/10 顶部标记 (向下三角, 去重: gap>10 分簇, 每簇取最高分, 平手取最高价=峰)
    sig = df[df['score'] >= 5]
    picked = []
    if not sig.empty:
        sig_pos = df['date'].searchsorted(sig['date'])
        clusters, cur, cur_pos = [], [sig.iloc[0]], [sig_pos[0]]
        for i in range(1, len(sig)):
            if sig_pos[i] - cur_pos[-1] <= 10:
                cur.append(sig.iloc[i]); cur_pos.append(sig_pos[i])
            else:
                clusters.append(cur); cur, cur_pos = [sig.iloc[i]], [sig_pos[i]]
        if cur: clusters.append(cur)
        for c in clusters:
            # 顶部取最高分, 平手取最高价 (峰)
            best = sorted(c, key=lambda r: (-int(r['score']), -r['hsi_close']))[0]
            picked.append(best.name)
    cap_df = df.loc[picked]
    cap_styles = {8: ('#4a148c', 20), 6: ('#6a1b9a', 16), 5: ('#9575cd', 11)}
    for tier in (8, 6, 5):
        sub = cap_df[cap_df['score'] == tier]
        if sub.empty:
            continue
        txt = [f"{d.strftime('%Y-%m-%d')}<br>score={int(s)}/10<br>HSI={h:,.0f}"
               for s, h, d in zip(sub['score'], sub['hsi_close'], sub['date'])]
        col, sz = cap_styles[tier]
        fig.add_trace(go.Scatter(x=sub['date'], y=sub['hsi_close'], name=f'Cap {tier}/6',
                                 mode='markers', showlegend=False,
                                 marker=dict(symbol='triangle-down', color=col, size=sz,
                                             opacity=0.9, line=dict(color='white', width=1)),
                                 text=txt, hovertemplate='%{text}<extra></extra>'),
                     row=1, col=1)

    # Row 2: RSI — 70/30 实线
    fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], name='RSI',
                             line=dict(color='#000000', width=1.2)), row=2, col=1)
    fig.add_hline(y=30, line_dash='solid', line_color='#26a69a', line_width=1, row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='#999999', line_width=0.8, row=2, col=1)
    fig.add_hline(y=70, line_dash='solid', line_color='#ef5350', line_width=1, row=2, col=1)

    # Row 3: 周线 KDJ
    fig.add_trace(go.Scatter(x=df['date'], y=df['w_kdj_j'], name='Weekly J',
                             line=dict(color='#e91e63', width=1.2)), row=3, col=1)
    for y, color in [(0, 'rgba(128,128,128,0.3)'), (50, 'rgba(128,128,128,0.2)'), (100, 'rgba(128,128,128,0.3)')]:
        fig.add_hline(y=y, line_dash='dot', line_color=color, line_width=0.5, row=3, col=1)

    # Row 4: rolling 累计顶部占比
    colors = {1: 'rgba(255,99,71,0.6)', 2: 'rgba(255,165,0,0.7)', 3: 'rgba(100,149,237,0.7)', 4: 'rgba(106,27,154,0.8)'}
    for w in [1, 2, 3, 4]:
        fig.add_trace(go.Scatter(x=df['date'], y=df[f'cum_pct_{w}w'], name=f'{w}W cum %',
                                 line=dict(color=colors[w], width=1.2 if w < 4 else 2)), row=4, col=1)

    # Row 5: 百分位
    fig.add_trace(go.Scatter(x=df['date'], y=df['pctile_4w'], name='4W percentile',
                             line=dict(color='indigo', width=1.5),
                             fill='tozeroy', fillcolor='rgba(75,0,130,0.1)'), row=5, col=1)
    for pct, label, color in [(90, '90th', 'purple'), (95, '95th', 'indigo')]:
        fig.add_hline(y=pct, line_dash='dash', line_color=color, line_width=0.8,
                      annotation_text=label, row=5, col=1)

    # Row 6: 宽度 (顶部看 ≤10th/5th, 随时间演化, 无未来信息)
    fig.add_trace(go.Scatter(x=df['date'], y=df['breadth_below_ma50'], name='Below MA50 %',
                             line=dict(color='#e91e63', width=1.5),
                             fill='tozeroy', fillcolor='rgba(233,30,99,0.1)'), row=6, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=b10, name='10th(exp)',
                             line=dict(color='purple', width=0.8, dash='dash'), showlegend=False), row=6, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=b05, name='5th(exp)',
                             line=dict(color='indigo', width=0.8, dash='dash'), showlegend=False), row=6, col=1)

    # Row 7: Blow-off Z-score
    fig.add_trace(go.Scatter(x=df['date'], y=df['cap_z'], name='Blow-off Z',
                             line=dict(color='#9b59b6', width=1.5)), row=7, col=1)
    fig.add_hline(y=2.5, line_dash='dash', line_color='#8e24aa', line_width=1, row=7, col=1)
    fig.add_hline(y=2.0, line_dash='dot', line_color='#f39c12', line_width=0.8, row=7, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(100,100,100,0.4)', line_width=0.8, row=7, col=1)

    # Row 8: 放量∪大涨% + 99th
    fig.add_trace(go.Bar(x=df['date'], y=df['union_up_pct'], name='放量∪大涨%',
                         marker_color='#6a1b9a', width=86400000*4, showlegend=False), row=8, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=union_up99, name='99th(exp)',
                             line=dict(color='#1a1a1a', width=1.2, dash='dash'), showlegend=False), row=8, col=1)

    # Row 9: 量价见顶信号百分位 (4个detector各自 expanding rank, 0-100; ≥90 即触发)
    vol_det_cols = [('sky_vol_pct', '天量天价', '#d81b60'),
                    ('vol_price_div_pct', '量价背离', '#f9a825'),
                    ('dist_top_pct', '放量滞涨', '#6a1b9a'),
                    ('shrink_new_high_pct', '缩量新高', '#00897b')]
    for col, label, color in vol_det_cols:
        pctile = df[col].expanding(min_periods=252).rank(pct=True) * 100
        fig.add_trace(go.Scatter(x=df['date'], y=pctile, name=label,
                                 line=dict(color=color, width=1.2), showlegend=True), row=9, col=1)
    fig.add_hline(y=90, line_dash='dash', line_color='red', line_width=0.8, row=9, col=1)
    fig.add_hline(y=95, line_dash='dot', line_color='darkred', line_width=0.6, row=9, col=1)

    # Row 10: HSI BIAS20 + expanding 99th
    fig.add_trace(go.Scatter(x=df['date'], y=df['bias20'], name='BIAS20',
                             line=dict(color='#ef6c00', width=1.5),
                             fill='tozeroy', fillcolor='rgba(239,108,0,0.08)'), row=10, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=bias95, name='95th(exp)',
                             line=dict(color='#1a1a1a', width=1.0, dash='dash'), showlegend=False), row=10, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(100,100,100,0.4)', line_width=0.8, row=10, col=1)

    fig.update_layout(
        title=f'{market} Top-Escape Signal + RSI/KDJ/Breadth/Volume-Structure',
        height=1500, width=1200,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.005, xanchor='right', x=1),
        hovermode='x', hoverdistance=20,
    )
    fig.update_yaxes(title_text='HSI (log)', type='log', row=1, col=1)
    fig.update_yaxes(title_text='RSI', range=[0, 100], tickvals=[0, 30, 70, 100], row=2, col=1)
    fig.update_yaxes(title_text='WJ', row=3, col=1)
    fig.update_yaxes(title_text='Cum %', row=4, col=1)
    fig.update_yaxes(title_text='Pctile', range=[50, 100], tickvals=[50, 75, 100], row=5, col=1)
    fig.update_yaxes(title_text='Below MA50 %', tickvals=[0, 25, 50, 75, 100], row=6, col=1)
    fig.update_yaxes(title_text='Blow Z', range=[-3, 6], tickvals=[-2, 0, 2, 2.5, 5],
                    zeroline=True, zerolinecolor='rgba(150,150,150,0.5)', zerolinewidth=1, row=7, col=1)
    fig.update_yaxes(title_text='放量∪涨%', range=[0, 80], row=8, col=1)
    fig.update_yaxes(title_text='量价pctile', range=[0, 100], tickvals=[0, 50, 90, 100], row=9, col=1)
    fig.update_yaxes(title_text='BIAS20', row=10, col=1)
    fig.update_annotations(font_size=9)

    # post_script 移到模块级 CROSSHAIR_POSTSCRIPT (供 standalone main 使用); combined 脚本用 crosshair_js(div_id)

    # ---- 顶部「逃顶信号仪表盘」表格 (11 项条件, 满足 ≥5 → 逃顶警告) ----
    last = df.iloc[-1]
    rsi_max10 = df['rsi'].tail(rsi_lb).max()
    rsi_met = bool((df['rsi'].tail(rsi_lb) > 70).any())
    b10_now = float(b10.iloc[-1])
    sky_now = float(sky99.iloc[-1]); div_now = float(div95.iloc[-1])
    dist_now = float(dist95.iloc[-1]); shrink_now = float(shrink95.iloc[-1]); bias_now = float(bias95.iloc[-1])
    _pctile = lambda c: df[c].expanding(min_periods=252).rank(pct=True).iloc[-1] * 100
    conds = [
        ('RSI(14)',          f'近{rsi_lb}日 RSI>70',      rsi_met,                              f"{last['rsi']:.1f} (高{rsi_max10:.1f})"),
        ('周KDJ.J(4周高)',    '近10日 曾>100',             bool(last['c_kdj']),                  f"4周高{last['w_kdj_j_high4']:.1f}" if pd.notna(last['w_kdj_j_high4']) else 'NA'),
        ('射击之星(高位)百分位','≥ 90',                    last['pctile_4w'] >= 90,              f"{last['pctile_4w']:.0f}"),
        ('市场宽度(<MA50)',   f'≤ {b10_now:.0f}% (10th)',  bool(last['c_brd']),                  f"{last['breadth_below_ma50']:.1f}%"),
        ('Blow-off Z',       '≥ 2.5',                     bool(last['c_cap']),                  f"{last['cap_z']:.2f}"),
        ('天量天价%',        f'≥ {sky_now:.1f}% (99th)',  bool(last['c_sky']),                  f"{last['sky_vol_pct']:.1f}% (pct{_pctile('sky_vol_pct'):.0f})"),
        ('量价背离%',        '≥1股 (>0)',                 bool(last['c_div']),                  f"{last['vol_price_div_pct']:.1f}% (pct{_pctile('vol_price_div_pct'):.0f})"),
        ('放量滞涨%',        f'≥ {dist_now:.1f}% (95th)', bool(last['c_dist']),                 f"{last['dist_top_pct']:.1f}% (pct{_pctile('dist_top_pct'):.0f})"),
        ('缩量新高%',        f'≥ {shrink_now:.1f}% (95th)',bool(last['c_shrink']),              f"{last['shrink_new_high_pct']:.1f}% (pct{_pctile('shrink_new_high_pct'):.0f})"),
        ('BIAS20',           f'近10日 曾≥{bias_now:.1f}%(95th)', bool(last['c_bias']),          f"{last['bias20']:.1f}%"),
    ]
    n_met = sum(1 for _, _, met, _ in conds if met)
    today_trig = n_met >= 5
    # "让子弹飞": 过去 PERSIST 交易日内出现过 score≥5 信号 → 信号仍有效 (簇内保持)
    PERSIST = 10
    sig_idx = df.index[df['score'] >= 5]
    recent_info, recent_trig = '', False
    if len(sig_idx):
        ri = sig_idx[-1]
        days_ago = len(df) - 1 - ri
        if days_ago <= PERSIST:
            rrow = df.iloc[ri]
            recent_info = f"🔔 近期信号 {rrow['date'].strftime('%Y-%m-%d')} score={int(rrow['score'])}/10 · {days_ago} 交易日前 · 仍有效"
            recent_trig = True
    trigger = today_trig or recent_trig
    cell = "padding:2px 10px;border:1px solid #e0e0e0;line-height:1.3"
    rows = ''
    for name, cond, met, val in conds:
        chk = '✅' if met else '⬜'
        rows += (
            f"<tr><td style='{cell}'>{name}</td>"
            f"<td style='{cell};color:#888'>{cond}</td>"
            f"<td style='{cell};text-align:right'>{val}</td>"
            f"<td style='{cell};text-align:center'>{chk}</td></tr>"
        )
    if today_trig:
        summary = f'逃顶警告！({n_met}/10)'
    elif recent_trig:
        summary = f'信号仍有效 ({n_met}/10)'
    else:
        summary = f'{n_met}/10 满足'
    summary_color = '#6a1b9a' if trigger else '#555'
    last_date = last['date'].strftime('%Y-%m-%d')
    # 历史下跌概率 + 预期收益: score≥阈值 的日期, 前瞻 5/20/60 日 HSI 收益 (下跌=信号应验)
    mask = df['score'] >= 5
    n_sig = int(mask.sum())
    wr, er = {}, {}
    for N in (5, 20, 60):
        r = df.loc[mask, f'fwd_{N}'].dropna()
        wr[N] = (r < 0).mean() * 100 if len(r) else float('nan')   # 下跌概率
        er[N] = r.mean() * 100 if len(r) else float('nan')
    wr_txt = f"5日 {wr[5]:.0f}% · 20日 {wr[20]:.0f}% · 60日 {wr[60]:.0f}%"
    er_txt = f"5日 {er[5]:+.1f}% · 20日 {er[20]:+.1f}% · 60日 {er[60]:+.1f}%"
    th = "padding:3px 10px;border:1px solid #ccc;background:#f6f6f6;font-weight:600"
    recent_row = (f"<tr><td colspan='4' style='{cell};color:#6a1b9a;font-weight:600;text-align:center;background:#faf5ff'>"
                  f"{recent_info}</td></tr>") if recent_info else ''
    table_html = (
        f"<table style='border-collapse:collapse;font-family:sans-serif;font-size:12px;margin:6px auto 0;'>"
        f"<caption style='caption-side:top;font-weight:bold;font-size:13px;margin-bottom:3px;'>"
        f"逃顶信号仪表盘 <span style='color:#999;font-weight:normal;font-size:11px'>(截至 {last_date})</span></caption>"
        f"<thead><tr><th style='{th}'>指标</th><th style='{th}'>阈值</th><th style='{th}'>当前值</th><th style='{th}'>状态</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"<tfoot>"
        f"{recent_row}"
        f"<tr><td colspan='3' style='padding:3px 10px;border:1px solid #ccc;text-align:right;font-weight:bold'>综合判断</td>"
        f"<td style='padding:3px 10px;border:1px solid #ccc;text-align:center;font-weight:bold;color:{summary_color};font-size:13px'>{summary}</td></tr>"
        f"<tr><td colspan='2' style='{cell};color:#666'>历史下跌概率 (score≥5, n={n_sig})</td><td colspan='2' style='{cell};color:#6a1b9a'>{wr_txt}</td></tr>"
        f"<tr><td colspan='2' style='{cell};color:#666'>未来预期收益</td><td colspan='2' style='{cell};color:#c0392b'>{er_txt}</td></tr>"
        f"</tfoot></table>"
    )
    print(f"[Top] {summary} | 下跌概率5/20/60={wr[5]:.0f}/{wr[20]:.0f}/{wr[60]:.0f} 预期{er[5]:+.1f}/{er[20]:+.1f}/{er[60]:+.1f} 样本n={n_sig}")
    # 多阈值校准表 (score>=k 的下跌概率/预期/样本)
    print("\n--- 多阈值校准 (score>=k) ---")
    print(f"{'k':>3} {'n':>5} {'wr5':>5} {'wr20':>5} {'wr60':>5} {'er5':>7} {'er20':>7} {'er60':>7}")
    for k in (3, 4, 5, 6, 7):
        m = df['score'] >= k
        n = int(m.sum())
        if n == 0:
            continue
        w = {N: (df.loc[m, f'fwd_{N}'].dropna() < 0).mean() * 100 for N in (5, 20, 60)}
        e = {N: df.loc[m, f'fwd_{N}'].dropna().mean() * 100 for N in (5, 20, 60)}
        print(f"{k:>3} {n:>5} {w[5]:>5.0f} {w[20]:>5.0f} {w[60]:>5.0f} {e[5]:>+7.1f} {e[20]:>+7.1f} {e[60]:>+7.1f}")
    # 各detector触发频率 (condition 命中率, 看哪些有边际信号)
    print("\n--- 各条件命中率 ---")
    for c in COND_COLS:
        print(f"  {c:12s} hit={int(df[c].sum()):>5}/{len(df)} ({df[c].mean()*100:.1f}%)")
    return fig, table_html


CROSSHAIR_POSTSCRIPT = """
    var gd = arguments[0];
    (function(){
      var ns = 'http://www.w3.org/2000/svg';
      var grp = null, svg = null;
      function setup(){
        svg = gd.querySelector('.main-svg');
        if (!svg) return false;
        if (grp) return true;
        grp = document.createElementNS(ns, 'g');
        grp.id = 'crosshair-grp';
        grp.style.pointerEvents = 'none';
        svg.appendChild(grp);
        return true;
      }
      function getSubplotRects(){
        var fl = gd._fullLayout; var rects = [];
        var xN = ['xaxis','xaxis2','xaxis3','xaxis4','xaxis5','xaxis6','xaxis7','xaxis8','xaxis9','xaxis10'];
        var yN = ['yaxis','yaxis2','yaxis3','yaxis4','yaxis5','yaxis6','yaxis7','yaxis8','yaxis9','yaxis10'];
        xN.forEach(function(axName, i){
          var ax = fl[axName], yAx = fl[yN[i]];
          if (!ax || !yAx) return;
          rects.push({ x: ax._offset, xlen: ax._length, y: yAx._offset, ylen: yAx._length });
        });
        return rects;
      }
      function draw(clientX){
        if (!setup()) return;
        var r = svg.getBoundingClientRect();
        var sx = clientX - r.left;
        while (grp.firstChild) grp.removeChild(grp.firstChild);
        var rects = getSubplotRects(); if (!rects.length) return;
        var f = rects[0];
        if (sx < f.x || sx > f.x + f.xlen) return;
        rects.forEach(function(rc){
          var ln = document.createElementNS(ns, 'line');
          ln.setAttribute('x1', sx); ln.setAttribute('x2', sx);
          ln.setAttribute('y1', rc.y); ln.setAttribute('y2', rc.y + rc.ylen);
          ln.setAttribute('stroke', 'rgba(60,60,60,0.55)');
          ln.setAttribute('stroke-width', '1.5');
          grp.appendChild(ln);
        });
      }
      function clear(){ if (grp) while (grp.firstChild) grp.removeChild(grp.firstChild); }
      gd.addEventListener('mousemove', function(e){ draw(e.clientX); });
      gd.addEventListener('mouseleave', clear);
      if (!setup()){ var a=0; (function retry(){ if (setup()||++a>60) return; requestAnimationFrame(retry); })(); }
    })();
    """


def main():
    fig, table_html = build()
    fig.write_html(OUT, include_plotlyjs=True, post_script=CROSSHAIR_POSTSCRIPT)
    with open(OUT, 'r', encoding='utf-8') as f:
        html = f.read()
    if '<body>' in html:
        html = html.replace('<body>', '<body>\n' + table_html, 1)
    else:
        idx = html.find('class="plotly-graph-div"')
        if idx > 0:
            div_start = html.rfind('<div', 0, idx)
            html = html[:div_start] + table_html + '\n' + html[div_start:]
    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[Done] -> {OUT}")


if __name__ == '__main__':
    main()
