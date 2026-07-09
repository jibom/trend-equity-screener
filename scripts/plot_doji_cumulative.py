"""可视化: HSI(log) + RSI + 周KDJ + 信号累计 + 宽度 + 百分位 + 投降式抛售Z + 放量/大跌.

8行: HSI(log) / RSI(14) / 周KDJ(J) / 累计信号 / 宽度 / 百分位 / Capitulation Z / 放量·大跌%
6 项抄底条件, 满足 ≥4 标注 Cap X/6; 顶部表格 + 历史胜率/预期收益。
Crosshair 跨图联动。

用法: python scripts/plot_doji_cumulative.py
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

CSV = os.path.join(os.path.dirname(__file__), '..', 'output', 'doji_signal_daily.csv')
OUT = os.path.join(os.path.dirname(__file__), '..', 'output', 'doji_cumulative_vs_hsi.html')


def build(csv=CSV, idx_code='HSI.HI', idx_table='hkindexeodprices', market='HSI'):
    df = pd.read_csv(csv)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values('date').reset_index(drop=True)

    # 拉指数 OHLC (拉到今天, 不被 CSV 末日卡住 — CSV 的 breadth 可能滞后到上次 backtest)
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

    # Capitulation Z-score (20d): 当日 log 收益在【前 20 日】log 收益分布中的 z-score (ddof=1)。
    # Z <= -2.5 视为投降式抛售信号 (与 v7 反推精确匹配: corr=1.0)。
    hsi_raw['logret'] = np.log(hsi_raw['S_DQ_CLOSE'] / hsi_raw['S_DQ_CLOSE'].shift(1))
    _prior = hsi_raw['logret'].shift(1)
    hsi_raw['cap_z'] = (hsi_raw['logret'] - _prior.rolling(20).mean()) / _prior.rolling(20).std()

    # 用 hsi_raw 日期扩展 df: CSV 末日之后用新鲜 HSI 价格/RSI/cap_z, breadth 列留 NaN (宽度滞后到上次 backtest)
    df = (hsi_raw[['date', 'S_DQ_CLOSE', 'rsi', 'cap_z', 'logret']]
          .rename(columns={'S_DQ_CLOSE': 'hsi_close'})
          .merge(df.drop(columns=['hsi_close']), on='date', how='left'))
    df['w_kdj_j'] = np.nan
    df['w_kdj_j_low4'] = np.nan

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
    w_j_low4 = pd.Series(w_j).rolling(4, min_periods=4).min().values   # 过去4周最低J
    w_records = pd.DataFrame({'date': df_w.index, 'w_kdj_j': w_j, 'w_kdj_j_low4': w_j_low4})
    w_records['date'] = pd.to_datetime(w_records['date'])
    for _, wr in w_records.iterrows():
        mask = df['date'] >= wr['date']
        df.loc[mask, 'w_kdj_j'] = wr['w_kdj_j']
        df.loc[mask, 'w_kdj_j_low4'] = wr['w_kdj_j_low4']

    # df 已含 rsi/cap_z/logret (上面 extend 时并入), 无需再 merge

    # Rolling 累计信号
    for weeks, days in [(1, 5), (2, 10), (3, 15), (4, 20)]:
        df[f'cum_pct_{weeks}w'] = df['active_pct'].rolling(days, min_periods=days).sum()
    df['pctile_4w'] = df['cum_pct_4w'].expanding(min_periods=60).rank(pct=True) * 100

    # ---- 6 项抄底条件逐日打分 (HSI 上的 Cap X/6 标记 + 顶部表格 + 胜率统计) ----
    # 阈值用 expanding (当日只用过去数据, shift(1), 252日热身) — 无未来信息, 可诚实回测
    b90 = df['breadth_below_ma50'].expanding(min_periods=252).quantile(0.90).shift(1)
    b95 = df['breadth_below_ma50'].expanding(min_periods=252).quantile(0.95).shift(1)
    union99 = df["union_pct"].expanding(min_periods=252).quantile(0.99).shift(1)
    rsi_lb = 10
    df['c_rsi']  = (df['rsi'].rolling(rsi_lb, min_periods=1).min() < 30).astype(int)   # 近10日 RSI<30
    df['c_kdj']  = (df['w_kdj_j_low4'] < 10).fillna(False).astype(int)   # 过去4周最低J < 10 (与doji信号同口径)
    df['c_doji'] = (df['pctile_4w'] >= 90).astype(int)
    df['c_brd']  = (df['breadth_below_ma50'] >= b90).fillna(False).astype(int)
    df['c_cap']  = (df['cap_z'] <= -2.5).astype(int)
    df['c_union'] = (df['union_pct'] >= union99).fillna(False).astype(int)   # 放量∪大跌 ≥ 99th
    df['score']  = df[['c_rsi','c_kdj','c_doji','c_brd','c_cap','c_union']].sum(axis=1)
    # 前瞻 N 日 HSI 收益 (胜率/预期收益; 末尾 N 天为 NaN)
    for N in (5, 20, 60):
        df[f'fwd_{N}'] = df['hsi_close'].shift(-N) / df['hsi_close'] - 1

    fig = make_subplots(rows=8, cols=1, shared_xaxes='all', vertical_spacing=0.018,
                        row_heights=[0.17, 0.13, 0.09, 0.12, 0.10, 0.14, 0.09, 0.11],
                        subplot_titles=(
                            f'{market} Index (log scale)',
                            f'{market} RSI (14)',
                            f'{market} Weekly KDJ (J only)',
                            '底部十字星信号数',
                            '4-Week Cumulative % — Historical Percentile (50-100)',
                            'Breadth: % Stocks Below MA50',
                            'Capitulation Z-score (20d)',
                            '放量∪大跌个股% (量比>2 或 日跌>3%)',
                        ))

    # Row 1: HSI (log scale)
    fig.add_trace(go.Scatter(x=df['date'], y=df['hsi_close'], name='HSI',
                             line=dict(color='#1f77b4', width=1.5),
                             showlegend=False), row=1, col=1)

    # Row 1 叠加: Cap X/6 标记 (6 项条件满足 ≥4 即标注; 去重避免连续重叠)
    # 去重: score≥4 的日期按 gap>10 交易日分簇, 每簇取最高分(平手取最低价)作代表
    sig = df[df['score'] >= 4]
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
            best = sorted(c, key=lambda r: (-int(r['score']), r['hsi_close']))[0]
            picked.append(best.name)
    cap_df = df.loc[picked]
    cap_styles = {6: ('#c0392b', 18), 5: ('#e74c3c', 14), 4: ('#e67e22', 10)}
    for tier in (6, 5, 4):
        sub = cap_df[cap_df['score'] == tier]
        if sub.empty:
            continue
        txt = [f"{d.strftime('%Y-%m-%d')}<br>score={int(s)}/6<br>HSI={h:,.0f}"
               for s, h, d in zip(sub['score'], sub['hsi_close'], sub['date'])]
        col, sz = cap_styles[tier]
        fig.add_trace(go.Scatter(x=sub['date'], y=sub['hsi_close'], name=f'Cap {tier}/6',
                                 mode='markers', showlegend=False,
                                 marker=dict(symbol='triangle-up', color=col, size=sz,
                                             opacity=0.9, line=dict(color='white', width=1)),
                                 text=txt, hovertemplate='%{text}<extra></extra>'),
                     row=1, col=1)

    # Row 2: RSI (14) — 黑色线, 70/30 实线
    fig.add_trace(go.Scatter(x=df['date'], y=df['rsi'], name='RSI',
                             line=dict(color='#000000', width=1.2)), row=2, col=1)
    fig.add_hline(y=30, line_dash='solid', line_color='#26a69a', line_width=1, row=2, col=1)
    fig.add_hline(y=50, line_dash='dot', line_color='#999999', line_width=0.8, row=2, col=1)
    fig.add_hline(y=70, line_dash='solid', line_color='#ef5350', line_width=1, row=2, col=1)

    # Row 3: 周线 KDJ (只显示 J)
    fig.add_trace(go.Scatter(x=df['date'], y=df['w_kdj_j'], name='Weekly J',
                             line=dict(color='#e91e63', width=1.2)), row=3, col=1)
    for y, color in [(0, 'rgba(128,128,128,0.3)'), (50, 'rgba(128,128,128,0.2)'), (100, 'rgba(128,128,128,0.3)')]:
        fig.add_hline(y=y, line_dash='dot', line_color=color, line_width=0.5, row=3, col=1)

    # Row 4: rolling 累计占比
    colors = {1: 'rgba(255,99,71,0.6)', 2: 'rgba(255,165,0,0.7)', 3: 'rgba(100,149,237,0.7)', 4: 'rgba(128,0,128,0.8)'}
    for w in [1, 2, 3, 4]:
        fig.add_trace(go.Scatter(x=df['date'], y=df[f'cum_pct_{w}w'], name=f'{w}W cum %',
                                 line=dict(color=colors[w], width=1.2 if w < 4 else 2)), row=4, col=1)

    # Row 5: 百分位
    fig.add_trace(go.Scatter(x=df['date'], y=df['pctile_4w'], name='4W percentile',
                             line=dict(color='darkred', width=1.5),
                             fill='tozeroy', fillcolor='rgba(139,0,0,0.1)'), row=5, col=1)
    for pct, label, color in [(90, '90th', 'red'), (95, '95th', 'darkred')]:
        fig.add_hline(y=pct, line_dash='dash', line_color=color, line_width=0.8,
                      annotation_text=label, row=5, col=1)

    # Row 6: 宽度 (阈值线用 expanding 90th/95th, 随时间演化, 无未来信息)
    fig.add_trace(go.Scatter(x=df['date'], y=df['breadth_below_ma50'], name='Below MA50 %',
                             line=dict(color='#e91e63', width=1.5),
                             fill='tozeroy', fillcolor='rgba(233,30,99,0.1)'), row=6, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=b90, name='90th(exp)',
                             line=dict(color='red', width=0.8, dash='dash'), showlegend=False), row=6, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=b95, name='95th(exp)',
                             line=dict(color='darkred', width=0.8, dash='dash'), showlegend=False), row=6, col=1)

    # Row 7: Capitulation Z-score (20d) —— 当日 log 收益相对前 20 日的 z-score
    fig.add_trace(go.Scatter(x=df['date'], y=df['cap_z'], name='Cap Z-score',
                             line=dict(color='#9b59b6', width=1.5)), row=7, col=1)
    # 阈值线: -2.5 红色虚线 (投降式抛售阈值), -2.0 橙点线, 0 灰点线
    fig.add_hline(y=-2.5, line_dash='dash', line_color='#e74c3c', line_width=1, row=7, col=1)
    fig.add_hline(y=-2.0, line_dash='dot', line_color='#f39c12', line_width=0.8, row=7, col=1)
    fig.add_hline(y=0, line_dash='dot', line_color='rgba(100,100,100,0.4)', line_width=0.8, row=7, col=1)

    # Row 8: 放量∪大跌个股% —— 柱状图 + 99th 分位阈值线
    # width 按 ms (日期轴): 4 天, 相邻日柱重叠成实心红色带, 避免亚像素柱在白底上看不清
    fig.add_trace(go.Bar(x=df['date'], y=df['union_pct'], name='放量∪大跌%',
                         marker_color='#b71c1c', width=86400000*4, showlegend=False), row=8, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=union99, name='99th(exp)',
                             line=dict(color='#1a1a1a', width=1.2, dash='dash'), showlegend=False), row=8, col=1)

    # 注意: 不在 Python 端用 update_layout(shapes=...) —— 那会覆盖 add_hline 已加的横线。
    # 跨子图竖线不依赖 plotly spike/shape (本环境不渲染), 改由 post_script 注入 SVG overlay。
    fig.update_layout(
        title=f'{market} Doji Bottom Signal + RSI/KDJ/Breadth',
        height=1200, width=1200,
        template='plotly_white',
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1),
        hovermode='x',
        hoverdistance=20,
    )
    fig.update_yaxes(title_text='HSI (log)', type='log', row=1, col=1)
    fig.update_yaxes(title_text='RSI', range=[0, 100], tickvals=[0, 30, 70, 100], row=2, col=1)
    fig.update_yaxes(title_text='WJ', row=3, col=1)
    fig.update_yaxes(title_text='Cum %', row=4, col=1)
    fig.update_yaxes(title_text='Pctile', range=[50, 100], tickvals=[50, 75, 100], row=5, col=1)
    fig.update_yaxes(title_text='Below MA50 %', tickvals=[0, 25, 50, 75, 100], row=6, col=1)
    fig.update_yaxes(title_text='Cap Z', range=[-6, 3], tickvals=[-5, -2.5, -2, 0, 1, 2],
                    zeroline=True, zerolinecolor='rgba(150,150,150,0.5)', zerolinewidth=1, row=7, col=1)
    fig.update_yaxes(title_text='放量∪跌%', range=[0, 80], row=8, col=1)
    # 子图标题字号缩小
    fig.update_annotations(font_size=10)

    # post_script 移到模块级 CROSSHAIR_POSTSCRIPT; combined 脚本用 crosshair_js(div_id)

    # ---- 顶部「抄底信号仪表盘」表格 (6 项条件, 满足 ≥4 → 满仓干) ----
    last = df.iloc[-1]
    rsi_min10 = df['rsi'].tail(rsi_lb).min()
    rsi_met = bool((df['rsi'].tail(rsi_lb) < 30).any())
    b90_now = float(b90.iloc[-1]); u99_now = float(union99.iloc[-1])
    conds = [
        ('RSI(14)',          f'近{rsi_lb}日 RSI<30',     rsi_met,                            f"{last['rsi']:.1f} (低{rsi_min10:.1f})"),
        ('周KDJ.J(4周低)',    '4周最低<10',               bool(last['c_kdj']),                f"4周低{last['w_kdj_j_low4']:.1f}" if pd.notna(last['w_kdj_j_low4']) else 'NA'),
        ('十字星百分位',      '≥ 90',                     last['pctile_4w'] >= 90,            f"{last['pctile_4w']:.0f}"),
        ('市场宽度(<MA50)',   f'≥ {b90_now:.0f}% (90th)', bool(last['c_brd']),                f"{last['breadth_below_ma50']:.1f}%"),
        ('投降式抛售Z',       '≤ -2.5',                   last['cap_z'] <= -2.5,              f"{last['cap_z']:.2f}"),
        ('放量∪大跌%',       f'≥ {u99_now:.1f}% (99th)',  bool(last['c_union']),              f"{last['union_pct']:.1f}%"),
    ]
    n_met = sum(1 for _, _, met, _ in conds if met)
    trigger = n_met >= 4   # 满足 4 个即标注/满仓干
    # 扁平样式
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
    summary = f'满仓干！({n_met}/6)' if trigger else f'{n_met}/6 满足'
    summary_color = '#c0392b' if trigger else '#555'
    last_date = last['date'].strftime('%Y-%m-%d')
    # 历史胜率 + 预期收益: score≥4 的日期, 前瞻 5/20/60 日 HSI 收益
    mask = df['score'] >= 4
    n_sig = int(mask.sum())
    wr, er = {}, {}
    for N in (5, 20, 60):
        r = df.loc[mask, f'fwd_{N}'].dropna()
        wr[N] = (r > 0).mean() * 100 if len(r) else float('nan')
        er[N] = r.mean() * 100 if len(r) else float('nan')
    wr_txt = f"5日 {wr[5]:.0f}% · 20日 {wr[20]:.0f}% · 60日 {wr[60]:.0f}%"
    er_txt = f"5日 {er[5]:+.1f}% · 20日 {er[20]:+.1f}% · 60日 {er[60]:+.1f}%"
    th = "padding:3px 10px;border:1px solid #ccc;background:#f6f6f6;font-weight:600"
    table_html = (
        f"<table style='border-collapse:collapse;font-family:sans-serif;font-size:12px;margin:6px auto 0;'>"
        f"<caption style='caption-side:top;font-weight:bold;font-size:13px;margin-bottom:3px;'>"
        f"抄底信号仪表盘 <span style='color:#999;font-weight:normal;font-size:11px'>(截至 {last_date})</span></caption>"
        f"<thead><tr><th style='{th}'>指标</th><th style='{th}'>阈值</th><th style='{th}'>当前值</th><th style='{th}'>状态</th></tr></thead>"
        f"<tbody>{rows}</tbody>"
        f"<tfoot>"
        f"<tr><td colspan='3' style='padding:3px 10px;border:1px solid #ccc;text-align:right;font-weight:bold'>综合判断</td>"
        f"<td style='padding:3px 10px;border:1px solid #ccc;text-align:center;font-weight:bold;color:{summary_color};font-size:13px'>{summary}</td></tr>"
        f"<tr><td colspan='2' style='{cell};color:#666'>历史胜率 (score≥4, n={n_sig})</td><td colspan='2' style='{cell};color:#2e7d32'>{wr_txt}</td></tr>"
        f"<tr><td colspan='2' style='{cell};color:#666'>未来预期收益</td><td colspan='2' style='{cell};color:#1565c0'>{er_txt}</td></tr>"
        f"</tfoot></table>"
    )
    print(f"[Bottom] {summary} | 胜率5/20/60={wr[5]:.0f}/{wr[20]:.0f}/{wr[60]:.0f} 预期{er[5]:+.1f}/{er[20]:+.1f}/{er[60]:+.1f} 样本n={n_sig}")
    return fig, table_html


CROSSHAIR_POSTSCRIPT = """
    /* ---- Cross-subplot crosshair via SVG overlay ---- */
    var gd = arguments[0];
    (function(){
      var ns = 'http://www.w3.org/2000/svg';
      var grp = null;
      var svg = null;
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
        var xNames = ['xaxis','xaxis2','xaxis3','xaxis4','xaxis5','xaxis6','xaxis7','xaxis8'];
        var yNames = ['yaxis','yaxis2','yaxis3','yaxis4','yaxis5','yaxis6','yaxis7','yaxis8'];
        xNames.forEach(function(axName, i){
          var ax = fl[axName]; var yAx = fl[yNames[i]];
          if (!ax || !yAx) return;
          rects.push({ x: ax._offset, xlen: ax._length, y: yAx._offset, ylen: yAx._length });
        });
        return rects;
      }
      function drawCrosshair(clientX){
        if (!setup()) return;
        var svgRect = svg.getBoundingClientRect();
        var svgX = clientX - svgRect.left;
        while (grp.firstChild) grp.removeChild(grp.firstChild);
        var rects = getSubplotRects();
        if (!rects.length) return;
        var first = rects[0];
        if (svgX < first.x || svgX > first.x + first.xlen) return;
        rects.forEach(function(r){
          var line = document.createElementNS(ns, 'line');
          line.setAttribute('x1', svgX); line.setAttribute('x2', svgX);
          line.setAttribute('y1', r.y); line.setAttribute('y2', r.y + r.ylen);
          line.setAttribute('stroke', 'rgba(60,60,60,0.55)');
          line.setAttribute('stroke-width', '1.5');
          grp.appendChild(line);
        });
      }
      function clearCrosshair(){ if (grp) while (grp.firstChild) grp.removeChild(grp.firstChild); }
      gd.addEventListener('mousemove', function(e){ drawCrosshair(e.clientX); });
      gd.addEventListener('mouseleave', clearCrosshair);
      if (!setup()){
        var attempts = 0;
        (function retry(){ if (setup() || ++attempts > 60) return; requestAnimationFrame(retry); })();
      }
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
