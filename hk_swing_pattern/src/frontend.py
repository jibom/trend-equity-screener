"""前端 HTML 生成: 股票名单 + 方向键翻页 + 多子图(日K线+均线/周KDJ/日MACD/RSI/筹码峰) + 贯穿 crosshair

风格参考 equity-trend-screener (红涨绿跌 K线 + 6 均线)。crosshair 用 SVG overlay 注入
(本机 plotly spike 不渲染), 贯穿前 4 行日期轴子图; 第 5 行筹码峰按价格轴与 K 线对齐。
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import get_plotlyjs

import chip as chipmod
from kdj_divergence import calc_kdj, resample_weekly


def _plotly_bundle() -> str:
    """提取正确的 plotly.js bundle (root.Plotly 全局)。
    get_plotlyjs() 返回 module 版 (root.moduleName), Plotly 未定义;
    to_html(include_plotlyjs='inline') 嵌的才是正确版本。"""
    import re
    dummy = go.Figure(go.Scatter(y=[1, 2, 3]))
    frag = dummy.to_html(include_plotlyjs="inline", full_html=False)
    scripts = re.findall(r"<script[^>]*>(.*?)</script>", frag, re.S)
    return max(scripts, key=len)

MA_WINDOWS = [5, 10, 20, 40, 50, 60]
_MA_COLORS = {5: "#2980b9", 10: "#e67e22", 20: "#27ae60",
              40: "#8e44ad", 50: "#c0392b", 60: "#16a085"}
DISPLAY_DAYS = 250
CHIP_WIN = 120


def _macd(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = pd.Series(close)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = (dif - dea) * 2
    return dif.values, dea.values, hist.values


def _rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    s = pd.Series(close)
    diff = s.diff()
    up = diff.clip(lower=0)
    dn = (-diff).clip(lower=0)
    avg_up = up.ewm(alpha=1 / n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).values


def build_figure(daily: pd.DataFrame, r: dict, cfg: dict, float_shares) -> go.Figure:
    d = daily.copy()
    close = d["fwd_close"].values
    # 指标 (全长算, 显示切片)
    for w in MA_WINDOWS:
        d[f"ma{w}"] = pd.Series(close).rolling(w, min_periods=1).mean().values
    d["dif"], d["dea"], d["hist"] = _macd(close)
    d["rsi"] = _rsi(close, 14)
    wk = calc_kdj(resample_weekly(d))

    disp = d.tail(DISPLAY_DAYS).reset_index(drop=True)
    date = disp["date"]
    # 显示窗口 y 范围 (K线 + 均线 + chip band)
    pmin = float(disp["fwd_low"].min())
    pmax = float(disp["fwd_high"].max())
    pad = (pmax - pmin) * 0.05 if pmax > pmin else 1
    pmin -= pad; pmax += pad

    # 筹码峰 (近 CHIP_WIN 日, 用最新日 snapshot)
    ch = cfg["chip"]
    exp = chipmod.compute_density_exp(d, tau=ch["exp_tau"], nbins=ch["nbins"], window=CHIP_WIN)
    tur = chipmod.compute_density_turnover(d, float_shares, nbins=ch["nbins"],
                                           window=CHIP_WIN, cap=ch["turnover_cap"])
    exp_bands = chipmod.find_bands(exp[0], exp[1], ch["peak_order"], ch["peak_band_ratio"],
                                   ch["sig_peak_ratio"], ch["smooth_win"]) if exp else []

    fig = make_subplots(rows=5, cols=1, shared_xaxes=False, vertical_spacing=0.035,
                        row_heights=[0.40, 0.15, 0.15, 0.13, 0.17],
                        subplot_titles=("日K线 + 均线 (绿带=筹码密集区, ✕=doji trigger, 虚线=背离anchor)",
                                        "周 KDJ", "日 MACD", "日 RSI(14)", "筹码峰密度 (蓝=指数衰减 橙=换手率衰减, y轴=价格)"))
    # 行1: K线 + 均线
    fig.add_trace(go.Candlestick(x=date, open=disp["fwd_open"], high=disp["fwd_high"],
                                 low=disp["fwd_low"], close=disp["fwd_close"], name="K线",
                                 increasing_line_color="red", decreasing_line_color="green",
                                 increasing_line_width=1.4, decreasing_line_width=1.4,
                                 whiskerwidth=0.6), row=1, col=1)
    for w in MA_WINDOWS:
        fig.add_trace(go.Scatter(x=date, y=disp[f"ma{w}"], name=f"MA{w}",
                                 line=dict(color=_MA_COLORS[w], width=1.1)), row=1, col=1)
    # 筹码密集带 (hrect 在 row1)
    for b in exp_bands:
        fig.add_hrect(y0=b[0], y1=b[1], fillcolor="#27ae60", opacity=0.12,
                      line_width=0, row=1, col=1)
    # doji trigger + anchor
    try:
        td = pd.to_datetime(r["TrigDate"])
        fig.add_trace(go.Scatter(x=[td], y=[r["TrigClose"]], mode="markers",
                                 marker=dict(size=13, color="#e44", symbol="x-thin",
                                             line=dict(width=2, color="#e44")),
                                 name="doji"), row=1, col=1)
    except Exception:
        pass
    fig.add_hline(y=r["Anchor"], line_dash="dot", line_color="#888", row=1, col=1)

    # 行2: 周KDJ
    wk_disp = wk.tail(78)
    fig.add_trace(go.Scatter(x=wk_disp["date"], y=wk_disp["k"], name="K",
                             line=dict(color="#2980b9", width=1.3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=wk_disp["date"], y=wk_disp["d"], name="D",
                             line=dict(color="#e67e22", width=1.3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=wk_disp["date"], y=wk_disp["j"], name="J",
                             line=dict(color="#9b59b6", width=1.3)), row=2, col=1)
    fig.add_hline(y=20, line_color="#ccc", line_width=0.8, row=2, col=1)
    fig.add_hline(y=80, line_color="#ccc", line_width=0.8, row=2, col=1)

    # 行3: 日MACD
    hc = ["#e44" if v >= 0 else "#2ecc71" for v in disp["hist"]]
    fig.add_trace(go.Bar(x=date, y=disp["hist"], name="MACD柱", marker_color=hc), row=3, col=1)
    fig.add_trace(go.Scatter(x=date, y=disp["dif"], name="DIF",
                             line=dict(color="#2980b9", width=1.3)), row=3, col=1)
    fig.add_trace(go.Scatter(x=date, y=disp["dea"], name="DEA",
                             line=dict(color="#e67e22", width=1.3)), row=3, col=1)
    fig.add_hline(y=0, line_color="#bbb", line_width=0.8, row=3, col=1)

    # 行4: RSI
    fig.add_trace(go.Scatter(x=date, y=disp["rsi"], name="RSI14",
                             line=dict(color="#8e44ad", width=1.4)), row=4, col=1)
    fig.add_hline(y=30, line_color="#ccc", line_width=0.8, row=4, col=1)
    fig.add_hline(y=70, line_color="#ccc", line_width=0.8, row=4, col=1)

    # 行5: 筹码峰密度 (x=密度, y=价格, 与 row1 y 同范围)
    if exp:
        fig.add_trace(go.Scatter(x=exp[1], y=exp[0], fill="tozeroy", name="指数衰减",
                                 line=dict(color="#2980b9", width=1.1)), row=5, col=1)
    if tur:
        fig.add_trace(go.Scatter(x=tur[1], y=tur[0], fill="tozeroy", name="换手率衰减",
                                 line=dict(color="#e67e22", width=1.1)), row=5, col=1)
    # 标注 dense band 价位
    for b in exp_bands:
        fig.add_hrect(y0=b[0], y1=b[1], fillcolor="#27ae60", opacity=0.10,
                      line_width=0, row=5, col=1)

    # 行2-4 共享 row1 日期 x 轴
    for rr in (2, 3, 4):
        fig.update_xaxes(matches="x", row=rr, col=1)
    # y 范围同步: row1 与 row5 (价格)
    fig.update_yaxes(range=[pmin, pmax], autorange=False, row=1, col=1)
    fig.update_yaxes(range=[pmin, pmax], autorange=False, row=5, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_xaxes(showspikes=False)
    fig.update_layout(height=920, margin=dict(l=50, r=20, t=50, b=30),
                      showlegend=False, plot_bgcolor="#fff",
                      hovermode="x unified" if False else "closest")
    return fig


# ---- crosshair JS (SVG overlay, 贯穿日期轴子图 xaxis..xaxis4) ----
CROSSHAIR_JS = """
function attachCrosshair(gd){
  if (!gd || gd._xhair) return;
  gd._xhair = true;
  var ns = 'http://www.w3.org/2000/svg';
  var grp = null, svg = null;
  function setup(){
    svg = gd.querySelector('.main-svg');
    if (!svg) return false;
    if (!grp){
      grp = document.createElementNS(ns,'g'); grp.id='crosshair-grp';
      grp.style.pointerEvents='none'; svg.appendChild(grp);
    }
    return true;
  }
  function rects(){
    var fl = gd._fullLayout, out = [];
    // 只取日期轴子图 (前4行): xaxis..xaxis4
    var xs = ['xaxis','xaxis2','xaxis3','xaxis4'];
    var ys = ['yaxis','yaxis2','yaxis3','yaxis4'];
    xs.forEach(function(xn,i){
      var ax = fl[xn], yx = fl[ys[i]];
      if (ax && yx) out.push({x:ax._offset, xlen:ax._length, y:yx._offset, ylen:yx._length});
    });
    return out;
  }
  function draw(clientX){
    if (!setup()) return;
    var r = svg.getBoundingClientRect();
    var sx = clientX - r.left;
    while (grp.firstChild) grp.removeChild(grp.firstChild);
    var rs = rects(); if (!rs.length) return;
    var f = rs[0];
    if (sx < f.x || sx > f.x + f.xlen) return;
    rs.forEach(function(rc){
      var ln = document.createElementNS(ns,'line');
      ln.setAttribute('x1',sx); ln.setAttribute('x2',sx);
      ln.setAttribute('y1',rc.y); ln.setAttribute('y2',rc.y+rc.ylen);
      ln.setAttribute('stroke','rgba(60,60,60,0.55)');
      ln.setAttribute('stroke-width','1.5');
      grp.appendChild(ln);
    });
  }
  function clear(){ if (grp) while (grp.firstChild) grp.removeChild(grp.firstChild); }
  gd.addEventListener('mousemove', function(e){ draw(e.clientX); });
  gd.addEventListener('mouseleave', clear);
  if (!setup()){
    var n = 0; (function retry(){ if (setup() || ++n > 60) return; requestAnimationFrame(retry); })();
  }
}
"""


def build_page(df: pd.DataFrame, gmap: dict, cfg: dict, asof: str, float_map: dict) -> str:
    stocks = []
    for _, r in df.iterrows():
        daily = gmap.get(r["Ticker"])
        if daily is None:
            continue
        try:
            fig = build_figure(daily, r, cfg, float_map.get(r["Ticker"]))
        except Exception as e:
            print(f"  [plot err] {r['Ticker']}: {e}")
            continue
        fd = json.loads(fig.to_json())
        stocks.append({
            "ticker": r["Ticker"], "name": r["Name"], "side": r["Side"],
            "tier": r["Tier"], "score": int(r["Score"]),
            "data": fd["data"], "layout": fd["layout"],
        })

    side_color = {"底买": "#1f6fb4", "顶卖": "#a12c5b"}
    tier_color = {"高": "#1e8449", "中": "#b9770e", "低": "#888"}
    rows_html = []
    for i, s in enumerate(stocks):
        rows_html.append(
            f"<div class='row' onclick='show({i})' data-i='{i}'>"
            f"<span class='rk'>{i+1}</span>"
            f"<span class='tk'>{s['ticker']}</span>"
            f"<span class='nm'>{s['name']}</span>"
            f"<span class='sd' style='color:{side_color.get(s['side'],'#333')}'>{s['side']}</span>"
            f"<span class='tr' style='color:{tier_color.get(s['tier'],'#888')}'>{s['tier']}{s['score']}</span>"
            f"</div>")

    plotly_js = _plotly_bundle()
    html = f"""<!DOCTYPE html><html lang='zh'><head><meta charset='utf-8'>
<title>背离+十字星+筹码峰 {asof}</title>
<script>{plotly_js}</script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI','Microsoft YaHei',Arial,sans-serif;background:#f5f6fa;color:#2c3e50}}
.layout{{display:grid;grid-template-columns:300px 1fr;height:100vh}}
.sidebar{{background:#fff;border-right:1px solid #e0e0e0;overflow-y:auto;padding:8px 0}}
.sidebar h1{{font-size:14px;color:#1F4E78;padding:10px 14px 6px;position:sticky;top:0;background:#fff;z-index:2}}
.sidebar .hint{{font-size:11px;color:#999;padding:0 14px 8px;position:sticky;top:36px;background:#fff}}
.row{{display:grid;grid-template-columns:24px 70px 1fr 34px 44px;gap:4px;align-items:center;
padding:7px 14px;font-size:12px;cursor:pointer;border-bottom:1px solid #f2f2f2}}
.row:hover{{background:#eef4fb}}
.row.active{{background:#d6e8f8;font-weight:600}}
.rk{{color:#aaa;font-size:11px}}.tk{{color:#1F4E78;font-weight:600}}.nm{{color:#444;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.sd{{font-size:11px}}.tr{{font-size:11px;font-weight:700}}
.main{{display:flex;flex-direction:column;overflow:hidden}}
.topbar{{background:#fff;padding:8px 18px;border-bottom:1px solid #e0e0e0;font-size:13px;display:flex;
justify-content:space-between;align-items:center}}
#curinfo{{font-weight:600;color:#1F4E78}}
.keys{{font-size:11px;color:#999}}.keys kbd{{background:#eee;border:1px solid #ccc;border-radius:3px;padding:1px 5px;font-family:monospace}}
#chart{{flex:1;overflow:auto;padding:6px 10px}}
</style></head><body>
<div class='layout'>
  <div class='sidebar'>
    <h1>背离+十字星+筹码峰</h1>
    <div class='hint'>截至 {asof} · {len(stocks)} 只 · 点选或 ← → 翻页</div>
    {''.join(rows_html)}
  </div>
  <div class='main'>
    <div class='topbar'><span id='curinfo'></span>
      <span class='keys'><kbd>←</kbd> <kbd>→</kbd> 翻页 · 鼠标移图看 crosshair</span></div>
    <div id='chart'></div>
  </div>
</div>
<script>
var STOCKS = {json.dumps(stocks, ensure_ascii=False)};
{CROSSHAIR_JS}
var cur = 0;
function show(i){{
  cur = ((i % STOCKS.length) + STOCKS.length) % STOCKS.length;
  var s = STOCKS[cur];
  Plotly.newPlot('chart', s.data, s.layout, {{responsive:true, displayModeBar:false}})
    .then(function(gd){{ attachCrosshair(gd); }});
  document.getElementById('curinfo').textContent =
    (cur+1)+'/'+STOCKS.length+'  '+s.ticker+' '+s.name+'  '+s.side+' '+s.tier+' ('+s.score+'分)';
  document.querySelectorAll('.row').forEach(function(r,k){{
    r.classList.toggle('active', k===cur);}});
  document.querySelector('.sidebar').scrollTop = (cur-3)*36;
}}
document.addEventListener('keydown', function(e){{
  if(e.key==='ArrowRight'||e.key==='ArrowDown'){{e.preventDefault();show(cur+1);}}
  else if(e.key==='ArrowLeft'||e.key==='ArrowUp'){{e.preventDefault();show(cur-1);}}
}});
show(0);
</script>
</body></html>"""
    return html
