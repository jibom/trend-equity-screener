"""合并抄底+逃顶为单文件 2-tab HTML (抄底 / 逃顶)。

复用 plot_doji_cumulative.build() 与 plot_top_cumulative.build() 返回的 (fig, table_html)。
plotly.js 只嵌一次 (pyo.get_plotlyjs), 每个 fig 用 Plotly.newPlot + 手动注入 crosshair JS。
tab 切换时调 Plotly.Plots.resize 修隐藏 div 尺寸。
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import plotly.io as pio
import plotly.offline as pyo
import plot_doji_cumulative
import plot_top_cumulative

OUT = os.path.join(os.path.dirname(__file__), '..', 'output', 'hsi_timing_combined.html')


def crosshair_js(div_id: str) -> str:
    """跨子图十字光标: 列出 xaxis..xaxis10, 不存在的轴自动跳过 (8子图/10子图通用)。"""
    return f"""
    (function(){{
      var gd = document.getElementById('{div_id}');
      if (!gd) return;
      var ns = 'http://www.w3.org/2000/svg';
      var grp = null, svg = null;
      function setup(){{
        svg = gd.querySelector('.main-svg');
        if (!svg) return false;
        if (grp) return true;
        grp = document.createElementNS(ns, 'g');
        grp.id = 'crosshair-grp-{div_id}';
        grp.style.pointerEvents = 'none';
        svg.appendChild(grp);
        return true;
      }}
      function getSubplotRects(){{
        var fl = gd._fullLayout; var rects = [];
        var xN = ['xaxis','xaxis2','xaxis3','xaxis4','xaxis5','xaxis6','xaxis7','xaxis8','xaxis9','xaxis10'];
        var yN = ['yaxis','yaxis2','yaxis3','yaxis4','yaxis5','yaxis6','yaxis7','yaxis8','yaxis9','yaxis10'];
        xN.forEach(function(axName, i){{
          var ax = fl[axName], yAx = fl[yN[i]];
          if (!ax || !yAx) return;
          rects.push({{ x: ax._offset, xlen: ax._length, y: yAx._offset, ylen: yAx._length }});
        }});
        return rects;
      }}
      function draw(clientX){{
        if (!setup()) return;
        var r = svg.getBoundingClientRect();
        var sx = clientX - r.left;
        while (grp.firstChild) grp.removeChild(grp.firstChild);
        var rects = getSubplotRects(); if (!rects.length) return;
        var f = rects[0];
        if (sx < f.x || sx > f.x + f.xlen) return;
        rects.forEach(function(rc){{
          var ln = document.createElementNS(ns, 'line');
          ln.setAttribute('x1', sx); ln.setAttribute('x2', sx);
          ln.setAttribute('y1', rc.y); ln.setAttribute('y2', rc.y + rc.ylen);
          ln.setAttribute('stroke', 'rgba(60,60,60,0.55)');
          ln.setAttribute('stroke-width', '1.5');
          grp.appendChild(ln);
        }});
      }}
      function clear(){{ if (grp) while (grp.firstChild) grp.removeChild(grp.firstChild); }}
      gd.addEventListener('mousemove', function(e){{ draw(e.clientX); }});
      gd.addEventListener('mouseleave', clear);
      if (!setup()){{ var a=0; (function retry(){{ if (setup()||++a>60) return; requestAnimationFrame(retry); }})(); }}
    }})();
    """


def main():
    print("[Build] 抄底 fig ...")
    fig_b, tbl_b = plot_doji_cumulative.build()
    print("[Build] 逃顶 fig ...")
    fig_t, tbl_t = plot_top_cumulative.build()

    plotly_js = pyo.get_plotlyjs()
    tabs = [('抄底 (底部择时)', 'bottom', fig_b, tbl_b),
            ('逃顶 (顶部择时)', 'top', fig_t, tbl_t)]

    buttons = ''.join(
        f'<button class="tabbtn" onclick="switchTab(\'{tid}\')">{name}</button>'
        for name, tid, _, _ in tabs)

    contents = []
    for i, (name, tid, fig, tbl) in enumerate(tabs):
        div_id = f'fig-{tid}'
        fig_json = pio.to_json(fig)
        display = 'block' if i == 0 else 'none'
        block = (
            f'<div id="tab-{tid}" class="tabcontent" style="display:{display}">'
            f'{tbl}'
            f'<div id="{div_id}" class="plotly-graph-div" style="margin:6px auto;"></div>'
            f'<script>Plotly.newPlot("{div_id}", {fig_json});</script>'
            f'<script>{crosshair_js(div_id)}</script>'
            f'</div>'
        )
        contents.append(block)

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>HSI 抄底+逃顶 择时</title>
<script>{plotly_js}</script>
<style>
body {{ font-family: sans-serif; margin: 0; background: #fafafa; }}
.tabbar {{ position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ddd; display: flex; z-index: 10; padding: 0 8px; }}
.tabbtn {{ border: none; background: none; padding: 12px 20px; cursor: pointer; font-size: 14px; color: #555; border-bottom: 3px solid transparent; }}
.tabbtn:hover {{ color: #000; background: #f0f0f0; }}
.tabbtn.active {{ color: #1565c0; border-bottom-color: #1565c0; font-weight: 600; }}
.tabcontent {{ padding: 4px 8px 24px; }}
</style></head><body>
<div class="tabbar">{buttons}</div>
{''.join(contents)}
<script>
function switchTab(id){{
  document.querySelectorAll('.tabcontent').forEach(function(e){{ e.style.display='none'; }});
  document.querySelectorAll('.tabbtn').forEach(function(b){{ b.classList.remove('active'); }});
  document.getElementById('tab-'+id).style.display='block';
  var btns=document.querySelectorAll('.tabbtn');
  for (var i=0;i<btns.length;i++){{ if(btns[i].getAttribute('onclick').indexOf("'"+id+"'")>-1){{ btns[i].classList.add('active'); break; }} }}
  setTimeout(function(){{
    document.getElementById('tab-'+id).querySelectorAll('.plotly-graph-div').forEach(function(gd){{
      if(window.Plotly && Plotly.Plots) Plotly.Plots.resize(gd);
    }});
  }}, 50);
}}
document.querySelector('.tabbtn').classList.add('active');
</script>
</body></html>"""

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[Done] -> {OUT}  (2 tabs: 抄底 + 逃顶)")


if __name__ == '__main__':
    main()
