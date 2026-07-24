"""生成两 tab 网站页 index.html: HK Swing / US Swing。
每个 tab: 顶部下载按钮 + DataTables 可搜索/排序表格。读 HK/US xlsx 的 swing sheet。"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd


def _read(xlsx: Path):
    """返回 (table_html, asof, n) 或 (None, '', 0)。"""
    if not Path(xlsx).exists():
        return None, "", 0
    df = pd.read_excel(xlsx, sheet_name="swing").fillna("")
    asof = ""
    try:
        info = pd.read_excel(xlsx, sheet_name="指标说明", header=None)
        m = re.search(r"截至[:：]\s*(\S+)", str(info.iloc[1, 0]))
        if m:
            asof = m.group(1)
    except Exception:
        pass
    table = df.to_html(index=False, table_id=None, classes="display swing-table",
                       border=0, escape=False)
    return table, asof, len(df)


def _panel(tab_id: str, label: str, xlsx_name: str, table_html, asof, n):
    meta = f"截至 {asof} · {n} 只" if asof else f"{n} 只"
    body = table_html if table_html else '<p style="padding:20px;color:#888">暂无数据</p>'
    return f"""
<div id="{tab_id}" class="panel">
  <div class="panel-head">
    <span class="meta">{label} — {meta}</span>
    <a class="btn" href="{xlsx_name}" download>⬇ 下载 {xlsx_name.replace('.xlsx','')}</a>
  </div>
  <div class="wrap">{body}</div>
</div>"""


def build(hk_xlsx: Path, us_xlsx: Path, out_html: Path):
    hk_tbl, hk_asof, hk_n = _read(hk_xlsx)
    us_tbl, us_asof, us_n = _read(us_xlsx)
    hk_panel = _panel("panel-hk", "港股 Swing Pattern", "HK_Swing_Pattern.xlsx", hk_tbl, hk_asof, hk_n)
    us_panel = _panel("panel-us", "美股 Swing Pattern", "US_Swing_Pattern.xlsx", us_tbl, us_asof, us_n)
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swing Pattern (HK / US)</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI','Microsoft YaHei',Arial,sans-serif;background:#f5f6fa;color:#2c3e50}}
.header{{background:linear-gradient(135deg,#1F4E78,#2980b9);color:#fff;padding:14px 24px}}
.header h1{{font-size:19px;font-weight:600}}
.tabs{{display:flex;background:#fff;border-bottom:2px solid #1F4E78}}
.tab{{padding:12px 28px;cursor:pointer;font-weight:600;font-size:14px;color:#5a6470;border-bottom:3px solid transparent}}
.tab.active{{color:#1F4E78;border-bottom-color:#1F4E78;background:#eef4fb}}
.panel{{display:none}}
.panel.active{{display:block}}
.panel-head{{display:flex;align-items:center;justify-content:space-between;padding:12px 24px;flex-wrap:wrap;gap:8px}}
.panel-head .meta{{font-size:13px;color:#5a6470}}
.btn{{display:inline-block;background:#1F4E78;color:#fff;padding:9px 18px;border-radius:6px;
text-decoration:none;font-weight:700;font-size:13px}}
.btn:hover{{background:#2980b9}}
.wrap{{padding:6px 20px 20px}}
table.display{{background:#fff;border-collapse:collapse;width:100%!important;font-size:13px}}
table.display thead th{{background:#1F4E78;color:#fff;padding:8px 10px;text-align:left;white-space:nowrap}}
table.display tbody td{{padding:6px 10px;border-bottom:1px solid #f0f0f0;white-space:nowrap}}
table.display tbody tr:hover{{background:#eef4fb}}
.dataTables_filter input,.dataTables_length select{{padding:4px;border:1px solid #ccc;border-radius:3px}}
</style></head><body>
<div class="header"><h1>Swing Pattern — HK / US</h1></div>
<div class="tabs">
  <div class="tab active" onclick="showTab('hk')">港股 HK</div>
  <div class="tab" onclick="showTab('us')">美股 US</div>
</div>
{hk_panel}{us_panel}
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script>
function showTab(which){{
  document.querySelectorAll('.tab').forEach((t,i)=>t.classList.toggle('active',i===(which==='hk'?0:1)));
  document.getElementById('panel-hk').classList.toggle('active',which==='hk');
  document.getElementById('panel-us').classList.toggle('active',which==='us');
}}
$(function(){{
  $('.swing-table').each(function(){{
    $(this).DataTable({{
      pageLength:50, lengthMenu:[25,50,100,200,500], scrollX:true, order:[],
      language:{{search:"搜索:",lengthMenu:"每页 _MENU_",info:"_START_-_END_ / _TOTAL_",
        infoEmpty:"无",paginate:{{first:"首",last:"末",next:"下一页",previous:"上一页"}}}}
    }});
  }});
}});
</script>
</body></html>"""
    out_html.write_text(html, encoding="utf-8")
