"""生成多 tab 网站页 index.html: 港股 / 美股 / A股 / ETF Swing Pattern。
每个 tab: 顶部下载按钮 + DataTables 可搜索/排序表格。读各 xlsx 的 swing sheet。

build(panels, out_html): panels = [(tab_id, label, xlsx_name, xlsx_path), ...]
default_panels(ROOT): 返回 4 个标准面板 (HK/US/A股/ETF), 各 run_swing*.py 共用。"""
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


def default_panels(root: Path) -> list[tuple[str, str, str, Path]]:
    """4 个标准面板: (tab_id, label, xlsx_name, xlsx_path)。xlsx 复制到根目录供下载。"""
    r = Path(root)
    return [
        ("panel-hk",  "港股 Swing Pattern",  "HK_Swing_Pattern.xlsx",  r / "HK_Swing_Pattern.xlsx"),
        ("panel-us",  "美股 Swing Pattern",  "US_Swing_Pattern.xlsx",  r / "US_Swing_Pattern.xlsx"),
        ("panel-a",   "A股 Swing Pattern",   "A_Swing_Pattern.xlsx",   r / "A_Swing_Pattern.xlsx"),
        ("panel-etf", "ETF Swing Pattern",   "ETF_Swing_Pattern.xlsx", r / "ETF_Swing_Pattern.xlsx"),
    ]


def build(panels: list[tuple[str, str, str, Path]], out_html: Path):
    """panels = [(tab_id, label, xlsx_name, xlsx_path), ...] → 渲染 N tab 首页。"""
    specs = []
    for tab_id, label, xlsx_name, xlsx_path in panels:
        tbl, asof, n = _read(xlsx_path)
        specs.append((tab_id, label, xlsx_name, tbl, asof, n))
    panel_html = "\n".join(
        _panel(tab_id, label, xlsx_name, tbl, asof, n)
        for tab_id, label, xlsx_name, tbl, asof, n in specs
    )
    # tab 按钮 (第一个 active)
    tab_btns = "\n".join(
        f'  <div class="tab{" active" if i == 0 else ""}" onclick="showTab(\'{tab_id}\')">{label.split(" ")[0]}</div>'
        for i, (tab_id, label, _, _, _, _) in enumerate(specs)
    )
    tab_ids = [s[0] for s in specs]
    ids_js = ", ".join(f"'{t}'" for t in tab_ids)
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Swing Pattern (HK / US / A / ETF)</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI','Microsoft YaHei',Arial,sans-serif;background:#f5f6fa;color:#2c3e50}}
.header{{background:linear-gradient(135deg,#1F4E78,#2980b9);color:#fff;padding:14px 24px}}
.header h1{{font-size:19px;font-weight:600}}
.tabs{{display:flex;background:#fff;border-bottom:2px solid #1F4E78;flex-wrap:wrap}}
.tab{{padding:12px 24px;cursor:pointer;font-weight:600;font-size:14px;color:#5a6470;border-bottom:3px solid transparent}}
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
<div class="header"><h1>Swing Pattern — HK / US / A股 / ETF</h1></div>
<div class="tabs">
{tab_btns}
</div>
{panel_html}
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script>
var IDS = [{ids_js}];
function showTab(id){{
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active', t.getAttribute('onclick').includes("'"+id+"'")));
  IDS.forEach(x=>document.getElementById(x).classList.toggle('active', x===id));
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
