"""生成网站页 index.html: 顶部下载按钮 + DataTables 可搜索/排序表格。
读取 HK_Swing_Pattern.xlsx 的 swing sheet。"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def build(xlsx_path: Path, asof: str, n: int, out_html: Path):
    df = pd.read_excel(xlsx_path, sheet_name="swing")
    df = df.fillna("")
    table_html = df.to_html(index=False, table_id="swing", classes="display",
                            border=0, escape=False)
    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>港股 Swing Pattern</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.7/css/jquery.dataTables.min.css">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI','Microsoft YaHei',Arial,sans-serif;background:#f5f6fa;color:#2c3e50}}
.header{{background:linear-gradient(135deg,#1F4E78,#2980b9);color:#fff;padding:18px 28px;
display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}}
.header h1{{font-size:20px;font-weight:600}}
.header .meta{{font-size:13px;opacity:.85}}
.btn{{display:inline-block;background:#fff;color:#1F4E78;padding:10px 20px;border-radius:6px;
text-decoration:none;font-weight:700;font-size:14px;box-shadow:0 2px 4px rgba(0,0,0,.15)}}
.btn:hover{{background:#e8f4ff}}
.wrap{{padding:14px 20px}}
table.display{{background:#fff;border-collapse:collapse;width:100%!important;font-size:13px}}
table.display thead th{{background:#1F4E78;color:#fff;padding:8px 10px;text-align:left;white-space:nowrap}}
table.display tbody td{{padding:6px 10px;border-bottom:1px solid #f0f0f0;white-space:nowrap}}
table.display tbody tr:hover{{background:#eef4fb}}
.dataTables_wrapper{{padding-top:6px}}
.dataTables_filter input,.dataTables_length select{{padding:4px;border:1px solid #ccc;border-radius:3px}}
</style></head><body>
<div class="header">
  <div><h1>港股 Swing Pattern</h1><div class="meta">截至 {asof} · {n} 只 · 每日 20:00 自动更新</div></div>
  <a class="btn" href="HK_Swing_Pattern.xlsx" download>⬇ 下载 HK_Swing_Pattern</a>
</div>
<div class="wrap">{table_html}</div>
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
<script>
$(function(){{
  $('#swing').DataTable({{
    pageLength: 50, lengthMenu: [25,50,100,200,500], scrollX: true,
    order: [], language: {{search:"搜索:", lengthMenu:"每页 _MENU_", info:"_START_-_END_ / _TOTAL_",
      infoEmpty:"无", paginate:{{first:"首",last:"末",next:"下一页",previous:"上一页"}}}}
  }});
}});
</script>
</body></html>"""
    out_html.write_text(html, encoding="utf-8")
