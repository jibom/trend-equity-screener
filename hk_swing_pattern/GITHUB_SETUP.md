# GitHub 自动化部署指引

每日北京时间 20:00 自动跑 `run_swing.py` → 生成 `HK_Swing_Pattern.xlsx` + `index.html` → 提交回仓库 → GitHub Pages 展示，顶部按钮下载。

## 1. 本地验证（先确认自包含能跑）

```bash
cd D:\Project\hk_div_doji_chip
python run_swing.py
```
应生成 `HK_Swing_Pattern.xlsx`（output/ 和根目录各一份）+ 根目录 `index.html`。
用浏览器打开 `index.html` 确认表格 + 下载按钮正常。

> `.env` 已含 DB 凭据，已被 `.gitignore` 排除，**不会上传**。

## 2. 建 GitHub 仓库

1. 登录 GitHub → New repository，名字随意（如 `hk-swing-pattern`），**Public**（Pages 免费需 public）或 Private（Pages 需 Pro）。
2. 不要勾选 README/.gitignore（本地已有）。

## 3. 推送代码

```bash
cd D:\Project\hk_div_doji_chip
git init
git add .
git status   # 确认 .env 不在列表里!
git commit -m "init: swing pattern auto-update"
git branch -M main
git remote add origin https://github.com/<你的用户名>/hk-swing-pattern.git
git push -u origin main
```

**务必确认** `git status` 里没有 `.env`（凭据不能进仓库）。

## 4. 配置 Secrets（GitHub Actions 连 DB 用）

仓库 → Settings → Secrets and variables → Actions → New repository secret，逐个添加：

| Secret 名 | 值 |
|---|---|
| DB_HOST | （打开本地 `.env`，复制 DB_HOST 的值） |
| DB_PORT | 3306 |
| DB_USER | （打开本地 `.env`，复制 DB_USER 的值） |
| DB_PASSWORD | （打开本地 `.env`，复制 DB_PASSWORD 的值） |
| DB_NAME | jianxin |
| EODHD_TOKEN | （打开本地 `.env`，复制 EODHD_TOKEN 的值） |

> 真实凭据只在本地 `.env`（已 gitignore）。GitHub 用 Secrets，不进仓库。
> EODHD 为第一数据源（并行快），Wind 为回退。

## 5. 开 GitHub Pages

仓库 → Settings → Pages →
- Source: **Deploy from a branch**
- Branch: `main` / 文件夹: `/ (root)` → Save

等 1-2 分钟，访问 `https://<用户名>.github.io/hk-swing-pattern/` 即可看到表格。

## 6. 触发首次更新

仓库 → Actions → "每日更新 Swing Pattern" workflow → Run workflow（手动触发一次）。
成功后会自动 commit `HK_Swing_Pattern.xlsx` + `index.html`，Pages 随即展示。

之后每天 UTC 12:00（北京 20:00）自动跑。

## 文件清单

- `run_swing.py` — 生成表格入口
- `gen_site.py` — Excel → index.html（下载按钮 + DataTables 表格）
- `src/` — patterns/swing/climax/provider/data_provider/kdj_divergence（自包含，无外部路径依赖）
- `config.yaml` — 参数（pool 指向 output/hk_stock_list_pool.csv）
- `output/hk_stock_list_pool.csv` — 股票池（会入库）
- `requirements.txt` — Python 依赖
- `.github/workflows/update.yml` — 每日 20:00 cron + 手动触发
- `.gitignore` — 排除 .env / 缓存 / 生成文件

## 注意

- GitHub Actions cron 不保证准时，高峰可能延迟 5-30 分钟。要严格 20:00 可改本地 Task Scheduler 跑 + push。
- DB 有 T+1 延迟，20:00 跑能拿到当日或前一日数据。
- 首次 Pages 生效前，根目录需已有 `index.html`（手动触发 workflow 一次即生成）。
