"""A股 swing 池: 全A 按近 20 日成交额(S_DQ_AMOUNT) SUM 排序取 top600, 剔退市/ST。

数据源: jianxin Wind MySQL
  asharedescription  — 全A 名单 + 名称 (S_INFO_NAME) + 退市日 (S_INFO_DELISTDATE)
  ashareeodprices    — 日K 成交额 (S_DQ_AMOUNT), 仅取 S_DQ_TRADESTATUS='交易'

缓存 a_pool.csv (含 _asof 标记): asof 变化即重建 (单条 GROUP BY 查询, 秒级)。
与 us_pool 接口对齐: load_a_pool(cache, asof, rebuild) -> [(code, name, sector="")]。
sector 留空 (与 HK 池一致; 申万行业名表无易取映射, 且 A股 RS 不含行业基准)。"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import pymysql
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

TOP_N = 600
TURNOVER_DAYS = 20   # 近 20 个交易日成交额


def _conn():
    return pymysql.connect(
        host=os.environ["DB_HOST"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"], database=os.environ.get("DB_NAME") or "jianxin",
        port=int(os.environ.get("DB_PORT") or "3306"), charset="utf8mb4",
    )


def _decode_name(s) -> str:
    """DB 名称列存 GBK 字节却声明为 utf8mb4 → utf8mb4 连接拿到 mojibake 串
    (每个字符码点 == 原始 GBK 字节 0x00-0xFF)。encode('latin1') 还原字节, 再 gbk 解码。"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return ""
    try:
        return str(s).encode("latin1").decode("gbk")
    except Exception:
        return str(s)


def build_a_pool(asof: str, top: int = TOP_N) -> list[tuple[str, str, str]]:
    """返回 [(code, name, "")] 按近20日成交额降序 top 只, 剔退市/ST。"""
    # 成交额排名窗口: asof 往前推 ~45 日历日 (覆盖 20 个交易日 + 缓冲)
    end = asof.replace("-", "")
    start = (datetime.strptime(asof, "%Y-%m-%d") - timedelta(days=45)).strftime("%Y%m%d")
    with _conn() as c:
        # 全A 在册名单 + 名称 (.SH/.SZ, 未退市); 名称列存 GBK, utf8mb4 连接得 mojibake, latin1+gbk 还原
        desc = pd.read_sql(
            "SELECT S_INFO_WINDCODE code, S_INFO_NAME name FROM asharedescription "
            "WHERE (S_INFO_DELISTDATE IS NULL OR S_INFO_DELISTDATE='' OR S_INFO_DELISTDATE > %s)",
            c, params=(end,))
        desc = desc[desc["code"].str.endswith((".SH", ".SZ"))].copy()
        desc["name"] = desc["name"].map(_decode_name)
        # 剔 ST/*ST (名称含)
        desc = desc[~desc["name"].str.contains("ST", na=False)]
        # 近20日成交额 SUM
        amt = pd.read_sql(
            "SELECT S_INFO_WINDCODE code, SUM(S_DQ_AMOUNT) amt FROM ashareeodprices "
            "WHERE TRADE_DT BETWEEN %s AND %s AND S_DQ_TRADESTATUS='交易' "
            "GROUP BY S_INFO_WINDCODE", c, params=(start, end))
    m = amt.merge(desc, on="code", how="inner").sort_values("amt", ascending=False)
    m = m.head(top)
    return [(r.code, r.name, "") for r in m.itertuples(index=False)]


def load_a_pool(cache: Path, asof: str, rebuild: bool = False, top: int = TOP_N) -> list[tuple[str, str, str]]:
    """带缓存加载。缓存首行 `# _asof=...` 标记; asof 不符或 --rebuild 时重建。"""
    cache = Path(cache)
    if not rebuild and cache.exists():
        try:
            with open(cache, encoding="utf-8-sig") as f:
                first = f.readline().strip()
            cached_asof = ""
            if first.startswith("# _asof="):
                cached_asof = first.split("=", 1)[1]
            if cached_asof == asof:
                df = pd.read_csv(cache, comment="#").fillna("")
                return [(str(r.code), str(r.name_cn), str(r.gics_sector)) for r in df.itertuples(index=False)]
        except Exception:
            pass
    print(f"[A-Pool] 重建 top{top} 成交额池 (asof={asof}) ...")
    pool = build_a_pool(asof, top=top)
    df = pd.DataFrame(pool, columns=["code", "name_cn", "gics_sector"])
    try:
        with open(cache, "w", encoding="utf-8-sig") as f:
            f.write(f"# _asof={asof}\n")
            df.to_csv(f, index=False)
    except Exception as e:
        print(f"[A-Pool] 缓存写入失败: {e}")
    print(f"[A-Pool] {len(pool)} 只 -> {cache.name}")
    return pool


if __name__ == "__main__":
    import sys
    asof = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    p = build_a_pool(asof)
    print(f"{len(p)} 只; 前5:")
    for c, n, _ in p[:5]:
        print(f"  {c}  {n}")
