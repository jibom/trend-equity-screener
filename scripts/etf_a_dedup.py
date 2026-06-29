"""China A ETF 去重: 读 xlsx 清单 → 去债券/货币 → Wind 取跟踪基准+20日均成交额 → 按基准去重(留流动性最高) → 输出.
Wind: chinamutualfunddescription.F_INFO_BENCHMARK (跟踪指数), chinaclosedfundeodprice.S_DQ_AMOUNT (成交额,千元).
无基准的(Wind 描述表缺新ETF)按名称主题去重(剥发行人+ETF/LOF+中证/上证前缀).
用法: python scripts/etf_a_dedup.py
"""
from __future__ import annotations
import os, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pymysql
import pandas as pd
from db_config import DB_CONFIG

XLSX = r"D:\Users\Jibo Ma\Downloads\China A ETF.xlsx"
OUT = os.path.join(os.path.dirname(__file__), "..", "output", "china_a_etf_dedup.csv"
)
EXCLUDE_CAT = {"债券型ETF", "货币型ETF", "商品型ETF"}
# 港股通行业ETF (保留): 恒生科技/医药/医疗/互联网/消费, 港股通创新药/互联网/科技/信息技术/金融/非银/消费, 香港证券
KEEP_HK = ['恒生科技', '恒生医药', '恒生医疗', '恒生互联网', '恒生消费', '恒生创新药',
           '港股通创新药', '港股通互联网', '港股通科技', '港股通信息技术', '港股通金融',
           '港股通非银', '港股通消费', '港股通医疗', '香港证券', '港股创新药', '港股互联网', '港股科技']
# 非A市场 (美股/日韩/欧洲/亚太/海外/中概) — 名称或基准含即剔除 (港股通行业ETF 豁免)
NON_A_KW = ['港股', '恒生', '中概', '纳指', '标普', '中韩', '日经', '德国', '法国', '亚太',
            '海外', '纳斯达克', '道琼', '美国', '日本', '韩国', '印度', '越南', '沙特', '英国',
            '互认', 'QDII', 'MSCI', '富时', 'FTSE', 'Dow', 'S&P', 'Nasdaq']
# 商品现货/期货 (基准含即剔除, 如 黄金现货/原油期货); A股股票类(黄金股/有色金属)保留
COMMODITY_KW = ['现货', '期货', '原油', 'Au99', 'Au9999', '商品指数', '黄金现货', '银期货']


def is_hk_sector(name):
    return any(k in str(name) for k in KEEP_HK)


# 港股通行业 → canonical 主题 (归并 恒生科技/恒生科技指数收益率 等; 港股通创新药/港股创新药/恒生创新药 等)
HK_THEME_MAP = [('恒生科技', 'HK恒生科技'), ('创新药', 'HK创新药'), ('互联网', 'HK互联网'),
                ('医药', 'HK医药医疗'), ('医疗', 'HK医药医疗'), ('证券', 'HK证券'),
                ('非银', 'HK非银'), ('信息技术', 'HK科技'), ('科技', 'HK科技'),
                ('金融', 'HK金融'), ('消费', 'HK消费')]


def hk_theme(name):
    n = str(name)
    for kw, t in HK_THEME_MAP:
        if kw in n:
            return t
    return n


def drop_non_a(name, benchmark):
    """是否剔除: 非A市场 且 不是港股通行业ETF; 或 商品现货/期货"""
    b = benchmark if benchmark != "—" else ""
    s = f"{name} {b}"
    non_a = any(k in s for k in NON_A_KW)
    hk_keep = is_hk_sector(name)
    if non_a and not hk_keep:
        return True
    if b and any(k in b for k in COMMODITY_KW):
        return True
    return False


def build_issuers(names):
    """从名称里抽发行人 (ETF/LOF 之后的尾部), 数据驱动."""
    out = set()
    for n in names:
        m = re.search(r'(ETF|LOF)(.*)$', str(n))
        if m and m.group(2).strip():
            out.add(m.group(2).strip())
    return sorted(out, key=len, reverse=True)


def theme(name, issuers):
    n = str(name)
    n = n.replace('ETF', '').replace('LOF', '')
    for iss in issuers:
        if iss in n:
            n = n.replace(iss, '')
    for pre in ('中证', '上证', '国证', '深证'):
        if n.startswith(pre):
            n = n[len(pre):]
    return n.strip()


def main():
    df = pd.read_excel(XLSX)
    df = df.rename(columns={"代码": "Ticker", "简称": "Name", "类别": "Category",
                            "成交额(亿元)": "AmtXlsx"})
    print(f"[xlsx] {len(df)} 只; 去债券/货币前类别:\n{df['Category'].value_counts().to_string()}")
    df = df[~df["Category"].isin(EXCLUDE_CAT)].copy()
    print(f"[去债券/货币后] {len(df)} 只")

    codes = df["Ticker"].tolist()
    conn = pymysql.connect(**DB_CONFIG); cur = conn.cursor()
    in_sql = ",".join(f"'{c}'" for c in codes)

    cur.execute(f"""SELECT F_INFO_WINDCODE, F_INFO_BENCHMARK FROM chinamutualfunddescription
                    WHERE F_INFO_WINDCODE IN ({in_sql})""")
    bench = {r[0]: r[1] for r in cur.fetchall()}
    print(f"[Wind] 基准命中 {len(bench)}/{len(codes)}")

    cur.execute(f"""SELECT S_INFO_WINDCODE, AVG(S_DQ_AMOUNT) AS amt
                    FROM chinaclosedfundeodprice
                    WHERE S_INFO_WINDCODE IN ({in_sql})
                      AND TRADE_DT >= DATE_SUB((SELECT MAX(TRADE_DT) FROM chinaclosedfundeodprice WHERE S_INFO_WINDCODE IN ({in_sql})), INTERVAL 45 DAY)
                    GROUP BY S_INFO_WINDCODE""")
    liq = {r[0]: float(r[1]) for r in cur.fetchall()}
    conn.close()
    print(f"[Wind] 成交额命中 {len(liq)}/{len(codes)}")

    df["Benchmark"] = df["Ticker"].map(lambda t: bench.get(t) or "—")
    df["Liquidity(亿元)"] = df["Ticker"].map(lambda t: round(liq.get(t, 0) / 1e5, 2))

    # 剔除非A市场(美股/日韩/海外/中概) + 商品现货/期货; 保留A股 + 港股通行业ETF
    before = len(df)
    df = df[~df.apply(lambda r: drop_non_a(r["Name"], r["Benchmark"]), axis=1)].copy()
    print(f"[去非A/商品后] {len(df)} 只 (剔除 {before - len(df)}; 保留港股通行业ETF)")

    # 去重 key: 有基准用基准; 无基准用名称主题
    issuers = build_issuers(df["Name"])
    # 去重 key: 港股通行业用 canonical 主题; A股一律用名称主题(剥发行人+ETF+中证/上证前缀)
    # 这样同主题不同发行人(如 半导体设备ETF易方达/国泰)一定合并, 不受 Wind 基准缺失影响
    df["DedupKey"] = df.apply(lambda r: hk_theme(r["Name"]) if is_hk_sector(r["Name"])
                              else theme(r["Name"], issuers), axis=1)
    out = df.sort_values("Liquidity(亿元)", ascending=False).drop_duplicates("DedupKey", keep="first")
    out = out.sort_values("Liquidity(亿元)", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out[["Ticker", "Name", "Benchmark", "Liquidity(亿元)", "Category", "DedupKey"]].to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n[Done] 去重后 {len(out)} 只 (去重前 {len(df)}) → {OUT}")
    print(out[["Ticker", "Name", "Benchmark", "Liquidity(亿元)"]].head(35).to_string(index=False))


if __name__ == "__main__":
    main()
