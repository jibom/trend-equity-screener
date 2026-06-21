"""构建A股趋势分析基准名单 (top 600 by 成交金额, 申万一级+三级分类).

数据源: jianxin MySQL (Wind 镜像, 同港股库).
  - ashareeodprices: A股EOD (Wind schema, S_DQ_AMOUNT 单位=千元, 同港股)
  - ashareswindustriesclass (CUR_SIGN=1): 股票→SW_IND_CODE (申万, 前4位=L1, 前8位=L3)
  - ashareindustriescode: SW code→名称
  - asharedescription: 股票简称 S_INFO_NAME

排名: 近7交易日日均成交金额(千元)降序取前600 (成交金额作流动性代理, 与港股/美股口径一致).
输出: configs/a_universe.csv (Ticker, Code, Name, Sector=申万一级, SubIndustry=申万三级, ...)

用法: python scripts/build_a_universe.py [--top 600]
"""
from __future__ import annotations
import os, sys, argparse
import pandas as pd
import pymysql

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))
from db_config import DB_CONFIG  # noqa: E402

OUT_CSV = os.path.join(PROJECT_DIR, "configs", "a_universe.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=600)
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    print(f"=== build A-share universe (top {args.top} by 成交金额, {args.days}日均值) ===")

    conn = pymysql.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        # 最新交易日 + 近N个交易日
        cur.execute("SELECT MAX(TRADE_DT) FROM ashareeodprices")
        latest = str(cur.fetchone()[0])
        cur.execute("SELECT DISTINCT TRADE_DT FROM ashareeodprices ORDER BY TRADE_DT DESC LIMIT %s", (args.days,))
        dts = [str(r[0]) for r in cur.fetchall()]
        start = min(dts)
        print(f"[DB] latest={latest}, 近{len(dts)}交易日 {start}~{latest}")

        # top600 by 日均成交金额 (千元), 排除停牌(volume=0)
        sql = (f"SELECT S_INFO_WINDCODE, AVG(S_DQ_AMOUNT) amt "
               f"FROM ashareeodprices WHERE TRADE_DT BETWEEN %s AND %s "
               f"AND S_DQ_VOLUME>0 GROUP BY S_INFO_WINDCODE ORDER BY amt DESC LIMIT %s")
        cur.execute(sql, (start, latest, args.top))
        top = pd.DataFrame(cur.fetchall(), columns=["Ticker", "avg_amt_qian"])
        top["avg_amt_qian"] = top["avg_amt_qian"].astype(float)
        print(f"[Rank] top {len(top)} by 成交金额; #1 {top.iloc[0]['Ticker']} "
              f"{top.iloc[0]['avg_amt_qian']/1e5:.1f}亿, #{len(top)} {top.iloc[-1]['Ticker']} "
              f"{top.iloc[-1]['avg_amt_qian']/1e5:.2f}亿")

        codes = top["Ticker"].tolist()
        ph = ",".join(["%s"] * len(codes))

        # 名称
        cur.execute(f"SELECT S_INFO_WINDCODE, S_INFO_NAME FROM asharedescription WHERE S_INFO_WINDCODE IN ({ph})", codes)
        name_map = dict(cur.fetchall())

        # 申万分类 (2021版新表, 覆盖科创板/新股; CUR_SIGN=1 现行)
        cur.execute(f"SELECT S_INFO_WINDCODE, SW_IND_CODE FROM ashareswnindustriesclass "
                    f"WHERE CUR_SIGN='1' AND S_INFO_WINDCODE IN ({ph})", codes)
        sw_map = dict(cur.fetchall())

        # SW code→名称 字典 (所需的 L1/L3)
        l1_codes, l3_codes = set(), set()
        for sw in sw_map.values():
            sw = str(sw)
            l1_codes.add(sw[:4]); l3_codes.add(sw[:8])
        need = [c.ljust(16, "0") for c in (l1_codes | l3_codes)]
        if need:
            nph = ",".join(["%s"] * len(need))
            cur.execute(f"SELECT INDUSTRIESCODE, INDUSTRIESNAME FROM ashareindustriescode WHERE INDUSTRIESCODE IN ({nph})", need)
            code_name = {r[0].strip(): r[1] for r in cur.fetchall()}
        else:
            code_name = {}

        def lname(code4):
            nm = code_name.get(code4.ljust(16, "0"), "未分类")
            return nm.rstrip("ⅠⅡⅢⅣ") if nm != "未分类" else nm   # 去掉三级名末尾的 Ⅲ 等
    finally:
        conn.close()

    rows = []
    for _, r in top.iterrows():
        tk = r["Ticker"]
        sw = str(sw_map.get(tk, ""))
        l1, l3 = (sw[:4], sw[:8]) if sw else ("", "")
        rows.append({"Ticker": tk, "Code": tk.split(".")[0], "Name": name_map.get(tk, ""),
                     "Sector": lname(l1), "SubIndustry": lname(l3),
                     "SWL1": l1, "SWL3": l3,
                     "AvgAmtQian": round(float(r["avg_amt_qian"]), 0), "LatestDate": latest})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[CSV] -> {OUT_CSV} ({len(df)} rows)")
    print(f"[GICS-SW] 申万一级={df['Sector'].nunique()} 申万三级={df['SubIndustry'].nunique()} "
          f"未分类={int((df['Sector']=='未分类').sum())}")
    print("\n[申万一级分布]")
    print(df.groupby("Sector").size().sort_values(ascending=False).head(15).to_string())


if __name__ == "__main__":
    main()
