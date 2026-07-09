"""对比 HSI 不同均线组合死叉/金叉的预测力 (forward 5/20/60 日收益与胜率)。"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np, pymysql
from db_config import DB_CONFIG

conn = pymysql.connect(**DB_CONFIG)
hsi = pd.read_sql("SELECT TRADE_DT, S_DQ_CLOSE FROM hkindexeodprices WHERE S_INFO_WINDCODE='HSI.HI' AND TRADE_DT BETWEEN '20140101' AND '20260707' ORDER BY TRADE_DT", conn)
conn.close()
hsi['date'] = pd.to_datetime(hsi['TRADE_DT'], format='%Y%m%d')
hsi = hsi.sort_values('date').reset_index(drop=True)
hsi['close'] = hsi['S_DQ_CLOSE'].astype(float)

# 各周期 MA
for w in (5, 10, 20, 60):
    hsi[f'ma{w}'] = hsi['close'].rolling(w, min_periods=w).mean()

# 前瞻收益
for N in (5, 20, 60):
    hsi[f'fwd_{N}'] = hsi['close'].shift(-N) / hsi['close'] - 1

combos = [(5, 10), (5, 20), (10, 20), (20, 60)]

print(f"{'combo':>8} {'dir':>6} {'n':>4} {'wr5':>5} {'wr20':>5} {'wr60':>5} {'er5':>7} {'er20':>7} {'er60':>7}")
print("-" * 70)
for s, l in combos:
    short = hsi[f'ma{s}']; long = hsi[f'ma{l}']
    prev_short = short.shift(1); prev_long = long.shift(1)
    death = (prev_short >= prev_long) & (short < long)       # 短下穿长
    golden = (prev_short <= prev_long) & (short > long)      # 短上穿长
    for name, mask, up in [('死叉', death, False), ('金叉', golden, True)]:
        sub = hsi[mask]
        n = len(sub)
        if n == 0:
            print(f"{f'{s}/{l}':>8} {name:>6} {n:>4} (no cross)")
            continue
        # 死叉看下跌概率 (fwd<0), 金叉看上涨概率 (fwd>0)
        w5 = (sub['fwd_5'].dropna() < 0).mean() * 100 if not up else (sub['fwd_5'].dropna() > 0).mean() * 100
        w20 = (sub['fwd_20'].dropna() < 0).mean() * 100 if not up else (sub['fwd_20'].dropna() > 0).mean() * 100
        w60 = (sub['fwd_60'].dropna() < 0).mean() * 100 if not up else (sub['fwd_60'].dropna() > 0).mean() * 100
        e5 = sub['fwd_5'].dropna().mean() * 100
        e20 = sub['fwd_20'].dropna().mean() * 100
        e60 = sub['fwd_60'].dropna().mean() * 100
        print(f"{f'{s}/{l}':>8} {name:>6} {n:>4} {w5:>5.0f} {w20:>5.0f} {w60:>5.0f} {e5:>+7.1f} {e20:>+7.1f} {e60:>+7.1f}")

# 基准
b60_up = (hsi['fwd_60'].dropna() > 0).mean() * 100
b60_dn = (hsi['fwd_60'].dropna() < 0).mean() * 100
print("-" * 70)
print(f"基准: 60日上涨概率={b60_up:.0f}%, 下跌概率={b60_dn:.0f}%, 60日预期={hsi['fwd_60'].dropna().mean()*100:+.1f}%")
