"""列出所有 Cap 标记日期 (score>=5, gap>10 去重), 检查 2015 是否有信号."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import pandas as pd, numpy as np, pymysql
from db_config import DB_CONFIG
from kdj_div_basic import calc_kdj

CSV = os.path.join(os.path.dirname(__file__), '..', 'output', 'top_signal_daily.csv')
df = pd.read_csv(CSV)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df = df.sort_values('date').reset_index(drop=True)

conn = pymysql.connect(**DB_CONFIG)
start = df['date'].min().strftime('%Y%m%d'); end = df['date'].max().strftime('%Y%m%d')
hsi = pd.read_sql(f"SELECT TRADE_DT, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE FROM hkindexeodprices WHERE S_INFO_WINDCODE='HSI.HI' AND TRADE_DT BETWEEN '{start}' AND '{end}' ORDER BY TRADE_DT", conn)
conn.close()
hsi['date'] = pd.to_datetime(hsi['TRADE_DT'], format='%Y%m%d')
for c in ['S_DQ_HIGH','S_DQ_LOW','S_DQ_CLOSE']: hsi[c]=hsi[c].astype(float)
close = hsi['S_DQ_CLOSE']
delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
ag=gain.ewm(alpha=1/14,adjust=False).mean(); al=loss.ewm(alpha=1/14,adjust=False).mean()
hsi['rsi']=(100-100/(1+ag/al.replace(0,np.nan))).fillna(50)
hsi['logret']=np.log(close/close.shift(1)); pr=hsi['logret'].shift(1)
hsi['cap_z']=(hsi['logret']-pr.rolling(20).mean())/pr.rolling(20).std()
df_d=pd.DataFrame({'date':hsi['date'],'high':hsi['S_DQ_HIGH'].values,'low':hsi['S_DQ_LOW'].values,'close':close.values}).set_index('date')
df_w=df_d.resample('W-FRI').agg({'high':'max','low':'min','close':'last'}).dropna()
wk=calc_kdj(pd.DataFrame({'fwd_close':df_w['close'].values,'fwd_high':df_w['high'].values,'fwd_low':df_w['low'].values}))
wj=wk['j'].values; wj_high4=pd.Series(wj).rolling(4,min_periods=4).max().values
wr=pd.DataFrame({'date':df_w.index,'w_kdj_j_high4':wj_high4}); wr['date']=pd.to_datetime(wr['date'])
wr=wr.sort_values('date').reset_index(drop=True)
df=df.sort_values('date').reset_index(drop=True)
df=pd.merge_asof(df, wr, on='date', direction='backward')
df=df.merge(hsi[['date','rsi','cap_z']],on='date',how='left')
df['cum_pct_4w']=df['active_top_pct'].rolling(20,min_periods=20).sum()
df['pctile_4w']=df['cum_pct_4w'].expanding(min_periods=60).rank(pct=True)*100
b10=df['breadth_below_ma50'].expanding(min_periods=252).quantile(0.10).shift(1)
sky99=df['sky_vol_pct'].expanding(min_periods=252).quantile(0.99).shift(1)
dist95=df['dist_top_pct'].expanding(min_periods=252).quantile(0.95).shift(1)
shrink95=df['shrink_new_high_pct'].expanding(min_periods=252).quantile(0.95).shift(1)
df['bias20']=(df['hsi_close']/df['hsi_close'].rolling(20).mean()-1)*100
bias95=df['bias20'].expanding(min_periods=252).quantile(0.95).shift(1)
LOOK=10
recent=lambda s: s.rolling(LOOK,min_periods=1).max().fillna(0).astype(int)
df['c_rsi']=(df['rsi'].rolling(10,min_periods=1).max()>70).astype(int)
df['c_kdj']=recent((df['w_kdj_j_high4']>100).fillna(False))
df['c_star']=(df['pctile_4w']>=90).astype(int)
df['c_brd']=(df['breadth_below_ma50']<=b10).fillna(False).astype(int)
df['c_cap']=(df['cap_z']>=2.5).astype(int)
df['c_sky']=(df['sky_vol_pct']>=sky99).fillna(False).astype(int)
df['c_div']=(df['vol_price_div_pct']>0).astype(int)
df['c_dist']=(df['dist_top_pct']>=dist95).fillna(False).astype(int)
df['c_shrink']=(df['shrink_new_high_pct']>=shrink95).fillna(False).astype(int)
df['c_bias']=recent((df['bias20']>=bias95).fillna(False))
CONDS=['c_rsi','c_kdj','c_star','c_brd','c_cap','c_sky','c_div','c_dist','c_shrink','c_bias']
df['score']=df[CONDS].sum(axis=1)

print("=== 2015 年所有 score>=4 的日期 ===")
s15 = df[(df['date']>='2015-01-01')&(df['date']<='2015-12-31')&(df['score']>=4)]
print(s15[['date','hsi_close','score','c_rsi','c_kdj','c_brd','c_cap','c_sky','c_div','c_shrink','c_bias']].to_string(index=False))

print("\n=== 所有 score>=5 日期 (Cap 候选) 按年统计 ===")
s5 = df[df['score']>=5]
print(s5.groupby(s5['date'].dt.year).size().to_string())

print("\n=== 所有 score>=5 日期 (原始, 去重前) ===")
print(s5[['date','hsi_close','score']].to_string(index=False))

# HSI 主要季度高点 → 检查每个高点前后15交易日是否有 score>=5
print("\n=== HSI 季度高点覆盖检查 (前后15交易日有无 score>=5) ===")
df2 = df.sort_values('date').reset_index(drop=True)
# 用 60 交易日 rolling max 找局部高点 (峰)
df2['is_peak'] = (df2['hsi_close'] == df2['hsi_close'].rolling(60, min_periods=30).max()) & \
                 (df2['hsi_close'].shift(20) < df2['hsi_close']) & (df2['hsi_close'].shift(-20) < df2['hsi_close'] if len(df2)>20 else False)
peaks = df2[df2['is_peak']]
for _, p in peaks.iterrows():
    pi = p.name
    lo, hi = max(0, pi-15), min(len(df2), pi+16)
    window = df2.iloc[lo:hi]
    sigs = window[window['score']>=5]
    status = f"score={int(sigs['score'].max())} on {sigs['date'].iloc[0].strftime('%Y-%m-%d')}" if len(sigs) else "❌ MISSED"
    print(f"  peak {p['date'].strftime('%Y-%m-%d')} HSI={p['hsi_close']:.0f}  → {status}")


print("\n=== Cap 标记 (score>=5, gap>10 去重后) 全部日期 ===")
sig = df[df['score']>=5].copy()
if not sig.empty:
    sig_pos = df['date'].searchsorted(sig['date'])
    clusters, cur, cur_pos = [], [sig.iloc[0]], [sig_pos[0]]
    for i in range(1,len(sig)):
        if sig_pos[i]-cur_pos[-1]<=10:
            cur.append(sig.iloc[i]); cur_pos.append(sig_pos[i])
        else:
            clusters.append(cur); cur, cur_pos=[sig.iloc[i]],[sig_pos[i]]
    if cur: clusters.append(cur)
    picked=[sorted(c, key=lambda r:(-int(r['score']),-r['hsi_close']))[0].name for c in clusters]
    pdf=df.loc[picked].sort_values('date')
    print(f"共 {len(pdf)} 个 Cap 标记:")
    print(pdf[['date','hsi_close','score']].to_string(index=False))
else:
    print("没有 score>=5 的日期!")
