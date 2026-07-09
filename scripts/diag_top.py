"""诊断逃顶各子信号有效性 + 2018顶部窗口复盘."""
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
wr=pd.DataFrame({'date':df_w.index,'w_kdj_j':wj,'w_kdj_j_high4':wj_high4}); wr['date']=pd.to_datetime(wr['date'])
for _,r in wr.iterrows():
    m=df['date']>=r['date']; df.loc[m,'w_kdj_j']=r['w_kdj_j']; df.loc[m,'w_kdj_j_high4']=r['w_kdj_j_high4']
df=df.merge(hsi[['date','rsi','cap_z']],on='date',how='left')
df['cum_pct_4w']=df['active_top_pct'].rolling(20,min_periods=20).sum()
df['pctile_4w']=df['cum_pct_4w'].expanding(min_periods=60).rank(pct=True)*100
b10=df['breadth_below_ma50'].expanding(min_periods=252).quantile(0.10).shift(1)
sky95=df['sky_vol_pct'].expanding(min_periods=252).quantile(0.99).shift(1)
dist95=df['dist_top_pct'].expanding(min_periods=252).quantile(0.95).shift(1)
shrink95=df['shrink_new_high_pct'].expanding(min_periods=252).quantile(0.95).shift(1)
df['bias20']=(df['hsi_close']/df['hsi_close'].rolling(20).mean()-1)*100
bias95=df['bias20'].expanding(min_periods=252).quantile(0.95).shift(1)
rsi_lb=10
LOOK=10
recent=lambda s: s.rolling(LOOK,min_periods=1).max().fillna(0).astype(int)
df['c_rsi']=(df['rsi'].rolling(rsi_lb,min_periods=1).max()>70).astype(int)
df['c_kdj']=recent((df['w_kdj_j_high4']>100).fillna(False))
df['c_star']=(df['pctile_4w']>=90).astype(int)
df['c_brd']=(df['breadth_below_ma50']<=b10).fillna(False).astype(int)
df['c_cap']=(df['cap_z']>=2.5).astype(int)
df['c_sky']=(df['sky_vol_pct']>=sky95).fillna(False).astype(int)
df['c_div']=(df['vol_price_div_pct']>0).astype(int)
df['c_dist']=(df['dist_top_pct']>=dist95).fillna(False).astype(int)
df['c_shrink']=(df['shrink_new_high_pct']>=shrink95).fillna(False).astype(int)
df['c_bias']=recent((df['bias20']>=bias95).fillna(False))
CONDS=['c_rsi','c_kdj','c_star','c_brd','c_cap','c_sky','c_div','c_dist','c_shrink','c_bias']
df['score']=df[CONDS].sum(axis=1)
for N in (5,20,60): df[f'fwd_{N}']=df['hsi_close'].shift(-N)/df['hsi_close']-1
pd.set_option('display.width',320); pd.set_option('display.max_columns',60)

print("=== 2018-01 ~ 2018-03 HSI 顶部窗口 (每周采样) ===")
w=df[(df['date']>='2018-01-01')&(df['date']<='2018-03-31')]
show=['date','hsi_close','score','c_rsi','c_kdj','c_brd','c_cap','c_bias','rsi','w_kdj_j_high4','bias20','cap_z','breadth_below_ma50']
print(w[show].iloc[::5].to_string(index=False))

print("\n=== 各子信号独立有效性 (触发后下跌概率 / 预期收益) ===")
base60=(df['fwd_60'].dropna()<0).mean()*100; base20=(df['fwd_20'].dropna()<0).mean()*100; base5=(df['fwd_5'].dropna()<0).mean()*100
ber60=df['fwd_60'].dropna().mean()*100
print(f"{'cond':10s} {'hit':>5} {'wr5':>5} {'wr20':>5} {'wr60':>5} {'er5':>7} {'er20':>7} {'er60':>7}  (基准 wr5/20/60={base5:.0f}/{base20:.0f}/{base60:.0f}% er60={ber60:+.1f}%)")
for c in CONDS:
    m=df[c]==1; n=int(m.sum())
    if n==0: print(f"{c:10s} {n:>5} (no fire)"); continue
    w5=(df.loc[m,'fwd_5'].dropna()<0).mean()*100; w20=(df.loc[m,'fwd_20'].dropna()<0).mean()*100; w60=(df.loc[m,'fwd_60'].dropna()<0).mean()*100
    e5=df.loc[m,'fwd_5'].dropna().mean()*100; e20=df.loc[m,'fwd_20'].dropna().mean()*100; e60=df.loc[m,'fwd_60'].dropna().mean()*100
    print(f"{c:10s} {n:>5} {w5:>5.0f} {w20:>5.0f} {w60:>5.0f} {e5:>+7.1f} {e20:>+7.1f} {e60:>+7.1f}")

print("\n=== 2015-04-27 (HSI 顶) 各 breadth expanding 百分位 ===")
ath = df[df['date']>='2015-04-27'].head(1)
idx = ath.index[0]
for col in ['active_top_pct','union_up_pct','sky_vol_pct','vol_price_div_pct','dist_top_pct','shrink_new_high_pct','breadth_below_ma50']:
    s = df[col].expanding(min_periods=252).rank(pct=True)*100
    print(f"  {col:24s} value={df[col].iloc[idx]:.3f}  pctile={s.iloc[idx]:.0f}")
print(f"  HSI={df['hsi_close'].iloc[idx]:.0f}  score={df['score'].iloc[idx]}  rsi={df['rsi'].iloc[idx]:.1f}  wJ_high4={df['w_kdj_j_high4'].iloc[idx]:.1f}  bias20={df['bias20'].iloc[idx]:.2f}%  cap_z={df['cap_z'].iloc[idx]:.2f}")
