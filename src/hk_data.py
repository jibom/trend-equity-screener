"""港股统一数据 provider: jianxin DB 主源, 缺数据时按 EODHD → yfinance 顺序补尾部。

jianxin 有完整历史但可能缺最近几天 (如港股分支停更)。本模块对每只代码:
1. 从 jianxin 拿全历史;
2. 若 jianxin 末日 < 目标 end, 仅补缺失的尾部几天 (EODHD → yfinance), 不重拉全历史;
3. 返回与 jianxin 同 schema 的 DataFrame, 现有 forward_adjust_group / compute_one 无感复用。

代码格式: 个股三源都是 9988.HK (无映射); HSI 指数 jianxin=HSI.HI, yfinance=^HSI (EODHD 无 HSI)。
HSICS 行业指数 (HSCIEN 等) EODHD/yfinance 均无, 仅 jianxin (停更时返回部分 + warning)。
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import pymysql
from db_config import DB_CONFIG

EODHD_KEY = os.getenv('EODHD_KEY', '6a10e8411d06d1.41490389')
EODHD_BASE = 'https://eodhd.com/api'

# 个股 schema (与 jianxin hkshareeodprices 对齐, 供 forward_adjust_group/compute_one 用)
STOCK_COLS = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE',
              'S_DQ_VOLUME', 'S_DQ_AMOUNT', 'S_DQ_ADJOPEN', 'S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'S_DQ_ADJCLOSE']
INDEX_COLS = ['S_INFO_WINDCODE', 'TRADE_DT', 'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']

# HSI 指数 jianxin → yfinance 代码映射 (EODHD 无 HSI)
YF_INDEX_MAP = {'HSI.HI': '^HSI', 'HSCEI.HI': '^HSCE', 'HSTECH.HI': '^HSTECH'}


# ---------------- jianxin 主源 ----------------

def _jianxin_index(code: str, start: str, end: str) -> pd.DataFrame:
    """start/end 为 YYYYMMDD。返回 INDEX_COLS。"""
    conn = pymysql.connect(**DB_CONFIG)
    try:
        df = pd.read_sql(
            f"SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE "
            f"FROM hkindexeodprices WHERE S_INFO_WINDCODE='{code}' "
            f"AND TRADE_DT BETWEEN '{start}' AND '{end}' ORDER BY TRADE_DT", conn)
    finally:
        conn.close()
    for c in ['S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']:
        df[c] = df[c].astype(float)
    return df


def _jianxin_stocks(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """分批查 jianxin hkshareeodprices。返回 STOCK_COLS。"""
    conn = pymysql.connect(**DB_CONFIG)
    parts = []
    try:
        BATCH = 50
        for bi in range(0, len(codes), BATCH):
            sql_codes = ','.join(f"'{c}'" for c in codes[bi:bi + BATCH])
            df_b = pd.read_sql(
                f"SELECT {', '.join(STOCK_COLS)} FROM hkshareeodprices "
                f"WHERE TRADE_DT BETWEEN '{start}' AND '{end}' AND S_INFO_WINDCODE IN ({sql_codes}) "
                f"ORDER BY S_INFO_WINDCODE, TRADE_DT", conn)
            parts.append(df_b)
    finally:
        conn.close()
    if not parts:
        return pd.DataFrame(columns=STOCK_COLS)
    df = pd.concat(parts, ignore_index=True)
    for c in STOCK_COLS:
        if c not in ('S_INFO_WINDCODE', 'TRADE_DT'):
            df[c] = df[c].astype(float)
    return df


# ---------------- EODHD 备用 (个股) ----------------

def _eodhd_bulk_day(date_ymd: str) -> dict[str, dict]:
    """EODHD /eod-bulk-last-day/HK?date=YYYY-MM-DD, 返回 {code: {open,high,low,close,volume}}。
    date_ymd 为 YYYYMMDD。非交易日或失败返回 {}。"""
    import requests
    d = f"{date_ymd[:4]}-{date_ymd[4:6]}-{date_ymd[6:8]}"
    try:
        r = requests.get(f"{EODHD_BASE}/eod-bulk-last-day/HK",
                         params={'api_token': EODHD_KEY, 'fmt': 'json', 'date': d}, timeout=60)
        if r.status_code != 200:
            return {}
        rows = r.json()
        out = {}
        for x in rows:
            code = x.get('code')
            if not code:
                continue
            if not code.endswith('.HK'):
                code = f"{code}.HK"   # bulk 返回的 code 无 .HK 后缀, 补上
            out[code] = {'open': float(x['open']), 'high': float(x['high']),
                         'low': float(x['low']), 'close': float(x['close']),
                         'volume': float(x['volume'])}
        return out
    except Exception:
        return {}


def _eodhd_tail(codes: list[str], tail_start: str, end: str) -> dict[str, list[dict]]:
    """用 bulk-last-day 逐交易日拉 EODHD, 返回 {code: [{TRADE_DT, OHLCV}]}, 覆盖 tail_start~end。
    只拉工作日; 空响应跳过。"""
    out = {c: [] for c in codes}
    d = pd.to_datetime(tail_start)
    e = pd.to_datetime(end)
    while d <= e:
        if d.weekday() < 5:  # Mon-Fri
            ds = d.strftime('%Y%m%d')
            bulk = _eodhd_bulk_day(ds)
            if bulk:
                for c in codes:
                    if c in bulk:
                        b = bulk[c]
                        out[c].append({'TRADE_DT': ds, 'S_DQ_OPEN': b['open'], 'S_DQ_HIGH': b['high'],
                                       'S_DQ_LOW': b['low'], 'S_DQ_CLOSE': b['close'], 'S_DQ_VOLUME': b['volume']})
        d += pd.Timedelta(days=1)
    return out


# ---------------- yfinance 备用 ----------------

def _yf_index(code: str, start: str, end: str) -> pd.DataFrame:
    """yfinance 拉指数 (^HSI 等), start/end YYYYMMDD。返回 INDEX_COLS (TRADE_DT YYYYMMDD)。"""
    import yfinance as yf
    yf_code = YF_INDEX_MAP.get(code)
    if not yf_code:
        return pd.DataFrame(columns=INDEX_COLS)
    s = pd.to_datetime(start).strftime('%Y-%m-%d')
    e = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # yfinance end exclusive
    try:
        tk = yf.Ticker(yf_code)
        h = tk.history(start=s, end=e, auto_adjust=False)
    except Exception:
        return pd.DataFrame(columns=INDEX_COLS)
    if h is None or h.empty:
        return pd.DataFrame(columns=INDEX_COLS)
    h = h.reset_index()
    h['TRADE_DT'] = pd.to_datetime(h['Date']).dt.strftime('%Y%m%d')
    out = pd.DataFrame({
        'S_INFO_WINDCODE': code,
        'TRADE_DT': h['TRADE_DT'],
        'S_DQ_OPEN': h['Open'].astype(float),
        'S_DQ_HIGH': h['High'].astype(float),
        'S_DQ_LOW': h['Low'].astype(float),
        'S_DQ_CLOSE': h['Close'].astype(float),
    })
    return out


def _yf_stocks(codes: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """yfinance.download 批量拉个股, 返回 {code: DataFrame[TRADE_DT, OHLCV]}。"""
    import yfinance as yf
    s = pd.to_datetime(start).strftime('%Y-%m-%d')
    e = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    try:
        h = yf.download(codes, start=s, end=e, auto_adjust=False, progress=False, threads=True)
    except Exception:
        return {}
    if h is None or h.empty:
        return {}
    out = {}
    multi = isinstance(h.columns, pd.MultiIndex)
    for c in codes:
        try:
            if multi:
                sub = h.xs(c, level=1, axis=1)
            else:
                sub = h.copy()
            sub = sub.reset_index().dropna(subset=['Open'])
            if sub.empty:
                continue
            sub['TRADE_DT'] = pd.to_datetime(sub['Date']).dt.strftime('%Y%m%d')
            out[c] = pd.DataFrame({
                'TRADE_DT': sub['TRADE_DT'],
                'S_DQ_OPEN': sub['Open'].astype(float),
                'S_DQ_HIGH': sub['High'].astype(float),
                'S_DQ_LOW': sub['Low'].astype(float),
                'S_DQ_CLOSE': sub['Close'].astype(float),
                'S_DQ_VOLUME': sub['Volume'].astype(float),
            })
        except Exception:
            continue
    return out


# ---------------- 公共 API ----------------

def fetch_hk_index(code: str, start: str, end: str) -> pd.DataFrame:
    """指数 OHLC (HSI.HI 等)。jianxin 主源; 缺尾部时 yfinance 补 (HSI→^HSI)。返回 INDEX_COLS。
    start/end 为 YYYYMMDD 字符串。"""
    df = _jianxin_index(code, start, end)
    jx_latest = df['TRADE_DT'].max() if not df.empty else None
    if jx_latest is not None and jx_latest >= end:
        return df
    # 补尾部
    tail_start = (pd.to_datetime(jx_latest) + pd.Timedelta(days=1)).strftime('%Y%m%d') if jx_latest else start
    if code in YF_INDEX_MAP:
        tail = _yf_index(code, tail_start, end)
        if not tail.empty:
            df = pd.concat([df, tail], ignore_index=True).drop_duplicates('TRADE_DT').sort_values('TRADE_DT').reset_index(drop=True)
            print(f"[hk_data] {code}: jianxin→{jx_latest}, yfinance 补尾至 {tail['TRADE_DT'].max()}")
    else:
        print(f"[hk_data] {code}: jianxin→{jx_latest}, 无备用源 (HSICS 等), 返回部分")
    return df


def fetch_hk_stocks(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """个股 OHLCV。jianxin 主源; 缺尾部时 EODHD → yfinance 补。返回 STOCK_COLS (复权列用 jianxin 末日 factor 对齐)。
    start/end 为 YYYYMMDD 字符串。"""
    df = _jianxin_stocks(codes, start, end)
    # 每只 code 的 jianxin 末日 + factor
    tail_rows = []
    need_tail = []  # 末 日 < end 的 code
    factor_by_code = {}  # code → factor (adj/raw)
    latest_by_code = {}
    for c in codes:
        sub = df[df['S_INFO_WINDCODE'] == c]
        if sub.empty:
            need_tail.append(c)
            latest_by_code[c] = None
            factor_by_code[c] = 1.0
            continue
        jx_latest = sub['TRADE_DT'].max()
        latest_by_code[c] = jx_latest
        last = sub[sub['TRADE_DT'] == jx_latest].iloc[0]
        factor = float(last['S_DQ_ADJCLOSE'] / last['S_DQ_CLOSE']) if last['S_DQ_CLOSE'] else 1.0
        factor_by_code[c] = factor
        if jx_latest < end:
            need_tail.append(c)
    if not need_tail:
        return df

    # 尾部起始 = need_tail 里最早的 latest+1 (或 start)
    valid_latests = [pd.to_datetime(latest_by_code[c]) for c in need_tail if latest_by_code[c]]
    tail_start = (min(valid_latests) + pd.Timedelta(days=1)).strftime('%Y%m%d') if valid_latests else start

    # 1) EODHD bulk 逐日
    eodhd_tail = _eodhd_tail(need_tail, tail_start, end) if valid_latests else {}
    still_need = [c for c in need_tail if not eodhd_tail.get(c)]
    for c in need_tail:
        for row in eodhd_tail.get(c, []):
            tail_rows.append(_make_stock_row(c, row, factor_by_code[c]))

    # 2) yfinance 补剩余
    if still_need:
        yf_tail = _yf_stocks(still_need, tail_start, end)
        for c in still_need:
            sub = yf_tail.get(c)
            if sub is None or sub.empty:
                continue
            for _, r in sub.iterrows():
                tail_rows.append(_make_stock_row(c, r, factor_by_code[c]))

    if tail_rows:
        tail_df = pd.DataFrame(tail_rows)
        for c in STOCK_COLS:
            if c not in ('S_INFO_WINDCODE', 'TRADE_DT') and c in tail_df.columns:
                tail_df[c] = tail_df[c].astype(float)
        df = pd.concat([df[STOCK_COLS], tail_df[STOCK_COLS]], ignore_index=True)
        df = df.drop_duplicates(subset=['S_INFO_WINDCODE', 'TRADE_DT']).sort_values(['S_INFO_WINDCODE', 'TRADE_DT']).reset_index(drop=True)
        supplemented = sorted({c for c in need_tail if any(r['S_INFO_WINDCODE'] == c for r in tail_rows)})
        print(f"[hk_data] 个股补尾: jianxin 缺 {len(need_tail)} 只, EODHD+yfinance 补 {len(supplemented)} 只 (tail {tail_start}~{end})")
    else:
        print(f"[hk_data] 个股补尾失败: jianxin 缺 {len(need_tail)} 只, 备用源均无 (tail {tail_start}~{end})")
    return df


def _make_stock_row(code: str, raw: dict, factor: float) -> dict:
    """把原始 OHLCV 行转成 jianxin schema, adj = raw × factor。"""
    o = float(raw['S_DQ_OPEN']); h = float(raw['S_DQ_HIGH']); lo = float(raw['S_DQ_LOW']); c = float(raw['S_DQ_CLOSE'])
    v = float(raw.get('S_DQ_VOLUME', 0))
    return {
        'S_INFO_WINDCODE': code,
        'TRADE_DT': str(raw['TRADE_DT']),
        'S_DQ_OPEN': o, 'S_DQ_HIGH': h, 'S_DQ_LOW': lo, 'S_DQ_CLOSE': c,
        'S_DQ_VOLUME': v, 'S_DQ_AMOUNT': c * v,
        'S_DQ_ADJOPEN': o * factor, 'S_DQ_ADJHIGH': h * factor, 'S_DQ_ADJLOW': lo * factor, 'S_DQ_ADJCLOSE': c * factor,
    }
