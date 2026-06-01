"""Data provider abstraction for Wyckoff screener.

Provides WindFetcher (MySQL) and forward_adjust().
Plug in your own provider by matching the output column schema.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import create_engine


REQUIRED_COLS = [
    'TRADE_DT',
    'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE',
    'S_DQ_ADJOPEN', 'S_DQ_ADJHIGH', 'S_DQ_ADJLOW', 'S_DQ_ADJCLOSE',
    'S_DQ_VOLUME',
]


class WindFetcher:
    """Reads HK EOD prices from the jianxin MySQL mirror of Wind."""

    DEFAULT_DB = dict(
        host=os.getenv('WIND_HOST', 'rm-uf62imd2xjxj647jho.mysql.rds.aliyuncs.com'),
        user=os.getenv('WIND_USER', 'yangdong_gf'),
        password=os.getenv('WIND_PASSWORD', '4S7Q4pNUzh'),
        database=os.getenv('WIND_DB', 'jianxin'),
        port=int(os.getenv('WIND_PORT', '3306')),
    )

    def __init__(self, db: dict | None = None, lookback_days: int = 420):
        cfg = db or self.DEFAULT_DB
        url = (f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
               f"@{cfg['host']}:{cfg['port']}/{cfg['database']}")
        self._engine = create_engine(url)
        self.lookback_days = lookback_days

    def fetch(self, code: str, asof: str) -> pd.DataFrame:
        start = (datetime.strptime(asof, '%Y-%m-%d')
                 - timedelta(days=int(self.lookback_days * 1.6))).strftime('%Y%m%d')
        end = asof.replace('-', '')
        sql = f"""SELECT {",".join(REQUIRED_COLS)}
                  FROM hkshareeodprices
                  WHERE S_INFO_WINDCODE=%s AND TRADE_DT BETWEEN %s AND %s
                  ORDER BY TRADE_DT"""
        return pd.read_sql(sql, self._engine, params=(code, start, end))

    def close(self):
        self._engine.dispose()


def forward_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Append forward-adjusted OHLCV columns to a raw Wind DataFrame.

    fwd_close(t) = adj_close(t) / latest_adj_factor
    where latest_adj_factor = adj_close(latest) / raw_close(latest).
    Today's fwd_close == today's raw_close (matches Bloomberg).

    Adds: date, fwd_open, fwd_high, fwd_low, fwd_close, volume, raw_close
    """
    if df.empty:
        return df
    df = df.sort_values('TRADE_DT').reset_index(drop=True).copy()
    df['date'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')

    latest_raw = df['S_DQ_CLOSE'].iloc[-1]
    latest_adj = df['S_DQ_ADJCLOSE'].iloc[-1]
    if pd.isna(latest_raw) or pd.isna(latest_adj) or latest_raw == 0:
        return df.iloc[0:0]

    factor = latest_adj / latest_raw
    df['fwd_open']  = df['S_DQ_ADJOPEN']  / factor
    df['fwd_high']  = df['S_DQ_ADJHIGH']  / factor
    df['fwd_low']   = df['S_DQ_ADJLOW']   / factor
    df['fwd_close'] = df['S_DQ_ADJCLOSE'] / factor
    df['volume']    = df['S_DQ_VOLUME']
    df['raw_close'] = df['S_DQ_CLOSE']
    return df
