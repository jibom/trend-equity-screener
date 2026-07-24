"""数据提供层 — 从 jianxin MySQL 读取港股日线数据 (自包含版, 凭据走环境变量, 无硬编码密码)。

环境变量: DB_HOST, DB_PORT(默认3306), DB_USER, DB_PASSWORD, DB_NAME(默认jianxin)
返回列: TRADE_DT, S_DQ_OPEN/HIGH/LOW/CLOSE, S_DQ_ADJOPEN/ADJHIGH/ADJLOW/ADJCLOSE, S_DQ_VOLUME
"""
from __future__ import annotations
import os
from datetime import datetime, timedelta
import pandas as pd
import pymysql
from sqlalchemy import create_engine, text

REQUIRED_COLS = [
    "TRADE_DT",
    "S_DQ_OPEN", "S_DQ_HIGH", "S_DQ_LOW", "S_DQ_CLOSE",
    "S_DQ_ADJOPEN", "S_DQ_ADJHIGH", "S_DQ_ADJLOW", "S_DQ_ADJCLOSE",
    "S_DQ_VOLUME",
]


class WindFetcher:
    """从 jianxin MySQL (Wind 镜像) 读取港股 EOD 数据。凭据从环境变量读取。"""

    def __init__(self, db: dict | None = None, lookback_days: int = 520, retries: int = 3):
        self._db = db or self._db_from_env()
        self.lookback_days = lookback_days
        self._retries = retries
        self._conn = self._connect()

    @staticmethod
    def _db_from_env() -> dict:
        user = os.environ.get("DB_USER")
        pwd = os.environ.get("DB_PASSWORD")
        host = os.environ.get("DB_HOST")
        if not (user and pwd and host):
            raise RuntimeError("缺少 DB 环境变量 (DB_HOST/DB_USER/DB_PASSWORD). "
                               "本地用 .env, GitHub Actions 用 Secrets.")
        return {
            "host": host, "user": user, "password": pwd,
            "database": os.environ.get("DB_NAME", "jianxin"),
            "port": int(os.environ.get("DB_PORT", "3306")),
            "charset": "utf8mb4",
        }

    def _connect(self):
        import time
        for attempt in range(self._retries):
            try:
                return pymysql.connect(**self._db)
            except Exception as e:
                if attempt < self._retries - 1:
                    print(f"  [DB] 连接失败 ({e}), {attempt+1}/{self._retries} 重试 ...")
                    time.sleep(2)
                else:
                    raise

    def fetch(self, code: str, asof: str) -> pd.DataFrame:
        """拉取单只股票日线 (前复权列 + 原始列)"""
        start = (datetime.strptime(asof, "%Y-%m-%d")
                 - timedelta(days=int(self.lookback_days * 1.6))).strftime("%Y%m%d")
        end = asof.replace("-", "")
        sql = f"""SELECT {",".join(REQUIRED_COLS)}
                  FROM hkshareeodprices
                  WHERE S_INFO_WINDCODE=%s AND TRADE_DT BETWEEN %s AND %s
                  ORDER BY TRADE_DT"""
        return pd.read_sql(sql, self._conn, params=(code, start, end))

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass


def forward_adjust(df: pd.DataFrame) -> pd.DataFrame:
    """后复权→前复权, 新增 fwd_open/high/low/close, volume, raw_close, date。最新日 fwd_close==raw_close。"""
    if df.empty:
        return df
    df = df.sort_values("TRADE_DT").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["TRADE_DT"], format="%Y%m%d")
    latest_raw = df["S_DQ_CLOSE"].iloc[-1]
    latest_adj = df["S_DQ_ADJCLOSE"].iloc[-1]
    if pd.isna(latest_raw) or pd.isna(latest_adj) or latest_raw == 0:
        return df.iloc[0:0]
    factor = latest_adj / latest_raw
    df["fwd_open"] = df["S_DQ_ADJOPEN"] / factor
    df["fwd_high"] = df["S_DQ_ADJHIGH"] / factor
    df["fwd_low"] = df["S_DQ_ADJLOW"] / factor
    df["fwd_close"] = df["S_DQ_ADJCLOSE"] / factor
    df["volume"] = df["S_DQ_VOLUME"]
    df["raw_close"] = df["S_DQ_CLOSE"]
    return df
