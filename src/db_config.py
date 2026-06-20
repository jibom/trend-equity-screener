"""DB config for trend-hotspot scripts (jianxin MySQL mirror of Wind).

复用 data_provider.DEFAULT_DB (其内部用 os.getenv 带 WIND_* 默认值), 本文件不存放任何凭据,
避免凭据再次进入 git 历史。CI 通过 WIND_* secrets 覆盖; 本地走 data_provider 的默认值。
"""
import os
from data_provider import WindFetcher

DB_CONFIG = {**WindFetcher.DEFAULT_DB, "charset": "utf8mb4"}

SECTOR_MAP_FILE = os.path.join(os.path.dirname(__file__), "..", "configs", "hk_sector_map.csv")
