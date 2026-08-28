"""DB config for trend-hotspot scripts (jianxin MySQL mirror of Wind).

复用 data_provider.WindFetcher._default_db() (env: WIND_* 或 CI 的 DB_*),
本文件不存放任何凭据, 避免凭据再次进入 git 历史。
本地凭据放项目根 .env (gitignored); CI 通过 secrets 提供。
"""
import os
from data_provider import WindFetcher

DB_CONFIG = {**WindFetcher._default_db(), "charset": "utf8mb4"}

SECTOR_MAP_FILE = os.path.join(os.path.dirname(__file__), "..", "configs", "hk_sector_map.csv")
