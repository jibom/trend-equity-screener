"""Configuration loader for Wyckoff v5.3 screener."""
from __future__ import annotations

import json
import os
from typing import Any


_DEFAULTS: dict[str, Any] = {
    "weekly_long_mas": [40, 50, 60, 70],
    "weekly_short_mas": [10, 20, 30, 40],
    "weekly_fast_ma": 5,
    "tolerance_window_weeks": 3,
    "tolerance_max_breaks": 2,

    "early_ma10_slope10d": 0.05,
    "early_entangle_disp": 0.08,
    "early_wma40_roc30": -0.05,

    "bear_gate_wma40_roc30": -0.10,

    "new_stock_days": 490,
    "new_stock_mas": [10, 20, 50],

    "exit_consec_below_ma10": 4,
    "pullback_max_days": 15,

    "exit_cooldown_days": 5,
    "exit_ma60_window": 60,
    "exit_ma60_slope_lookback": 5,

    "sos_volume_multiple": 1.5,
    "sos_big_body_pct": 0.60,
    "sos_close_loc": 0.60,

    "entangled_disp": 0.08,
    "setup_idle_days": 5,

    "lookback_days": 600,
    "warmup_days": 250,
    "sos_lookback_days": 30,
    "min_history_days": 230,
    "min_history_days_new": 60,
}


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load v5.3 config from JSON, falling back to built-in defaults."""
    cfg = dict(_DEFAULTS)
    if path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, '..', 'configs', 'v5_3.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            cfg.update(json.load(f))
    return cfg
