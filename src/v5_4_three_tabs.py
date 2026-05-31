"""v5.4 Three-tab outputter: 进场候选 / Trending / 警报.

Takes flat scan results and builds structured three-tab output.
All logic is display-layer only — no state machine modification.
"""
from __future__ import annotations

import math
import pandas as pd


def _nan_to_none(obj):
    """Recursively convert NaN/inf floats to None for valid JSON output."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _sector_counts(df: pd.DataFrame) -> dict[str, int]:
    return df['gics_sector'].value_counts().to_dict()


def build_tab1(df: pd.DataFrame) -> dict:
    """Tab 1 · 进场候选: SOS信号 + SETUP_OK/ENTANGLED候选池."""
    sec_counts = _sector_counts(df)

    # Upper section: SOS signals from SETUP_OK/ENTANGLED, last 5 trading days
    sos_pool = df[
        (df['state'].isin(['SETUP_OK', 'ENTANGLED'])) &
        (df['sos_setup_recent'] != '')
    ].copy()

    sos_tickers = set(sos_pool['ticker'])

    # Lower section: SETUP_OK/ENTANGLED pool (dedup from SOS section)
    pool_section = df[
        (df['state'].isin(['SETUP_OK', 'ENTANGLED'])) &
        (~df['ticker'].isin(sos_tickers))
    ].copy()

    # Sort SOS section: A → B → C, then sector count desc, then ma10_slope_pct desc
    sos_order = {'SOS-A': 0, 'SOS-B': 1, 'SOS-C': 2}
    sos_pool['sos_order'] = sos_pool['sos_setup_recent'].map(sos_order)
    sos_pool['sector_count'] = sos_pool['gics_sector'].map(sec_counts)
    sos_pool = sos_pool.sort_values(
        ['sos_order', 'sector_count', 'ma10_slope_pct'],
        ascending=[True, False, False],
    )

    # Sort pool section: sector count desc → ma10_slope_pct desc
    pool_section['sector_count'] = pool_section['gics_sector'].map(sec_counts)
    pool_section = pool_section.sort_values(
        ['sector_count', 'ma10_slope_pct'],
        ascending=[False, False],
    )

    # Drop internal sort columns before output
    drop_cols = ['sos_order', 'sector_count']
    sos_out = sos_pool.drop(columns=[c for c in drop_cols if c in sos_pool.columns]).to_dict('records')
    pool_out = pool_section.drop(columns=[c for c in drop_cols if c in pool_section.columns]).to_dict('records')

    return {
        'sos_section': sos_out,
        'pool_section': pool_out,
    }


def build_tab2(df: pd.DataFrame) -> list[dict]:
    """Tab 2 · Trending: TRENDING with valid substate (excluding recent EXIT and empty substate)."""
    trending = df[
        (df['state'] == 'TRENDING') &
        (df['substate'] != '') &
        (df.get('to_exit_recent', False) != True)
    ].copy()
    if trending.empty:
        return []

    def compute_display_tier(row):
        if row['substate'] in ('MID', 'EARLY') and row.get('recent_new_high_flag'):
            return 'STRONG'
        elif row['substate'] == 'NEW':
            return 'STRONG NEW'
        else:
            return row['substate']

    trending['display_tier'] = trending.apply(compute_display_tier, axis=1)

    sec_counts = _sector_counts(df)
    trending['sector_count'] = trending['gics_sector'].map(sec_counts)

    tier_order = {'STRONG': 0, 'STRONG NEW': 1, 'MID': 2, 'EARLY': 3, '': 4}
    trending['tier_order'] = trending['display_tier'].map(tier_order)

    trending = trending.sort_values(
        ['sector_count', 'tier_order', 'ma10_slope_pct'],
        ascending=[False, True, False],
    )

    # Drop internal sort columns before output
    drop_cols = ['sector_count', 'tier_order']
    trending_out = trending.drop(columns=[c for c in drop_cols if c in trending.columns]).to_dict('records')
    return trending_out


def build_tab3(df: pd.DataFrame) -> list[dict]:
    """Tab 3 · 警报: A·PULLBACK / B·EXIT / C·PULLBACK."""
    alerts = []

    # Ensure required columns exist with defaults
    for col, default in [('trending_to_pullback_recent', False), ('to_exit_recent', False),
                          ('days_in_pullback', 0), ('pullback_dd_pct', None),
                          ('name_cn', ''), ('gics_sector', ''), ('last_close', None),
                          ('days_in_state', 0)]:
        if col not in df.columns:
            df[col] = default

    # A·PULLBACK 🟡: TRENDING→PULLBACK in last 5d
    a_pullback = df[
        (df['state'] == 'PULLBACK') & (df['trending_to_pullback_recent'] == True)
    ]
    for _, row in a_pullback.iterrows():
        alerts.append({
            'ticker': row['ticker'],
            'name_cn': row.get('name_cn', row.get('name', '')),
            'gics_sector': row.get('gics_sector', ''),
            'alert_type': 'A_PULLBACK',
            'label': 'A·PULLBACK',
            'emoji': '\U0001f7e1',
            'pullback_dd_pct': row.get('pullback_dd_pct'),
            'days_in_pullback': row.get('days_in_pullback', 0),
            'last_close': row.get('last_close'),
        })

    # B·EXIT 🔴: EXIT triggered in last 5d
    b_exit = df[
        (df['state'] == 'EXIT') & (df['to_exit_recent'] == True)
    ]
    for _, row in b_exit.iterrows():
        alerts.append({
            'ticker': row['ticker'],
            'name_cn': row.get('name_cn', row.get('name', '')),
            'gics_sector': row.get('gics_sector', ''),
            'alert_type': 'B_EXIT',
            'label': 'B·EXIT',
            'emoji': '\U0001f534',
            'days_in_state': row.get('days_in_state', 0),
            'last_close': row.get('last_close'),
        })

    # C·PULLBACK 🟠: Still PULLBACK >= 10 days (not type A)
    c_pullback = df[
        (df['state'] == 'PULLBACK') &
        (df.get('days_in_pullback', 0) >= 10) &
        (df['trending_to_pullback_recent'] != True)
    ]
    for _, row in c_pullback.iterrows():
        alerts.append({
            'ticker': row['ticker'],
            'name_cn': row.get('name_cn', row.get('name', '')),
            'gics_sector': row.get('gics_sector', ''),
            'alert_type': 'C_PULLBACK',
            'label': 'C·PULLBACK',
            'emoji': '\U0001f7e0',
            'pullback_dd_pct': row.get('pullback_dd_pct'),
            'days_in_pullback': row.get('days_in_pullback', 0),
            'last_close': row.get('last_close'),
        })

    alert_order = {'A_PULLBACK': 0, 'B_EXIT': 1, 'C_PULLBACK': 2}
    alerts.sort(key=lambda x: (
        alert_order.get(x['alert_type'], 9),
        x.get('pullback_dd_pct') or 0,
    ))

    return alerts


def build_three_tabs(results_list: list[dict]) -> dict:
    """Build three-tab structured output from flat scan results."""
    df = pd.DataFrame(results_list)
    scan_date = df['date'].iloc[0] if len(df) else ''

    result = {
        'date': scan_date,
        'tab1': build_tab1(df),
        'tab2': build_tab2(df),
        'tab3': build_tab3(df),
    }
    return _nan_to_none(result)
