#!/usr/bin/env python3
"""
Pool state machine for Breakout Pool and Trend Pool.

Shared logic used by scan_us.py and scan_hk.py.

Pools:
  - Breakout Pool: stocks making 6-month highs for the first time in 42 days
  - Trend Pool: stocks with sustained trends (weekly bullish alignment)

State transitions:
  NEW → (5 days) → Trend Pool or EXITED
  STRONGEST ↔ PULLBACK ↔ WATCHING → BROKEN → (7 days) → EXITED
  EXITED → (re-breakout) → NEW (re-entry)
"""

import json
import os
import datetime as dt

try:
    import pandas as pd
except ImportError:
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "-q"])
    import pandas as pd

# ── Constants ──
BREAKOUT_LOOKBACK = 126       # ~6 months in trading days
BREAKOUT_FIRST_TIME = 42      # ~2 months in trading days
BREAKOUT_STAY_DAYS = 5        # Days to stay in breakout pool
BROKEN_REMOVE_DAYS = 7        # Days after BROKEN to remove from trend pool
FROM_BOTTOM_LOOKBACK_WEEKS = 4  # ~20 trading days = ~4 weeks

# Status hierarchy for upgrade/downgrade detection
STATUS_RANK = {
    "STRONGEST": 4,
    "PULLBACK": 3,
    "WATCHING": 2,
    "BROKEN": 1,
    "NEW": 0,
    "EXITED": -1,
}


# ── SMA helpers ──

def calc_sma(prices, period):
    """Calculate Simple Moving Average.

    >>> calc_sma([1, 2, 3, 4, 5], 3)
    4.0
    >>> calc_sma([1, 2], 3) is None
    True
    """
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def resample_to_weekly(dates, closes):
    """Resample daily closes to weekly (last trading day of each ISO week).

    Args:
        dates: list of date strings or datetime objects
        closes: list of close prices

    Returns:
        list of weekly close prices (chronological order)

    >>> resample_to_weekly(["2024-01-08","2024-01-09","2024-01-10","2024-01-11",
    ...                    "2024-01-12","2024-01-15","2024-01-16"],
    ...                    [100, 101, 102, 103, 104, 105, 106])
    [104, 106]
    """
    if len(dates) < 5:
        return []
    df = pd.DataFrame({"date": pd.to_datetime(dates), "close": closes})
    df = df.set_index("date").sort_index()
    iso = df.index.isocalendar()
    df["year"] = iso["year"].values
    df["week"] = iso["week"].values
    weekly = df.groupby(["year", "week"]).last()["close"]
    return weekly.tolist()


# ── Breakout detection ──

def check_6m_high(closes):
    """Check if current close is a 6-month high and first time in 42 days.

    Returns (is_high, is_first_in_42d):
      is_high: True if today's close >= max close in past 126 days
      is_first_in_42d: True if this is the first such event in past 42 days

    >>> # Monotonically increasing: today is high but yesterday was too → not first in 42d
    >>> check_6m_high(list(range(130)))[0]
    True
    >>> check_6m_high(list(range(130)))[1]
    False
    >>> # Flat then jump: today is the first 6m high in 42 days
    >>> series = [100]*126 + [101]
    >>> check_6m_high(series)[0]
    True
    >>> check_6m_high(series)[1]
    True
    >>> # Not a high
    >>> check_6m_high(list(range(100, 200)) + [150])[0]
    False
    """
    if len(closes) < BREAKOUT_LOOKBACK + 1:
        return False, False

    current_close = closes[-1]
    past_max = max(closes[-BREAKOUT_LOOKBACK - 1:-1])

    if current_close < past_max:
        return False, False

    # Today is a 6-month high. Check if it's the first in 42 days.
    for offset in range(1, min(BREAKOUT_FIRST_TIME, len(closes) - BREAKOUT_LOOKBACK)):
        idx = len(closes) - 1 - offset
        if idx < BREAKOUT_LOOKBACK:
            break
        day_close = closes[idx]
        day_past_max = max(closes[idx - BREAKOUT_LOOKBACK:idx])
        if day_close >= day_past_max:
            return True, False

    return True, True


# ── from_bottom detection ──

def check_from_bottom(weekly_closes):
    """Check from_bottom conditions:
    1. In past ~4 weeks, 5W MA crossed above 20W MA
    2. 20W MA slope 2nd derivative > 0

    Returns (from_bottom, ma_5w, ma_20w, ma_20w_slope_2nd)
    """
    if len(weekly_closes) < 22:
        return False, None, None, None

    # Check 1: 5W crossed above 20W in past ~4 weeks
    has_crossover = False
    for w in range(FROM_BOTTOM_LOOKBACK_WEEKS):
        end = len(weekly_closes) - w
        prev_end = end - 1
        if prev_end < 20:
            continue

        ma5_curr = sum(weekly_closes[end - 5:end]) / 5
        ma20_curr = sum(weekly_closes[end - 20:end]) / 20
        ma5_prev = sum(weekly_closes[prev_end - 5:prev_end]) / 5
        ma20_prev = sum(weekly_closes[prev_end - 20:prev_end]) / 20

        if ma5_prev <= ma20_prev and ma5_curr > ma20_curr:
            has_crossover = True
            break

    # Check 2: 20W MA slope 2nd derivative > 0
    ma_20w_series = []
    for i in range(24):
        end = len(weekly_closes) - i
        if end < 20:
            break
        ma20 = sum(weekly_closes[end - 20:end]) / 20
        ma_20w_series.insert(0, ma20)

    slope_2nd = 0.0
    if len(ma_20w_series) >= 3:
        slopes = [ma_20w_series[i + 1] - ma_20w_series[i]
                  for i in range(len(ma_20w_series) - 1)]
        slope_2nd = slopes[-1] - slopes[-2]

    ma_5w = sum(weekly_closes[-5:]) / 5
    ma_20w = sum(weekly_closes[-20:]) / 20

    from_bottom = has_crossover and slope_2nd > 0
    return from_bottom, ma_5w, ma_20w, slope_2nd


# ── Alignment checks ──

def check_weekly_alignment(ma_20w, ma_40w, ma_50w, ma_60w):
    """周线多头排列: 20W > 40W > 50W > 60W."""
    if any(v is None or v == 0 for v in [ma_20w, ma_40w, ma_50w, ma_60w]):
        return False
    return ma_20w > ma_40w > ma_50w > ma_60w


def check_daily_alignment(ma_10d, ma_15d, ma_20d, ma_30d, ma_40d):
    """日线多头排列: 10D > 15D > 20D > 30D > 40D."""
    if any(v is None or v == 0 for v in [ma_10d, ma_15d, ma_20d, ma_30d, ma_40d]):
        return False
    return ma_10d > ma_15d > ma_20d > ma_30d > ma_40d


# ── Trend Pool rating ──

def rate_trend_stock(last_close, ma_10d, ma_15d, ma_20d, ma_30d, ma_40d, ma_50d, weekly_aligned):
    """Rate a stock in Trend Pool.

    Rating (checked in order):
      BROKEN:   price < 50D OR weekly alignment broken
      STRONGEST: price > all daily MAs AND daily alignment
      WATCHING:  price < 30D or < 40D (daily alignment broken)
      PULLBACK:  price < 10D/15D/20D but 30D & 40D hold

    Returns (status, daily_aligned, broken_reason)
    """
    # BROKEN checks
    if ma_50d is not None and ma_50d > 0 and last_close < ma_50d:
        return "BROKEN", False, "跌破 50D"
    if not weekly_aligned:
        return "BROKEN", False, "周线多头排列破坏"

    daily_aligned = check_daily_alignment(ma_10d, ma_15d, ma_20d, ma_30d, ma_40d)

    # STRONGEST: price > all daily MAs AND daily alignment
    all_ma = [ma_10d, ma_15d, ma_20d, ma_30d, ma_40d]
    if daily_aligned and all(ma is not None and ma > 0 and last_close > ma for ma in all_ma):
        return "STRONGEST", daily_aligned, ""

    # WATCHING: price breaks 30D or 40D
    if (ma_30d is not None and ma_30d > 0 and last_close < ma_30d) or \
       (ma_40d is not None and ma_40d > 0 and last_close < ma_40d):
        return "WATCHING", daily_aligned, ""

    # PULLBACK: price breaks 10D/15D/20D but 30D & 40D hold
    breaks_short = any(
        ma is not None and ma > 0 and last_close < ma
        for ma in [ma_10d, ma_15d, ma_20d]
    )
    holds_long = (ma_30d is None or ma_30d == 0 or last_close >= ma_30d) and \
                 (ma_40d is None or ma_40d == 0 or last_close >= ma_40d)

    if breaks_short and holds_long:
        return "PULLBACK", daily_aligned, ""

    # Default: price > all MAs but daily alignment broken
    return "WATCHING", daily_aligned, ""


# ── Status change ──

def determine_status_change(prev_status, current_status):
    """Detect upgrade/downgrade between statuses.

    >>> determine_status_change("PULLBACK", "STRONGEST")
    'upgrade'
    >>> determine_status_change("STRONGEST", "WATCHING")
    'downgrade'
    >>> determine_status_change("STRONGEST", "STRONGEST")
    'none'
    """
    if prev_status == current_status:
        return "none"
    prev_rank = STATUS_RANK.get(prev_status, -1)
    curr_rank = STATUS_RANK.get(current_status, -1)
    if curr_rank > prev_rank:
        return "upgrade"
    elif curr_rank < prev_rank:
        return "downgrade"
    return "none"


# ── Trading day counter ──

def count_trading_days_since(dates, since_date_str, today_str):
    """Count trading days between since_date (exclusive) and today (inclusive).

    Args:
        dates: list of date strings in the price series
        since_date_str: start date string (YYYY-MM-DD), exclusive
        today_str: end date string (YYYY-MM-DD), inclusive
    """
    try:
        since_dt = dt.datetime.strptime(since_date_str, "%Y-%m-%d").date()
        today_dt = dt.datetime.strptime(today_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0
    count = 0
    for d in dates:
        d_dt = pd.to_datetime(d).date()
        if since_dt < d_dt <= today_dt:
            count += 1
    return count


# ── Pool state machine ──

def run_pool_state_machine(ticker, market, closes, dates, weekly_closes,
                           prev_entry, today_str, themes_info, bootstrap=False):
    """Run the pool state machine for a single stock.

    Args:
        ticker: stock ticker (e.g., "AAPL.US" or "0700.HK")
        market: "US" or "HK"
        closes: list of daily close prices (sorted by date, most recent last)
        dates: list of date strings corresponding to closes
        weekly_closes: list of weekly close prices
        prev_entry: previous pool entry dict, or None
        today_str: today's date string (YYYY-MM-DD)
        themes_info: dict with "sector" and "theme" keys, or {}

    Returns:
        Updated pool entry dict, or None if stock should be removed from pool
    """
    if len(closes) < BREAKOUT_LOOKBACK + 1:
        return None

    last_close = closes[-1]

    # ── Compute indicators ──
    ma_10d = calc_sma(closes, 10)
    ma_15d = calc_sma(closes, 15)
    ma_20d = calc_sma(closes, 20)
    ma_30d = calc_sma(closes, 30)
    ma_40d = calc_sma(closes, 40)
    ma_50d = calc_sma(closes, 50)

    ma_5w = calc_sma(weekly_closes, 5)
    ma_20w = calc_sma(weekly_closes, 20)
    ma_40w = calc_sma(weekly_closes, 40)
    ma_50w = calc_sma(weekly_closes, 50)
    ma_60w = calc_sma(weekly_closes, 60)

    weekly_aligned = check_weekly_alignment(ma_20w, ma_40w, ma_50w, ma_60w)
    daily_aligned = check_daily_alignment(ma_10d, ma_15d, ma_20d, ma_30d, ma_40d)

    is_high, is_first_in_42d = check_6m_high(closes)
    from_bottom, fb_ma_5w, fb_ma_20w, ma_20w_slope_2nd = check_from_bottom(weekly_closes)

    # Use from_bottom computation MAs if available (more accurate for the crossover check)
    if fb_ma_5w is not None:
        ma_5w = fb_ma_5w
    if fb_ma_20w is not None:
        ma_20w = fb_ma_20w

    # ── Previous state ──
    prev_pool = prev_entry.get("pool", "") if prev_entry else ""
    prev_status = prev_entry.get("pool_status", "") if prev_entry else ""

    # ── Build new entry ──
    entry = {
        "ticker": ticker,
        "market": market,
        "pool": prev_pool,
        "pool_status": prev_status,
        "prev_pool_status": prev_status,
        "status_change": "none",
        "first_breakout_date": prev_entry.get("first_breakout_date", "") if prev_entry else "",
        "last_breakout_date": prev_entry.get("last_breakout_date", "") if prev_entry else "",
        "breakout_count_60d": prev_entry.get("breakout_count_60d", 0) if prev_entry else 0,
        "from_bottom": prev_entry.get("from_bottom", False) if prev_entry else False,
        "days_in_trend": prev_entry.get("days_in_trend", 0) if prev_entry else 0,
        "last_close": round(last_close, 2),
        "weekly_aligned": weekly_aligned,
        "daily_aligned": daily_aligned,
        "ma_5w": round(ma_5w, 2) if ma_5w else 0,
        "ma_20w": round(ma_20w, 2) if ma_20w else 0,
        "ma_40w": round(ma_40w, 2) if ma_40w else 0,
        "ma_50w": round(ma_50w, 2) if ma_50w else 0,
        "ma_60w": round(ma_60w, 2) if ma_60w else 0,
        "ma_20w_slope_2nd": round(ma_20w_slope_2nd, 4),
        "ma_10d": round(ma_10d, 2) if ma_10d else 0,
        "ma_15d": round(ma_15d, 2) if ma_15d else 0,
        "ma_20d": round(ma_20d, 2) if ma_20d else 0,
        "ma_30d": round(ma_30d, 2) if ma_30d else 0,
        "ma_40d": round(ma_40d, 2) if ma_40d else 0,
        "ma_50d": round(ma_50d, 2) if ma_50d else 0,
        "sector": themes_info.get("sector", ""),
        "theme": themes_info.get("theme", ""),
        "broken_date": prev_entry.get("broken_date", "") if prev_entry else "",
        "breakout_dates": prev_entry.get("breakout_dates", []) if prev_entry else [],
        "last_update_date": today_str,
    }

    # ═══ Step 5: Re-entry for EXITED stocks ═══
    if prev_pool == "exited" and prev_status == "EXITED":
        if is_high and is_first_in_42d:
            entry["pool"] = "breakout"
            entry["pool_status"] = "NEW"
            entry["first_breakout_date"] = today_str
            entry["last_breakout_date"] = today_str
            entry["breakout_count_60d"] += 1
            entry["from_bottom"] = from_bottom
            entry["days_in_trend"] = 0
            entry["broken_date"] = ""
            entry["breakout_dates"] = entry["breakout_dates"] + [today_str]
            return entry
        else:
            # EXITED stock doesn't re-qualify → remove from pool data
            return None

    # ═══ Bootstrap: first run → fill pools from scratch ═══
    if bootstrap and prev_entry is None:
        # Priority 1: weekly aligned → Trend Pool directly (skip BROKEN)
        if weekly_aligned:
            status, daily_aligned, _ = rate_trend_stock(
                last_close, ma_10d, ma_15d, ma_20d, ma_30d, ma_40d, ma_50d, weekly_aligned
            )
            # BROKEN stocks are on their way out; don't bootstrap them
            if status != "BROKEN":
                entry["pool"] = "trend"
                entry["pool_status"] = status
                entry["daily_aligned"] = daily_aligned
                entry["prev_pool_status"] = ""
                entry["days_in_trend"] = 1
                entry["from_bottom"] = False
                if is_high:
                    entry["last_breakout_date"] = today_str
                    entry["breakout_count_60d"] = 1
                    entry["breakout_dates"] = [today_str]
                return entry
        # Priority 2: 6-month high first in 42d → Breakout Pool
        elif is_high and is_first_in_42d:
            entry["pool"] = "breakout"
            entry["pool_status"] = "NEW"
            entry["first_breakout_date"] = today_str
            entry["last_breakout_date"] = today_str
            entry["from_bottom"] = from_bottom
            entry["breakout_count_60d"] = 1
            entry["breakout_dates"] = [today_str]
            return entry
        else:
            return None

    # ═══ Step 1: Check for new 6-month high ═══
    if is_high:
        if is_first_in_42d and prev_pool not in ("breakout", "trend"):
            entry["pool"] = "breakout"
            entry["pool_status"] = "NEW"
            entry["first_breakout_date"] = today_str
            entry["from_bottom"] = from_bottom
            entry["breakout_count_60d"] += 1
            entry["breakout_dates"] = entry["breakout_dates"] + [today_str]
        entry["last_breakout_date"] = today_str

    # ═══ Step 2: Breakout Pool 5-day check ═══
    if entry["pool"] == "breakout" and entry["first_breakout_date"]:
        days_in_breakout = count_trading_days_since(
            dates, entry["first_breakout_date"], today_str
        )
        if days_in_breakout >= BREAKOUT_STAY_DAYS:
            if weekly_aligned:
                # 晋升 Trend Pool
                entry["pool"] = "trend"
                entry["days_in_trend"] = 1
                # Fall through to Step 3 for rating
            else:
                entry["pool"] = "exited"
                entry["pool_status"] = "EXITED"
                return entry

    # ═══ Step 3: Trend Pool daily rating ═══
    if entry["pool"] == "trend":
        status, daily_aligned, broken_reason = rate_trend_stock(
            last_close, ma_10d, ma_15d, ma_20d, ma_30d, ma_40d, ma_50d, weekly_aligned
        )
        entry["pool_status"] = status
        entry["daily_aligned"] = daily_aligned

        if status == "BROKEN":
            # Track when BROKEN first started
            if prev_status != "BROKEN":
                entry["broken_date"] = today_str
            else:
                entry["broken_date"] = prev_entry.get("broken_date", today_str) if prev_entry else today_str

            # Check if continuously BROKEN for 7 trading days → remove
            days_broken = count_trading_days_since(dates, entry["broken_date"], today_str)
            if days_broken >= BROKEN_REMOVE_DAYS:
                entry["pool"] = "exited"
                entry["pool_status"] = "EXITED"
                return entry
        else:
            # Recovered from BROKEN
            if prev_status == "BROKEN":
                entry["broken_date"] = ""

            # Increment days_in_trend
            if prev_entry and prev_entry.get("pool") == "trend":
                entry["days_in_trend"] = prev_entry.get("days_in_trend", 0) + 1

    # ═══ Step 4: Upgrade/downgrade detection ═══
    if entry["pool"] in ("trend", "breakout"):
        entry["status_change"] = determine_status_change(
            entry["prev_pool_status"], entry["pool_status"]
        )

    # Prune breakout_dates to last 90 calendar days (covers 60 trading days)
    try:
        cutoff = (dt.datetime.strptime(today_str, "%Y-%m-%d") - dt.timedelta(days=90)).strftime("%Y-%m-%d")
        entry["breakout_dates"] = [d for d in entry["breakout_dates"] if d >= cutoff]
    except ValueError:
        pass
    entry["breakout_count_60d"] = len(entry["breakout_dates"])

    # Only keep stocks in a meaningful pool (breakout or trend)
    # Empty pool or exited → not tracked
    if entry["pool"] not in ("breakout", "trend"):
        return None

    return entry


# ── File I/O ──

def load_pools(filepath):
    """Load pool state from JSON file. Returns dict keyed by ticker."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support both dict and list format
        if isinstance(data, list):
            return {e["ticker"]: e for e in data if "ticker" in e}
        return data
    except (json.JSONDecodeError, IOError):
        return {}


def save_pools(filepath, data):
    """Save pool state to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Save as sorted list for stable git diff
    items = sorted(data.values(), key=lambda x: x.get("ticker", ""))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def load_themes(filepath):
    """Load theme/sector tags from JSON file."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def generate_alerts(pools_data, today):
    """Generate alerts for today's events.

    Returns list of alert dicts with event types:
    NEW, upgrade, downgrade, BROKEN
    """
    alerts = []
    for ticker, entry in pools_data.items():
        if entry.get("last_update_date") != today:
            continue

        pool = entry.get("pool", "")
        status = entry.get("pool_status", "")
        prev_status = entry.get("prev_pool_status", "")
        change = entry.get("status_change", "none")

        if pool == "exited":
            continue

        # NEW breakout entry
        if status == "NEW" and prev_status != "NEW":
            alerts.append({
                "ticker": ticker,
                "event": "NEW",
                "from_bottom": entry.get("from_bottom", False),
                "date": today,
            })
        # BROKEN
        elif status == "BROKEN" and prev_status != "BROKEN":
            reason = ""
            if not entry.get("weekly_aligned", False):
                reason = "周线多头排列破坏"
            elif entry.get("ma_50d", 0) > 0 and entry.get("last_close", 0) < entry.get("ma_50d", 0):
                reason = "跌破 50D"
            alerts.append({
                "ticker": ticker,
                "event": "BROKEN",
                "reason": reason,
                "date": today,
            })
        # Upgrade
        elif change == "upgrade" and prev_status and status:
            alerts.append({
                "ticker": ticker,
                "event": "upgrade",
                "from": prev_status,
                "to": status,
                "date": today,
            })
        # Downgrade
        elif change == "downgrade" and prev_status and status:
            alerts.append({
                "ticker": ticker,
                "event": "downgrade",
                "from": prev_status,
                "to": status,
                "date": today,
            })

    # Sort by event priority: NEW > upgrade > downgrade > BROKEN
    event_order = {"NEW": 0, "upgrade": 1, "downgrade": 2, "BROKEN": 3}
    alerts.sort(key=lambda a: (event_order.get(a["event"], 99), a["ticker"]))
    return alerts


def save_alerts(filepath, alerts):
    """Save alerts to JSON file. Merges with existing alerts for the same day."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Load existing alerts to preserve same-day events from earlier runs
    existing = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = []

    # Collect dates from new alerts
    new_dates = {a.get("date") for a in alerts}
    # Keep existing alerts that are NOT from dates covered by new alerts
    preserved = [a for a in existing if a.get("date") not in new_dates]
    merged = preserved + alerts

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)


# ── Local test ──

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    # Simulated run: stock declining/flat, then sudden breakout on last day
    print("\n-- Simulation --")
    base_date = dt.date(2023, 1, 2)
    prices = []
    for i in range(300):
        if i <= 200:
            prices.append(150 - i * 0.3)  # Decline: 150 -> 90
        else:
            prices.append(90.0)            # Flat at 90
    prices[-1] = 160.0                     # Breakout!
    dates = [(base_date + dt.timedelta(days=i)).isoformat() for i in range(len(prices))]
    weekly = resample_to_weekly(dates, prices)

    is_high, is_first = check_6m_high(prices)
    print(f"6m high: {is_high}, first in 42d: {is_first}")

    result = run_pool_state_machine(
        ticker="TEST.US", market="US",
        closes=prices, dates=dates, weekly_closes=weekly,
        prev_entry=None, today_str="2024-10-25",
        themes_info={"sector": "Tech", "theme": "Test"},
    )
    if result:
        print(f"Pool: {result['pool']}, Status: {result['pool_status']}, from_bottom: {result['from_bottom']}")
    else:
        print("Not entered pool")
