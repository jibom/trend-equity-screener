"""Tests for v5.4 three-tab output logic against 2026-05-29 baseline."""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from v5_4_three_tabs import build_three_tabs, build_tab1, build_tab2, build_tab3


def _make_mock_results():
    """Create mock scan results matching the 2026-05-29 baseline."""
    results = []

    # Tab 1 SOS: 9992 泡泡玛特 SOS-B (Consumer Discretionary, SETUP_OK)
    results.append({
        'ticker': '9992.HK', 'name': '泡泡玛特', 'name_cn': '泡泡玛特',
        'gics_sector': 'Consumer Discretionary', 'sub_industry': '玩具',
        'state': 'SETUP_OK', 'substate': '',
        'last_close': 185.5, 'ma10': 180.2, 'ma20': 175.0, 'ma60': 165.0,
        'ma200': 150.0, 'ma200_slope': 12.5, 'ma10_slope_pct': 3.2,
        'days_in_state': 3, 'days_in_pullback': 0, 'pullback_dd_pct': None,
        'sos': 'SOS-B', 'bear_gate': False, 'is_new_stock': False,
        'sos_setup_recent': 'SOS-B', 'recent_new_high_flag': False,
        'trending_to_pullback_recent': False, 'to_exit_recent': False,
        'date': '2026-05-29',
    })

    # Tab 2: 7 IT STRONG
    for i, ticker in enumerate(['0981.HK', '0999.HK', '0700.HK', '0020.HK', '0285.HK', '0369.HK', '2015.HK']):
        results.append({
            'ticker': ticker, 'name': f'IT{i}', 'name_cn': f'IT股票{i}',
            'gics_sector': 'Information Technology', 'sub_industry': '软件',
            'state': 'TRENDING', 'substate': 'STRONG',
            'last_close': 100 + i * 5, 'ma10': 98 + i * 5, 'ma20': 95 + i * 5,
            'ma60': 90 + i * 5, 'ma200': 80 + i * 5, 'ma200_slope': 8.0 + i,
            'ma10_slope_pct': 2.0 + i * 0.3,
            'days_in_state': 20 + i, 'days_in_pullback': 0, 'pullback_dd_pct': None,
            'sos': '', 'bear_gate': False, 'is_new_stock': False,
            'sos_setup_recent': '', 'recent_new_high_flag': False,
            'trending_to_pullback_recent': False, 'to_exit_recent': False,
            'date': '2026-05-29',
        })

    # Tab 2: 1 Industrials STRONG
    results.append({
        'ticker': '0001.HK', 'name': 'CKH', 'name_cn': '长江和记',
        'gics_sector': 'Industrials', 'sub_industry': '综合企业',
        'state': 'TRENDING', 'substate': 'STRONG',
        'last_close': 70.5, 'ma10': 72.0, 'ma20': 71.8, 'ma60': 70.0,
        'ma200': 64.0, 'ma200_slope': 1.8, 'ma10_slope_pct': 1.5,
        'days_in_state': 25, 'days_in_pullback': 0, 'pullback_dd_pct': None,
        'sos': '', 'bear_gate': False, 'is_new_stock': False,
        'sos_setup_recent': '', 'recent_new_high_flag': False,
        'trending_to_pullback_recent': False, 'to_exit_recent': False,
        'date': '2026-05-29',
    })

    # Tab 2: 1 Materials MID
    results.append({
        'ticker': '1112.HK', 'name': 'BYD', 'name_cn': '比亚迪电子',
        'gics_sector': 'Materials', 'sub_industry': '电子',
        'state': 'TRENDING', 'substate': 'MID',
        'last_close': 45.0, 'ma10': 44.0, 'ma20': 43.0, 'ma60': 40.0,
        'ma200': 35.0, 'ma200_slope': 5.0, 'ma10_slope_pct': 1.0,
        'days_in_state': 12, 'days_in_pullback': 0, 'pullback_dd_pct': None,
        'sos': '', 'bear_gate': False, 'is_new_stock': False,
        'sos_setup_recent': '', 'recent_new_high_flag': False,
        'trending_to_pullback_recent': False, 'to_exit_recent': False,
        'date': '2026-05-29',
    })

    # Tab 3: 0941 中国移动 A·PULLBACK 🟡 (-1.9%, 2天前)
    results.append({
        'ticker': '0941.HK', 'name': 'China Mobile', 'name_cn': '中国移动',
        'gics_sector': 'Communication Services', 'sub_industry': '电信',
        'state': 'PULLBACK', 'substate': 'STRONG',
        'last_close': 72.0, 'ma10': 73.5, 'ma20': 73.0, 'ma60': 71.0,
        'ma200': 65.0, 'ma200_slope': 3.0, 'ma10_slope_pct': -0.5,
        'days_in_state': 2, 'days_in_pullback': 2, 'pullback_dd_pct': -1.9,
        'sos': '', 'bear_gate': False, 'is_new_stock': False,
        'sos_setup_recent': '', 'recent_new_high_flag': False,
        'trending_to_pullback_recent': True, 'to_exit_recent': False,
        'date': '2026-05-29',
    })

    # Some POOL stocks for sector count context
    results.append({
        'ticker': '0005.HK', 'name': 'HSBC', 'name_cn': '汇丰',
        'gics_sector': 'Financials', 'sub_industry': '银行',
        'state': 'POOL', 'substate': '',
        'last_close': 60.0, 'ma10': 61.0, 'ma20': 62.0, 'ma60': 63.0,
        'ma200': 60.0, 'ma200_slope': 0.5, 'ma10_slope_pct': 0.2,
        'days_in_state': 100, 'days_in_pullback': 0, 'pullback_dd_pct': None,
        'sos': '', 'bear_gate': False, 'is_new_stock': False,
        'sos_setup_recent': '', 'recent_new_high_flag': False,
        'trending_to_pullback_recent': False, 'to_exit_recent': False,
        'date': '2026-05-29',
    })

    return results


class TestBuildThreeTabs:
    def setup_method(self):
        self.results = _make_mock_results()
        self.three_tabs = build_three_tabs(self.results)

    def test_structure(self):
        assert 'date' in self.three_tabs
        assert 'tab1' in self.three_tabs
        assert 'tab2' in self.three_tabs
        assert 'tab3' in self.three_tabs
        assert self.three_tabs['date'] == '2026-05-29'

    def test_tab1_sos_section(self):
        sos = self.three_tabs['tab1']['sos_section']
        assert len(sos) == 1
        assert sos[0]['ticker'] == '9992.HK'
        assert sos[0]['sos_setup_recent'] == 'SOS-B'
        assert sos[0]['state'] == 'SETUP_OK'

    def test_tab1_pool_section(self):
        pool = self.three_tabs['tab1']['pool_section']
        assert len(pool) == 0  # 9992 already in SOS section, deduped

    def test_tab2_count(self):
        assert len(self.three_tabs['tab2']) == 9  # 7 IT + 1 Industrials + 1 Materials

    def test_tab2_it_all_strong(self):
        it_stocks = [e for e in self.three_tabs['tab2'] if e['gics_sector'] == 'Information Technology']
        assert len(it_stocks) == 7
        assert all(e['display_tier'] == 'STRONG' for e in it_stocks)

    def test_tab2_sector_sort(self):
        """IT (7) should come before Industrials (1) and Materials (1)."""
        sectors = [e['gics_sector'] for e in self.three_tabs['tab2']]
        it_idx = sectors.index('Information Technology')
        ind_idx = sectors.index('Industrials')
        assert it_idx < ind_idx

    def test_tab3_alerts(self):
        alerts = self.three_tabs['tab3']
        assert len(alerts) == 1
        assert alerts[0]['ticker'] == '0941.HK'
        assert alerts[0]['alert_type'] == 'A_PULLBACK'
        assert alerts[0]['pullback_dd_pct'] == -1.9


class TestDisplayTier:
    def test_mid_with_new_high_upgrades_to_strong(self):
        df = pd.DataFrame([{
            'ticker': 'TEST.HK', 'state': 'TRENDING', 'substate': 'MID',
            'gics_sector': 'IT', 'recent_new_high_flag': True,
            'ma10_slope_pct': 5.0, 'days_in_state': 10,
            'name_cn': '测试', 'sub_industry': '', 'ma200_slope': 3.0,
            'ma60': 100, 'ma10': 110, 'last_close': 115,
        }])
        result = build_tab2(df)
        assert result[0]['display_tier'] == 'STRONG'

    def test_early_with_new_high_upgrades_to_strong(self):
        df = pd.DataFrame([{
            'ticker': 'TEST.HK', 'state': 'TRENDING', 'substate': 'EARLY',
            'gics_sector': 'IT', 'recent_new_high_flag': True,
            'ma10_slope_pct': 5.0, 'days_in_state': 10,
            'name_cn': '测试', 'sub_industry': '', 'ma200_slope': 3.0,
            'ma60': 100, 'ma10': 110, 'last_close': 115,
        }])
        result = build_tab2(df)
        assert result[0]['display_tier'] == 'STRONG'

    def test_mid_without_new_high_stays(self):
        df = pd.DataFrame([{
            'ticker': 'TEST.HK', 'state': 'TRENDING', 'substate': 'MID',
            'gics_sector': 'IT', 'recent_new_high_flag': False,
            'ma10_slope_pct': 5.0, 'days_in_state': 10,
            'name_cn': '测试', 'sub_industry': '', 'ma200_slope': 3.0,
            'ma60': 100, 'ma10': 110, 'last_close': 115,
        }])
        result = build_tab2(df)
        assert result[0]['display_tier'] == 'MID'

    def test_new_becomes_strong_new(self):
        df = pd.DataFrame([{
            'ticker': 'TEST.HK', 'state': 'TRENDING', 'substate': 'NEW',
            'gics_sector': 'IT', 'recent_new_high_flag': False,
            'ma10_slope_pct': 5.0, 'days_in_state': 10,
            'name_cn': '测试', 'sub_industry': '', 'ma200_slope': 3.0,
            'ma60': 100, 'ma10': 110, 'last_close': 115,
        }])
        result = build_tab2(df)
        assert result[0]['display_tier'] == 'STRONG NEW'

    def test_strong_stays_strong(self):
        df = pd.DataFrame([{
            'ticker': 'TEST.HK', 'state': 'TRENDING', 'substate': 'STRONG',
            'gics_sector': 'IT', 'recent_new_high_flag': False,
            'ma10_slope_pct': 5.0, 'days_in_state': 10,
            'name_cn': '测试', 'sub_industry': '', 'ma200_slope': 3.0,
            'ma60': 100, 'ma10': 110, 'last_close': 115,
        }])
        result = build_tab2(df)
        assert result[0]['display_tier'] == 'STRONG'


class TestAlerts:
    def test_a_pullback(self):
        df = pd.DataFrame([{
            'ticker': '0941.HK', 'state': 'PULLBACK', 'substate': 'STRONG',
            'gics_sector': 'Communication Services', 'trending_to_pullback_recent': True,
            'days_in_pullback': 2, 'pullback_dd_pct': -1.9,
            'name_cn': '中国移动', 'last_close': 72.0,
        }])
        alerts = build_tab3(df)
        assert len(alerts) == 1
        assert alerts[0]['alert_type'] == 'A_PULLBACK'

    def test_b_exit(self):
        df = pd.DataFrame([{
            'ticker': '0001.HK', 'state': 'EXIT', 'substate': '',
            'gics_sector': 'Industrials', 'to_exit_recent': True,
            'days_in_state': 2,
            'name_cn': '长江和记', 'last_close': 70.0,
        }])
        alerts = build_tab3(df)
        assert len(alerts) == 1
        assert alerts[0]['alert_type'] == 'B_EXIT'

    def test_c_pullback(self):
        df = pd.DataFrame([{
            'ticker': '0002.HK', 'state': 'PULLBACK', 'substate': 'MID',
            'gics_sector': 'Financials', 'trending_to_pullback_recent': False,
            'days_in_pullback': 12, 'pullback_dd_pct': -3.5,
            'name_cn': '中电控股', 'last_close': 65.0,
        }])
        alerts = build_tab3(df)
        assert len(alerts) == 1
        assert alerts[0]['alert_type'] == 'C_PULLBACK'

    def test_recent_pullback_not_c(self):
        """A recent pullback (type A) should NOT also be type C."""
        df = pd.DataFrame([{
            'ticker': '0941.HK', 'state': 'PULLBACK', 'substate': 'STRONG',
            'gics_sector': 'IT', 'trending_to_pullback_recent': True,
            'days_in_pullback': 12, 'pullback_dd_pct': -3.5,
            'name_cn': '测试', 'last_close': 72.0,
        }])
        alerts = build_tab3(df)
        assert len(alerts) == 1
        assert alerts[0]['alert_type'] == 'A_PULLBACK'

    def test_pullback_under_10_not_c(self):
        """PULLBACK < 10 days and not recent should not generate any alert."""
        df = pd.DataFrame([{
            'ticker': '0003.HK', 'state': 'PULLBACK', 'substate': 'MID',
            'gics_sector': 'IT', 'trending_to_pullback_recent': False,
            'days_in_pullback': 5, 'pullback_dd_pct': -1.0,
            'name_cn': '测试', 'last_close': 50.0,
        }])
        alerts = build_tab3(df)
        assert len(alerts) == 0


class TestTab1Dedup:
    def test_sos_stock_not_in_pool(self):
        """Stock in SOS section should be excluded from pool section."""
        df = pd.DataFrame([
            {
                'ticker': '9992.HK', 'state': 'SETUP_OK', 'substate': '',
                'gics_sector': 'Consumer Discretionary',
                'sos_setup_recent': 'SOS-B', 'ma10_slope_pct': 3.0,
                'name_cn': '泡泡玛特', 'sub_industry': '', 'last_close': 185.0,
                'ma10': 180.0, 'ma200_slope': 12.0,
            },
            {
                'ticker': '0006.HK', 'state': 'SETUP_OK', 'substate': '',
                'gics_sector': 'Financials',
                'sos_setup_recent': '', 'ma10_slope_pct': 1.0,
                'name_cn': '电能实业', 'sub_industry': '', 'last_close': 50.0,
                'ma10': 49.0, 'ma200_slope': 5.0,
            },
        ])
        tab1 = build_tab1(df)
        sos_tickers = [e['ticker'] for e in tab1['sos_section']]
        pool_tickers = [e['ticker'] for e in tab1['pool_section']]
        assert '9992.HK' in sos_tickers
        assert '9992.HK' not in pool_tickers
        assert '0006.HK' in pool_tickers
