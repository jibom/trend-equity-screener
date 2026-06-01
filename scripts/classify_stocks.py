"""Classify stocks into investment themes using Anthropic Claude.

Reads scan results, groups stocks by sector/trend into themes,
and outputs structured JSON for the frontend.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PUBLIC_DATA = os.path.join(PROJECT_DIR, 'public', 'data')


def classify_by_sector(results: list[dict]) -> dict:
    """Rule-based theme classification by GICS sector and sub-industry."""
    themes = {}

    sector_themes = {
        'Information Technology': '科技',
        'Financials': '金融',
        'Health Care': '医疗健康',
        'Consumer Discretionary': '可选消费',
        'Consumer Staples': '必选消费',
        'Energy': '能源',
        'Industrials': '工业',
        'Materials': '材料',
        'Real Estate': '房地产',
        'Communication Services': '通信服务',
        'Utilities': '公用事业',
    }

    sub_industry_keywords = {
        '半导体': ['Semiconductor', 'Chip', 'Integrated Circuit'],
        'AI/算力': ['Artificial Intelligence', 'Cloud Computing', 'Data Center'],
        '软件': ['Software', 'SaaS', 'Application Software'],
        '银行': ['Bank', 'Banking'],
        '保险': ['Insurance'],
        '新能源': ['Solar', 'Wind', 'Renewable', 'Clean Energy'],
        '电动车': ['Electric Vehicle', 'EV', 'Battery'],
        '医药': ['Pharmaceutical', 'Biotech', 'Drug'],
        '互联网': ['Internet', 'E-commerce', 'Social Media'],
        '电信': ['Telecom', 'Telecommunication', 'Wireless'],
    }

    for item in results:
        sector = item.get('gics_sector', '')
        sub = item.get('sub_industry', '')
        theme_name = sector_themes.get(sector, '其他')

        # Refine by sub-industry keywords
        for cn_theme, keywords in sub_industry_keywords.items():
            if any(kw.lower() in (sub or '').lower() for kw in keywords):
                theme_name = cn_theme
                break

        if theme_name not in themes:
            themes[theme_name] = {
                'name': theme_name,
                'stocks': [],
                'sector': sector,
            }
        themes[theme_name]['stocks'].append({
            'ticker': item['ticker'],
            'name_cn': item.get('name_cn', ''),
            'state': item.get('state', item.get('pool_status', '')),
            'substate': item.get('substate', ''),
            'last_close': item.get('last_close'),
        })

    return themes


def classify_us_by_sector(results: list[dict]) -> dict:
    """Rule-based theme classification for US stocks."""
    themes = {}

    for item in results:
        pool_status = item.get('pool_status', '')
        if pool_status in ('BROKEN', 'RECOVERING'):
            continue

        sector = ''
        ticker = item.get('ticker', '').replace('.US', '')

        # Classify by ticker patterns
        semis = {'NVDA', 'AMD', 'AVGO', 'INTC', 'MU', 'QCOM', 'TXN', 'AMAT', 'LRCX',
                 'KLAC', 'MRVL', 'ON', 'MCHP', 'NXPI', 'MPWR', 'ADI', 'TER', 'ASML',
                 'ARM', 'TSM', 'SMCI', 'SMH', 'SOXX', 'SOXL', 'TECL', 'NVDL'}
        ai_datacenter = {'VRT', 'FSLR', 'FLNC', 'SMCI', 'DELL', 'GE', 'GEV', 'MOD'}
        financials = {'GS', 'MS', 'C', 'BK', 'PNC', 'STT', 'TFC', 'BNY', 'RF', 'USB'}
        energy = {'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'FANG', 'OKE', 'KMI', 'MPC', 'VLO'}

        if ticker in semis:
            theme_name = '半导体'
        elif ticker in ai_datacenter:
            theme_name = 'AI/算力基建'
        elif ticker in financials:
            theme_name = '金融'
        elif ticker in energy:
            theme_name = '能源'
        else:
            theme_name = '其他'

        if theme_name not in themes:
            themes[theme_name] = {'name': theme_name, 'stocks': []}
        themes[theme_name]['stocks'].append({
            'ticker': item['ticker'],
            'pool_status': pool_status,
            'last_close': item.get('last_close'),
        })

    return themes


def main():
    parser = argparse.ArgumentParser(description='Classify stocks into themes')
    parser.add_argument('--market', choices=['hk', 'us', 'all'], default='all')
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PUBLIC_DATA, exist_ok=True)

    if args.market in ('hk', 'all'):
        hk_pools_path = os.path.join(PUBLIC_DATA, 'pools_hk.json')
        if os.path.isfile(hk_pools_path):
            with open(hk_pools_path, encoding='utf-8') as f:
                hk_results = json.load(f)
            themes = classify_by_sector(hk_results)

            # Save to both data/ and public/data/
            for out_dir in [DATA_DIR, PUBLIC_DATA]:
                themes_path = os.path.join(out_dir, 'themes_hk.json')
                with open(themes_path, 'w', encoding='utf-8') as f:
                    json.dump(themes, f, ensure_ascii=False, indent=2)
            print(f"HK: {len(themes)} themes from {len(hk_results)} stocks")

    if args.market in ('us', 'all'):
        us_pools_path = os.path.join(DATA_DIR, 'pools_us.json')
        if os.path.isfile(us_pools_path):
            with open(us_pools_path, encoding='utf-8') as f:
                us_results = json.load(f)
            themes = classify_us_by_sector(us_results)

            for out_dir in [DATA_DIR, PUBLIC_DATA]:
                themes_path = os.path.join(out_dir, 'themes_us.json')
                with open(themes_path, 'w', encoding='utf-8') as f:
                    json.dump(themes, f, ensure_ascii=False, indent=2)
            print(f"US: {len(themes)} themes from {len(us_results)} stocks")

    # Also generate ref files (stock reference data)
    if args.market in ('hk', 'all'):
        hk_pools_path = os.path.join(PUBLIC_DATA, 'pools_hk.json')
        if os.path.isfile(hk_pools_path):
            with open(hk_pools_path, encoding='utf-8') as f:
                hk_results = json.load(f)
            ref = {}
            for item in hk_results:
                ref[item['ticker']] = {
                    'name': item.get('name_cn', ''),
                    'sector': item.get('gics_sector', ''),
                }
            for out_dir in [DATA_DIR, PUBLIC_DATA]:
                with open(os.path.join(out_dir, 'ref_hk.json'), 'w', encoding='utf-8') as f:
                    json.dump(ref, f, ensure_ascii=False, indent=2)

    if args.market in ('us', 'all'):
        us_pools_path = os.path.join(DATA_DIR, 'pools_us.json')
        if os.path.isfile(us_pools_path):
            with open(us_pools_path, encoding='utf-8') as f:
                us_results = json.load(f)
            ref = {}
            for item in us_results:
                ref[item['ticker']] = {
                    'name': item.get('ticker', '').replace('.US', ''),
                    'sector': '',
                }
            for out_dir in [DATA_DIR, PUBLIC_DATA]:
                with open(os.path.join(out_dir, 'ref_us.json'), 'w', encoding='utf-8') as f:
                    json.dump(ref, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
