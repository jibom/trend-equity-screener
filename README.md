# Trend Equity Screener

Wyckoff v5.2 港股趋势扫描器。

## 状态机

```
POOL → TRENDING → PULLBACK → EXIT
         ↑                     |
         └─────────────────────┘  (冷却后可重入)
```

## TRENDING 子状态

| 子状态 | 条件 |
|--------|------|
| STRONG | 周线长4线齐(40>50>60>70) + 周线短4线齐(10>20>30>40) + 容忍度 |
| MID    | 周线长4线齐 + 周线短不齐 |
| EARLY  | 日线MA10>20>50 + MA10 10日斜率>5% + 周线纠缠(离散度<=8%) + 周线不向下 |
| NEW    | 新股(上市<70周) + 日线MA10>20>50 |

## SOS 信号

| 信号 | 条件 |
|------|------|
| SOS-A | 放量(>=1.5x MA30) + 大阳线(实体>=60% + 收盘位>=70%) + 新高 |
| SOS-B | 放量 或 大阳线 + 新高 |

SOS-A/B 是进入 TRENDING 的独立触发器（无需 substate 满足），但受空头闸限制。

## 关键规则

- **空头闸**: WMA40 30日 RoC < -10% 时，SOS 触发被忽略
- **EXIT 冷却**: EXIT 至少停留 5 个交易日
- **PULLBACK 超时**: >=15 日强制 EXIT
- **新股锁定**: 上市 <490 天永远归 NEW

## 项目结构

```
configs/v5_2.json     — 参数配置
src/
  config.py           — 配置加载
  data_provider.py    — 数据获取 (WindFetcher) + 前复权
  indicators.py       — 技术指标计算 (日线MA, 周线MA, 对齐, 离散度等)
  substate.py         — 子状态判定
  sos.py              — SOS 分类
  state_machine.py    — 状态机 + 回测引擎
  scanner.py          — 批量扫描器
tests/                — 单元测试
```

## 使用

```bash
# 单票回测
python src/state_machine.py --stock 0992.HK --asof 2026-05-29 --months 12

# 批量扫描
python src/scanner.py --asof 2026-05-29 --pool configs/pool_hk.csv --workers 4

# 运行测试
pytest tests/
```

## 依赖

- pandas, numpy, pymysql, pytest
