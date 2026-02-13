# 量化选股工具开源版

基于 MA20 均线角度的 A 股量化选股工具，自动扫描全市场股票，识别强势上涨趋势，生成每日 BUY/SELL 信号。支持 RSI、MACD 等多种技术指标，提供完整的回测分析功能。

## 📊 功能特性

### 核心功能
- **MA20 角度选股**: 计算个股 MA20 均线斜率，识别趋势强度
- **RSI 指标**: 相对强弱指数，识别超买超卖区域
- **MACD 指标**: 移动平均收敛 divergence，识别金叉死叉信号
- **每日信号监控**: 自动扫描 A 股，生成 BUY/SELL/HOLD 信号
- **自定义股票池**: 支持自定义关注的股票列表

### 回测功能
- **多策略回测**: 支持 MA20、RSI、MACD 单独或组合回测
- **收益分析**: 总收益率、年化收益率
- **风险指标**: 夏普比率、索提诺比率、最大回撤
- **交易统计**: 胜率、盈亏比、持仓天数
- **批量回测**: 支持多只股票同时回测对比

### 策略组合
- **MA20 + RSI 组合**: 趋势 + 动量指标
- **MA20 + MACD 组合**: 趋势 + 趋势指标
- **完整组合**: MA20 + RSI + MACD 综合信号

## 🚀 快速开始

### 安装依赖

```bash
cd quant_project
pip install -r requirements.txt
```

### 运行选股

```bash
# 默认扫描全部 A 股
python main.py --scan all

# 扫描指定股票池
python main.py --watchlist 600000,000001,300750

# 运行回测
python main.py --backtest --symbol 600000 --start 2024-01-01
```

### Python API - 基础使用

```python
from stock_strategy import StockSelector, TechnicalIndicator
from stock_backtest import Backtester

# 创建选股器
selector = StockSelector()

# 获取综合信号
result = selector.get_signal("600000")
print(f"MA20角度: {result.ma20_angle:.2f}°")
print(f"RSI: {result.rsi:.2f}")
print(f"MACD(DIF): {result.macd:.4f}")
print(f"信号: {result.signal}")
print(f"描述: {result.signal_desc}")
```

### Python API - 技术指标计算

```python
from stock_strategy import calculate_rsi, calculate_macd

# 计算 RSI
rsi = calculate_rsi("600000", period=14)
print(f"RSI(14): {rsi:.2f}")

# 计算 MACD
dif, dea, macd = calculate_macd("600000")
print(f"DIF: {dif:.4f}")
print(f"DEA: {dea:.4f}")
print(f"MACD: {macd:.4f}")
```

### Python API - 回测分析

```python
from stock_backtest import Backtester, quick_backtest

# 快速回测
result = quick_backtest("600000", "2024-01-01", "2025-01-01")
print(f"总收益率: {result.total_return:+.2f}%")
print(f"年化收益率: {result.annual_return:+.2f}%")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
print(f"胜率: {result.win_rate:.1f}%")
print(f"最大回撤: {result.max_drawdown_pct:.2f}%")

# 多策略组合回测
from stock_backtest import run_multi_strategy_backtest

result = run_multi_strategy_backtest(
    symbol="600000",
    start_date="2024-01-01",
    end_date="2025-01-01",
    use_ma20=True,
    use_rsi=True,
    use_macd=True
)
```

### Python API - 策略组合演示

```python
# 运行多策略演示
python multi_strategy_demo.py
```

## 📁 项目结构

```
quant_project/
├── README.md              # 本文档
├── main.py                # 主程序入口
├── requirements.txt       # Python 依赖
├── .gitignore            # Git 忽略配置
├── stock_strategy.py     # MA20 + RSI + MACD 选股策略
├── stock_backtest.py     # 回测模块（增强版）
├── multi_strategy_demo.py # 策略组合演示
├── data_manager.py       # 数据管理
├── ml_selector.py        # 🤖 机器学习选股 (付费版)
├── scoring_system.py     # 📊 多维度评分系统 (付费版)
├── smart_stock_picker.py # 🎯 智能选股工具 (付费版)
└── strategies/           # 策略基类（保留）
    └── strategy_base.py
```

## 📈 策略说明

### MA20 角度选股策略

**核心逻辑**:
- **MA20 角度 > 3°**: 强势上涨趋势 → BUY 信号
- **MA20 角度 < 0°**: 下跌趋势 → SELL 信号  
- **MA20 角度 0-3°**: 震荡整理 → HOLD

### RSI 指标

**参数**:
- 周期: 14 (默认)
- 超买阈值: 70
- 超卖阈值: 30

**信号**:
- **RSI ≥ 70**: 超买区域，可能回调 → 建议减仓
- **RSI ≤ 30**: 超卖区域，可能反弹 → 建议关注
- **RSI 30-70**: 中性区域 → 观望

### MACD 指标

**参数**:
- 快速 EMA: 12
- 慢速 EMA: 26
- Signal 线: 9

**信号**:
- **金叉 (GOLD_CROSS)**: DIF 上穿 DEA → 买入信号
- **死叉 (DEAD_CROSS)**: DIF 下穿 DEA → 卖出信号
- **中性**: 无交叉 → 观望

### 组合策略信号

| MA20角度 | RSI | MACD | 综合信号 |
|----------|-----|------|----------|
| > 3° | ≤ 30 | 金叉 | 🟢 强买入 |
| > 3° | 中性 | 中性 | 🟢 买入 |
| 0-3° | ≤ 30 | 金叉 | 🟡 关注 |
| 0-3° | 中性 | 中性 | 🟡 观望 |
| < 0° | ≥ 70 | 死叉 | 🔴 强卖出 |
| < 0° | 任意 | 任意 | 🔴 卖出 |

### 选股规则

1. 每日收盘后计算所有股票的 MA20 角度、RSI、MACD
2. 按 MA20 角度降序排列，优先选择角度最大的股票
3. 结合 RSI 超卖 + MACD 金叉 信号确认
4. 排除 ST 股票、新股（上市不满 60 日）

## 📊 回测参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 初始资金 | 100,000 | 回测起始资金 |
| 止损比例 | 5% | 跌破买入价 5% 止损 |
| 止盈比例 | 15% | 涨破买入价 15% 止盈 |
| 持仓周期 | 10日 | 超过10日强制平仓 |
| 仓位比例 | 80% | 单次买入仓位 |
| 手续费率 | 0.03% | 买卖双向 |
| 滑点率 | 0.1% | 买卖滑点 |

### 回测指标

| 指标 | 说明 | 评级标准 |
|------|------|----------|
| **总收益率** | 期末资金 / 期初资金 - 1 | 🟢 > 20% |
| **年化收益率** | 复合年化收益 | 🟢 > 15% |
| **夏普比率** | 风险调整后收益 | 🟢 ≥ 1.0, 🟡 ≥ 0, 🔴 < 0 |
| **索提诺比率** | 下行风险调整收益 | 🟢 > 1.0 |
| **胜率** | 盈利交易占比 | 🟢 ≥ 50% |
| **盈亏比** | 平均盈利 / 平均亏损 | 🟢 > 1.5 |
| **最大回撤** | 账户净值从峰值最大跌幅 | 🟢 < 10%, 🟡 < 20% |
| **波动率** | 收益率标准差 | 越低越稳定 |

## 🎯 付费版专属功能

> **新增机器学习选股和综合评分系统，为付费版提供差异化功能**

### 🤖 机器学习选股 (ml_selector.py)

基于 sklearn 的机器学习模型，预测明日涨跌概率。

**支持的模型**:
- `LogisticRegression`: 逻辑回归，适合二分类
- `RandomForest`: 随机森林，提供特征重要性

**特征工程**:
| 特征 | 说明 |
|------|------|
| `ma20_angle` | MA20 均线角度 |
| `rsi` | RSI 指标 |
| `macd_diff` | MACD 差值 (DIF) |
| `volume_change` | 成交量变化率 |
| `price_momentum` | 价格动量 (5日) |
| `volatility` | 波动率 |
| `rsi_position` | RSI 位置 (0-1 归一化) |
| `macd_histogram` | MACD 柱状图 |

**使用示例**:

```python
from ml_selector import MLSelector

# 创建选股器
selector = MLSelector(model_type='random_forest')

# 加载数据并训练
df = load_stock_data("600000")
result = selector.train(df, verbose=True)

# 预测
prediction = selector.predict(df)
print(f"信号: {prediction['signal']}")
print(f"上涨概率: {prediction['up_probability']:.2%}")
print(f"置信度: {prediction['confidence']:.2%}")

# 获取特征重要性
weights = selector.get_feature_importance()
```

### 📊 综合评分系统 (scoring_system.py)

多维度评分 0-100 分，综合评估技术面。

**评分维度**:

| 维度 | 权重 | 说明 |
|------|------|------|
| 趋势强度 | 25% | MA角度、均线位置、短期趋势 |
| 动量 | 25% | 各周期涨幅、成交量确认 |
| 波动率 | 15% | 波动水平、稳定性 (越低越好) |
| RSI位置 | 20% | RSI水平及趋势 |
| MACD状态 | 15% | 柱状图方向、金叉死叉 |

**信号分级**:

| 评分 | 信号 | 操作建议 |
|------|------|----------|
| 80+ | 🟢 强力买入 | 强烈看涨，积极参与 |
| 60-80 | 🟢 买入 | 温和看涨，可以关注 |
| 40-60 | 🟡 持有 | 建议观望 |
| <40 | 🔴 卖出 | 建议回避 |

**使用示例**:

```python
from scoring_system import ScoringSystem, print_score_result

# 创建评分器
scoring = ScoringSystem()

# 计算评分
result = scoring.calculate(df)

# 打印结果
print_score_result(result, "600000")

# 批量评分
results = scoring.batch_score({
    "600000": df1,
    "000001": df2,
})

# 获取TOP股票
top_stocks = scoring.get_top_picks(stock_data, top_n=10, min_score=50)
```

### 🎯 智能选股 (smart_stock_picker.py)

结合基本面筛选 + 技术面评分 + ML预测的智能选股工具。

**基本面筛选**:
- PE: 0-50
- PB: 0-5
- 市值: 50-5000亿
- 排除 ST 股票
- 排除新股

**使用示例**:

```python
from smart_stock_picker import SmartStockPicker, quick_scan

# 快速扫描
picks = quick_scan(
    symbols=['600519', '600036', '000858'],
    use_ml=True,      # 启用ML辅助
    min_score=60      # 最低评分
)

# 打印结果
picker.print_results(picks)

# 导出结果
picker.export_results(picks, '精选股票', format='csv')

# 自定义选股器
picker = SmartStockPicker({
    'pe_max': 30,
    'pb_max': 3,
    'market_cap_min': 100,
})

# 启用ML
picker.enable_ml('random_forest')

# 扫描市场
picks = picker.scan_market(all_symbols)
```

### 📈 功能对比

| 功能 | 开源版 | 付费版 |
|------|--------|--------|
| MA20 角度选股 | ✅ | ✅ |
| RSI/MACD 指标 | ✅ | ✅ |
| 基础回测 | ✅ | ✅ |
| 多维度评分 | ❌ | ✅ |
| ML 涨跌预测 | ❌ | ✅ |
| 基本面筛选 | ❌ | ✅ |
| 特征重要性分析 | ❌ | ✅ |
| 智能选股导出 | ❌ | ✅ |

## 📦 依赖说明

付费版额外依赖:

| 包名 | 版本 | 用途 |
|------|------|------|
| scikit-learn | >=1.3.0 | 机器学习框架 |
| scipy | >=1.11.0 | 科学计算 |

```bash
pip install scikit-learn scipy
```

## ⚠️ 风险提示

1. **历史表现不代表未来收益**
2. **MA20 角度策略适用于趋势行情，震荡市可能失效**
3. **RSI 和 MACD 为辅助指标，建议结合使用**
4. **建议结合成交量、基本面等综合判断**
5. **实盘交易前请进行充分的回测和模拟盘验证**

## 📝 更新日志

### v1.2.0 (2026-02-13) - 付费版核心功能
- ✅ **机器学习选股模块** `ml_selector.py`
  - LogisticRegression 和 RandomForest 模型
  - 8个技术指标特征工程
  - 预测明日涨跌概率和置信度
  - 支持模型保存和加载
  - 特征重要性分析

- ✅ **评分系统模块** `scoring_system.py`
  - 5维度综合评分 (趋势/动量/波动率/RSI/MACD)
  - 权重可配置
  - 明确的信号分级 (强力买入/买入/持有/卖出)
  - 详细的操作建议

- ✅ **智能选股脚本** `smart_stock_picker.py`
  - 基本面筛选 (PE/PB/市值)
  - 技术面评分排名
  - ML辅助预测集成
  - 批量扫描和结果导出

- ✅ **文档更新**: README.md 添加付费版功能说明

### v1.1.0 (2026-02-13)
- ✅ **新增 RSI 指标**: 添加 RSI 计算函数和超买超卖信号
- ✅ **新增 MACD 指标**: 添加 MACD 金叉死叉检测
- ✅ **增强回测报告**: 添加夏普比率、索提诺比率
- ✅ **增强交易统计**: 添加胜率、盈亏比、连胜连负统计
- ✅ **策略组合演示**: 创建 multi_strategy_demo.py
- ✅ **数据类更新**: StockSignal 新增 rsi、rsi_signal、macd、macd_signal 字段

### v1.0.0 (2026-02-12)
- 初始开源版本
- MA20 角度选股功能
- 基础回测模块
- 飞书信号推送（可选）

## 🎯 未来规划

付费版 v1.2.0 已发布！

**v1.2.0 (2026-02-13) - 付费版核心功能**:
- ✅ **机器学习选股**: MLSelector 支持 LogisticRegression/RandomForest
- ✅ **多维度评分系统**: 5维度综合评分 0-100 分
- ✅ **智能选股工具**: 基本面+技术面+ML综合筛选
- ✅ **特征重要性分析**: 可解释的模型权重
- ✅ **批量选股**: 支持多线程并行扫描

**即将推出**:
- ⏳ 布林带、KDJ、OBV 等更多技术指标
- ⏳ 组合优化器: 自动寻找最优参数
- ⏳ 实时监控: 盘中实时信号推送
- ⏳ Web 界面: 可视化操作界面

## 📧 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。

---

*本项目仅供学习和研究使用，不构成任何投资建议。*
