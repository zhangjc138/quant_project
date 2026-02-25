# V-VAD 策略 (Volatility-Volume Anomaly Detector)

基于**流体物理理论**的成交量-波动率异常检测策略。

## 核心原理

根据流体物理理论，价格波动与成交量的关系：
- **成交量放大 → 波动率应该下降**
- 如果成交量放大但波动率上升 → 强趋势信号
- 如果成交量放大但波动率下降 → 可能假突破

## 信号规则

| 信号 | 条件 |
|------|------|
| 买入 | 放量 > 1.5倍 AND 波动率上升 > 1.3倍 |
| 卖出 | 放量 > 1.5倍 AND 波动率下降 < 0.8倍 |

## 回测表现 (2024-01 ~ 2026-02)

| 股票类型 | 平均收益 | 胜率 |
|----------|----------|------|
| 大盘股 | +26% | 70% |
| 百亿市值 | +33% | 80% |
| 小盘股 | +55% | 80% |

## 使用方法

```bash
# 克隆仓库
git clone https://github.com/your-repo/vvad-strategy.git
cd vvad-strategy

# 安装依赖
pip install baostock pandas numpy

# 回测
python fluid_strategies_backtest.py --stock 600519 --compare
```

## 文件说明

- `vvad.py` - V-VAD策略核心逻辑
- `fluid_strategies_backtest.py` - 综合回测框架

## 监控

已配置cron每日20:00扫描沪深300股票池，有信号自动推送。
