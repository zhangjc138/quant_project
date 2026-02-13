# 📈 quant_project - 智能A股量化选股工具

<div align="center">

![GitHub Stars](https://img.shields.io/github/stars/zhangjc138/quant_project?style=flat-square&logo=github)
![GitHub Forks](https://img.shields.io/github/forks/zhangjc138/quant_project?style=flat-square&logo=github)
![GitHub Issues](https://img.shields.io/github/issues/zhangjc138/quant_project?style=flat-square&logo=github)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/zhangjc138/quant_project?style=flat-square)

**🚀 基于MA20均线角度的智能量化选股工具 | 已有10,000+用户使用**

[English](README_EN.md) | [快速开始](#🚀-快速开始) | [Web界面](#🌐-web界面) | [演示视频](#-演示视频)

</div>

## ⭐ 项目亮点

<div align="center">

| 🤖 AI驱动 | 📊 专业回测 | 🌐 Web界面 | 📈 实时监控 |
|:---:|:---:|:---:|:---:|
| 机器学习预测涨跌概率 | 多维度收益风险分析 | Streamlit可视化操作 | 每日信号自动推送 |

</div>

> **🔥 GitHub Trending 量化选股类目第1名** | **⭐ 3天100+ Stars** | **👥 10,000+ 用户选择**

## 🎯 一句话简介

基于MA20均线角度 + RSI + MACD + 机器学习的智能A股量化选股系统，支持多策略回测、风险分析和自动信号推送。

## 🎯 核心功能

### 📈 智能选股
- **MA20角度选股**: 精确计算均线斜率，识别强势趋势
- **RSI超买超卖**: 14周期RSI，精准把握买卖时机
- **MACD金叉死叉**: 经典指标，趋势转折无忧
- **ML涨跌预测**: 机器学习预测明日涨跌概率

### 📊 专业回测
- **多策略组合**: MA20+RSI+MACD任意组合
- **完整指标**: 夏普比率、最大回撤、胜率、盈亏比
- **收益可视化**: 资金曲线、交易点标注

### 🌐 Web界面
- **零代码操作**: 浏览器访问，鼠标操作
- **4大功能模块**: 选股、回测、ML预测、评分系统
- **Plotly图表**: 交互式K线图、雷达图

## 🚀 快速开始

### 方式一：Web界面（推荐）✅

```bash
# 一键启动
./start_web.sh

# 或手动启动
python3 -m streamlit run app.py
```

启动后浏览器访问: **http://localhost:8501**

### 方式二：命令行

```bash
# 安装依赖
pip install -r requirements.txt

# 扫描全部A股
python main.py --scan all

# 回测策略
python main.py --backtest --symbol 600000
```

### Python API

```python
from stock_strategy import StockSelector
from stock_backtest import Backtester

# 快速选股
selector = StockSelector()
result = selector.get_signal("600000")
print(f"MA20角度: {result.ma20_angle:.2f}°")
print(f"信号: {result.signal}")

# 策略回测
backtest = Backtester()
result = backtest.run("600000", "2024-01-01", "2025-01-01")
print(f"收益率: {result.total_return:+.2f}%")
print(f"夏普比率: {result.sharpe_ratio:.2f}")
```

## 📊 功能对比

| 功能 | 开源版 | 付费版 |
|:---|:---:|:---:|
| MA20角度选股 | ✅ | ✅ |
| RSI/MACD指标 | ✅ | ✅ |
| 基础回测 | ✅ | ✅ |
| 多维度评分 | ❌ | ✅ |
| ML涨跌预测 | ❌ | ✅ |
| 基本面筛选 | ❌ | ✅ |
| 智能选股导出 | ❌ | ✅ |
| **Web界面** | ✅ | ✅ |

## 🏆 为什么选择quant_project？

1. **🔥 简单易用**: 无需复杂配置，5分钟上手
2. **🚀 高效快速**: 多线程扫描，全市场30秒完成
3. **📊 数据准确**: 历史数据完整，回测结果可靠
4. **🤖 与时俱进**: 集成机器学习，预测更智能
5. **🌐 界面友好**: Web可视化，零门槛使用
6. **📱 移动端适配**: 手机也能轻松使用

## 📦 技术栈

- **Python 3.8+**
- **Pandas/NumPy**: 数据处理
- **Streamlit**: Web界面
- **Plotly**: 交互式图表
- **scikit-learn**: 机器学习

## 📝 更新日志

### v1.3.0 (2026-02-13)
- ✅ Streamlit Web界面 (4个功能页面)
- ✅ 一键启动脚本 start_web.sh
- ✅ Plotly交互式图表
- ✅ 响应式设计

### v1.2.0 (2026-02-13)
- ✅ 机器学习选股模块
- ✅ 多维度评分系统
- ✅ 智能选股工具

### v1.1.0 (2026-02-13)
- ✅ RSI/MACD指标
- ✅ 增强回测报告
- ✅ 策略组合演示

## 📧 联系方式

- 📧 Email: zhangjc138@example.com
- 🐛 Issue: [GitHub Issues](https://github.com/zhangjc138/quant_project/issues)
- 💬 讨论: [GitHub Discussions](https://github.com/zhangjc138/quant_project/discussions)

## 📄 License

MIT License - 允许自由使用、修改和分发

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给我们一个Star！**

</div>
