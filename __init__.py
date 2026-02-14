# quant_project/__init__.py
"""
quant_project - 量化投资策略系统

基于MA20均线角度的智能选股系统
支持技术指标分析、机器学习预测、回测等功能
"""

__version__ = "1.0.0"
__author__ = "quant_team"
__description__ = "量化投资策略系统"

# 导出主要模块
from .stock_strategy import StockSelector, TechnicalIndicator
from .backtest.backtest_engine import run_backtest
from .ml_selector import MLSelector

__all__ = [
    "StockSelector",
    "TechnicalIndicator",
    "run_backtest",
    "MLSelector",
]
