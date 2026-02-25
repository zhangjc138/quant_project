# quant_project/strategies/__init__.py
"""
策略模块

包含各种交易策略
"""

from .strategy_base import TrendStrategyBase, TrendStrategyOptimized

__all__ = ["TrendStrategyBase", "TrendStrategyOptimized"]
