# quant_project/backtest/__init__.py
"""
回测模块

包含:
- 回测引擎
- 绩效分析
"""

from .backtest_engine import run_backtest, analyze_results

__all__ = ["run_backtest", "analyze_results"]
