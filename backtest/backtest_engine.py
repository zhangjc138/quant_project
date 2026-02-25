# quant_project/backtest/backtest_engine.py
"""
回测引擎
使用 backtrader 进行策略回测
"""

import backtrader as bt
import pandas as pd
from datetime import datetime
import sys
sys.path.append('..')
from strategies.strategy_base import TrendStrategyBase, TrendStrategyOptimized


def run_backtest(
    data_df: pd.DataFrame,
    strategy_class=TrendStrategyBase,
    initial_cash: float = 100000,
    **strategy_params
):
    """
    运行回测
    
    Args:
        data_df: 行情数据 DataFrame
        strategy_class: 策略类
        initial_cash: 初始资金
        **strategy_params: 策略参数
    """
    # 创建 cerebro 引擎
    cerebro = bt.Cerebro()
    
    # 添加策略
    cerebro.addstrategy(strategy_class, **strategy_params)
    
    # 准备数据
    data = bt.feeds.PandasData(
        dataname=data_df,
        datetime='date',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest='hold',
        fromdate=datetime(2000, 1, 1),  # 足够早的日期
        todate=datetime(2099, 12, 31),
    )
    
    cerebro.adddata(data)
    
    # 设置经纪商
    cerebro.broker.setcash(initial_cash)
    
    # 设置交易费用 (期货 万分之二)
    cerebro.broker.setcommission(commission=0.0002)
    
    # 设置滑点
    cerebro.broker.set_slippage_perc(0.0005)
    
    # 打印初始资金
    print(f'初始资金: {initial_cash:,.2f}')
    
    # 运行回测
    results = cerebro.run()
    
    # 打印最终资金
    final_value = cerebro.broker.getvalue()
    print(f'最终资金: {final_value:,.2f}')
    print(f'收益率: {(final_value - initial_cash) / initial_cash * 100:.2f}%')
    
    # 返回结果
    return results[0], cerebro


def analyze_results(results, cerebro):
    """分析回测结果"""
    # 获取绘图
    cerebro.plot(
        style='candlestick',
        barup='red',
        bardown='green',
        volume=True,
        figsize=(14, 10)
    )


def print_trade_statistics(results):
    """打印交易统计"""
    if hasattr(results, 'trade_log'):
        print("\n=== 交易记录 ===")
        for trade in results.trade_log:
            print(trade)


if __name__ == "__main__":
    # 测试回测
    from data_manager import fetch_futures_daily
    
    # 获取数据
    print("获取数据...")
    df = fetch_futures_daily("IF2006", "20200101", "20231231")
    
    if df is not None:
        print(f"数据量: {len(df)} 条")
        
        # 运行回测
        print("\n运行回测...")
        results, cerebro = run_backtest(
            df,
            strategy_class=TrendStrategyOptimized,
            fast_period=10,
            slow_period=30,
            atr_period=14,
            atr_multiplier=2.0
        )
