# quant_project/strategies/strategy_base.py
"""
趋势跟踪策略基类
基于双均线交叉 + ATR 止损
"""

import backtrader as bt
import pandas as pd


class TrendStrategyBase(bt.Strategy):
    """
    趋势跟踪策略基类
    
    参数:
    - fast_period: 快速均线周期
    - slow_period: 慢速均线周期
    - atr_period: ATR 周期
    - atr_multiplier: ATR 止损倍数
    """
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
    )
    
    def __init__(self):
        # 计算均线
        self.sma_fast = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period
        )
        self.sma_slow = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period
        )
        
        # 计算ATR用于止损
        self.atr = bt.indicators.ATR(
            self.data, period=self.params.atr_period
        )
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(
            self.sma_fast, self.sma_slow
        )
        
        # 订单状态追踪
        self.order = None
        self.buy_price = None
        self.buy_size = None
        
        # 日志
        self.trade_log = []
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.buy_price = order.executed.price
                self.buy_size = order.executed.size
                self.log(f'买入: 价格={order.executed.price:.2f}, 数量={order.executed.size}')
            else:
                self.log(f'卖出: 价格={order.executed.price:.2f}, 数量={order.executed.size}')
                
        self.order = None
    
    def log(self, txt, dt=None):
        """日志输出"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')
    
    def next(self):
        # 检查订单状态
        if self.order:
            return
        
        # 获取当前数据
        close = self.data.close[0]
        atr = self.atr[0]
        
        # 金叉买入
        if self.crossover > 0:
            # 计算仓位 (1ATR止损)
            risk_per_unit = self.params.atr_multiplier * atr
            if risk_per_unit > 0:
                # 假设账户2%风险
                target_risk = 0.02 * self.broker.getvalue()
                size = int(target_risk / risk_per_unit)
                size = min(size, 100)  # 限制最大仓位
                
                if size > 0:
                    self.order = self.buy(size=size)
                    self.buy_price = close
                    self.log(f'信号买入: 价格={close:.2f}, 止损={close - risk_per_unit:.2f}')
        
        # 死叉卖出
        elif self.crossover < 0 and self.position:
            self.order = self.sell(size=self.position.size)
            self.log(f'信号卖出: 价格={close:.2f}')
        
        # 止损逻辑
        elif self.position and close < self.buy_price - self.params.atr_multiplier * atr:
            self.order = self.sell(size=self.position.size)
            self.log(f'止损卖出: 价格={close:.2f}')
    
    def stop(self):
        """策略结束时打印结果"""
        self.log(f'策略结束, 最终价值={self.broker.getvalue():.2f}')


class TrendStrategyOptimized(TrendStrategyBase):
    """
    优化版趋势策略
    - 加入趋势过滤器（只在趋势明确时交易）
    - 加入移动止盈
    """
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('trail_stop_atr', 3.0),  # 移动止盈ATR倍数
        ('trend_filter_period', 50),  # 趋势过滤周期
    )
    
    def __init__(self):
        super().__init__()
        
        # 趋势过滤器：长期均线方向
        self.sma_trend = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.trend_filter_period
        )
        
        # 价格相对均线位置
        self.price_vs_trend = self.data.close - self.sma_trend
    
    def next(self):
        if self.order:
            return
        
        close = self.data.close[0]
        atr = self.atr[0]
        
        # 趋势过滤：价格在长期均线上方才做多
        trend_up = close > self.sma_trend[0]
        
        # 金叉 + 趋势确认
        if self.crossover > 0 and trend_up and not self.position:
            risk_per_unit = self.params.atr_multiplier * atr
            if risk_per_unit > 0:
                target_risk = 0.02 * self.broker.getvalue()
                size = int(target_risk / risk_per_unit)
                
                self.order = self.buy(size=size)
                self.log(f'趋势确认买入: 价格={close:.2f}')
        
        # 死叉平仓
        elif self.crossover < 0 and self.position:
            self.order = self.sell(size=self.position.size)
            self.log(f'信号卖出: 价格={close:.2f}')
        
        # 止损
        elif self.position and close < self.buy_price - self.params.atr_multiplier * atr:
            self.order = self.sell(size=self.position.size)
            self.log(f'止损: 价格={close:.2f}')
        
        # 移动止盈
        elif self.position and close > self.buy_price + self.params.trail_stop_atr * atr:
            # 止盈后跟踪止损
            new_stop = close - self.params.atr_multiplier * atr
            if new_stop > self.buy_price:
                self.order = self.sell(size=self.position.size)
                self.log(f'止盈: 价格={close:.2f}')
