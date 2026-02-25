#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
期货策略回测分析
- 波动率自适应趋势跟踪策略
- 日内交易策略
- 多周期缠论策略
- RB、CU等品种回测
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.strategy_base import TrendStrategyBase, TrendStrategyOptimized


class VolatilityAdaptiveTrendStrategy(bt.Strategy):
    """
    波动率自适应趋势跟踪策略
    
    特点：
    - 根据市场波动率动态调整仓位和止损
    - 高波动率时减小仓位、扩大止损
    - 低波动率时增大仓位、缩小止损
    """
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
        ('volatility_lookback', 20),  # 波动率回看周期
        ('atr_period', 14),
        ('base_atr_multiplier', 2.0),  # 基础止损ATR倍数
        ('risk_per_trade', 0.02),  # 每笔交易风险比例
    )
    
    def __init__(self):
        # 均线
        self.sma_fast = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        self.sma_slow = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        
        # ATR
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        # 波动率 (收益率标准差)
        self.returns = bt.indicators.RateOfChange(self.data.close, period=1)
        self.volatility = bt.indicators.StandardDeviation(self.returns, period=self.params.volatility_lookback)
        
        # 波动率百分位 (用于判断当前波动率位置)
        self.vol_percentile = bt.indicators.Percentile(self.volatility, period=self.params.volatility_lookback)
        
        # 交叉信号
        self.crossover = bt.indicators.CrossOver(self.sma_fast, self.sma_slow)
        
        self.order = None
        self.trade_log = []
        
    def next(self):
        if self.order:
            return
        
        close = self.data.close[0]
        atr = self.atr[0]
        vol = self.volatility[0]
        
        # 波动率自适应调整
        # 波动率越高，止损越宽，仓位越小
        vol_factor = min(vol / self.volatility[0] if self.volatility[0] > 0 else 1, 3.0)
        adjusted_atr_multiplier = self.params.base_atr_multiplier * vol_factor
        
        # 计算仓位
        risk_per_unit = adjusted_atr_multiplier * atr
        if risk_per_unit > 0:
            target_risk = self.params.risk_per_trade * self.broker.getvalue()
            size = int(target_risk / risk_per_unit)
            size = max(1, size)  # 最小1手
        
        # 金叉买入
        if self.crossover > 0 and not self.position:
            if risk_per_unit > 0:
                self.order = self.buy(size=size)
                self.log(f'VolAdaptive BUY: {close:.2f}, vol_factor={vol_factor:.2f}')
                self.trade_log.append({
                    'date': self.datas[0].datetime.date(0),
                    'type': 'BUY',
                    'price': close,
                    'vol_factor': vol_factor
                })
        
        # 死叉卖出
        elif self.crossover < 0 and self.position:
            self.order = self.sell(size=self.position.size)
            self.log(f'Crossover SELL: {close:.2f}')
            self.trade_log.append({
                'date': self.datas[0].datetime.date(0),
                'type': 'SELL',
                'price': close
            })
        
        # 动态止损
        elif self.position:
            stop_price = self.buy_price - adjusted_atr_multiplier * atr
            if close < stop_price:
                self.order = self.sell(size=self.position.size)
                self.log(f'StopLoss: {close:.2f}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')


class IntradayStrategy(bt.Strategy):
    """
    日内交易策略
    
    特点：
    - 当日开仓当日平仓
    - 开盘区间突破入场
    - 收盘前强制平仓
    """
    
    params = (
        ('breakout_range', 0.002),  # 突破区间 (0.2%)
        ('close_time', '14:45'),  # 收盘平仓时间
        ('stop_loss_atr', 2.0),  # 止损ATR倍数
        ('profit_target_atr', 3.0),  # 止盈ATR倍数
    )
    
    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=14)
        self.order = None
        self.intraday_trades = []
        
    def next(self):
        dt = self.datas[0].datetime
        current_time = dt.time(0)
        close_time = datetime.strptime(self.params.close_time, '%H:%M').time()
        
        # 接近收盘，强制平仓
        if current_time >= close_time and self.position:
            self.order = self.close()
            self.log(f'Close: {self.data.close[0]:.2f}')
            return
        
        if self.order:
            return
        
        open_price = self.data.open[0]
        high_price = self.data.high[0]
        low_price = self.data.low[0]
        atr = self.atr[0]
        
        # 计算突破区间
        breakout_up = open_price * (1 + self.params.breakout_range)
        breakout_down = open_price * (1 - self.params.breakout_range)
        
        # 向上突破做多
        if high_price > breakout_up and not self.position:
            self.order = self.buy()
            self.log(f'Intraday LONG: {breakout_up:.2f}')
            self.intraday_trades.append({
                'date': dt.date(0),
                'entry': high_price,
                'type': 'LONG'
            })
        
        # 向下突破做空
        elif low_price < breakout_down and not self.position:
            self.order = self.sell()
            self.log(f'Intraday SHORT: {breakout_down:.2f}')
            self.intraday_trades.append({
                'date': dt.date(0),
                'entry': low_price,
                'type': 'SHORT'
            })
        
        # 止损/止盈检查
        elif self.position:
            if self.position.size > 0:  # 多头
                stop_loss = self.buy_price - self.params.stop_loss_atr * atr
                profit_target = self.buy_price + self.params.profit_target_atr * atr
                if self.data.close[0] < stop_loss:
                    self.order = self.close()
                    self.log(f'StopLoss LONG: {self.data.close[0]:.2f}')
                elif self.data.close[0] > profit_target:
                    self.order = self.close()
                    self.log(f'TakeProfit LONG: {self.data.close[0]:.2f}')
            else:  # 空头
                stop_loss = self.sell_price + self.params.stop_loss_atr * atr
                profit_target = self.sell_price - self.params.profit_target_atr * atr
                if self.data.close[0] > stop_loss:
                    self.order = self.close()
                    self.log(f'StopLoss SHORT: {self.data.close[0]:.2f}')
                elif self.data.close[0] < profit_target:
                    self.order = self.close()
                    self.log(f'TakeProfit SHORT: {self.data.close[0]:.2f}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')
    
    def stop(self):
        self.log(f'Intraday Done, Value={self.broker.getvalue():.2f}')


class MultiTimeframeChanTheoryStrategy(bt.Strategy):
    """
    多周期缠论策略
    
    特点：
    - 结合大周期判断趋势方向
    - 小周期寻找精确入场点
    - 识别笔和线段结构
    - 利用背驰判断转折
    """
    
    params = (
        ('big_period', 60),
        ('mid_period', 30),
        ('small_period', 10),
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
    )
    
    def __init__(self):
        # 多周期均线
        self.sma_big = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.big_period)
        self.sma_mid = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.mid_period)
        self.sma_small = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.small_period)
        
        # ATR
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        # MACD
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=12,
            period_me2=26,
            period_signal=9
        )
        
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        
        self.order = None
        self.chan_trades = []
        
    def next(self):
        if self.order:
            return
        
        close = self.data.close[0]
        atr = self.atr[0]
        
        # 多周期方向确认
        trend_up = close > self.sma_big[0]
        
        # 买入信号条件
        small_above_mid = self.sma_small[0] > self.sma_mid[0]
        small_above_mid_yesterday = self.sma_small[-1] <= self.sma_mid[-1]
        
        golden_cross = small_above_mid and small_above_mid_yesterday
        
        # RSI不超买超卖
        rsi_ok = 30 < self.rsi[0] < 70
        
        # 背驰判断
        macd_hist = self.macd.macd[0] - self.macd.signal[0]
        macd_hist_prev = self.macd.macd[-1] - self.macd.signal[-1]
        bullish_divergence = (close < self.data.close[-1] and macd_hist > macd_hist_prev)
        
        # 买入
        if (golden_cross or bullish_divergence) and trend_up and rsi_ok and not self.position:
            risk = self.params.atr_multiplier * atr
            if risk > 0:
                size = int(0.02 * self.broker.getvalue() / risk)
                size = max(1, size)
                
                self.order = self.buy(size=size)
                self.log(f'Chan BUY: {close:.2f}, RSI={self.rsi[0]:.1f}')
                self.chan_trades.append({
                    'date': self.datas[0].datetime.date(0),
                    'entry': close,
                    'signal': 'GOLDEN_CROSS' if golden_cross else 'DIVERGENCE'
                })
        
        # 卖出
        elif not trend_up and self.position:
            self.order = self.sell(size=self.position.size)
            self.log(f'Chan SELL: {close:.2f}')
        
        # 止损
        elif self.position and close < self.buy_price - self.params.atr_multiplier * atr:
            self.order = self.sell(size=self.position.size)
            self.log(f'Chan StopLoss: {close:.2f}')
    
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')


def run_backtest(data_df, strategy_class, strategy_name, initial_cash=100000, **kwargs):
    """运行回测"""
    print(f"
{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, **kwargs)
    
    data = bt.feeds.PandasData(
        dataname=data_df,
        datetime='date',
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest='open_interest',
        fromdate=datetime(2000, 1, 1),
        todate=datetime(2099, 12, 31),
    )
    
    cerebro.adddata(data)
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0002)
    cerebro.broker.set_slippage_perc(0.0005)
    
    print(f"Initial: {initial_cash:,.2f}")
    print(f"Data: {len(data_df)} bars")
    
    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    results = cerebro.run()
    strat = results[0]
    
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash * 100
    
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    
    sharpe_ratio = sharpe.get('sharperatio', 'N/A')
    max_drawdown = drawdown.get('max', {}).get('drawdown', 'N/A')
    
    print(f"
--- Results ---")
    print(f"Final Value: {final_value:,.2f}")
    print(f"Return: {total_return:.2f}%")
    print(f"Sharpe: {sharpe_ratio}")
    print(f"Max Drawdown: {max_drawdown}%")
    
    return {
        'strategy_name': strategy_name,
        'initial_cash': initial_cash,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trades': getattr(strat, 'trade_log', []) or getattr(strat, 'intraday_trades', []) or getattr(strat, 'chan_trades', []),
        'results': results,
        'cerebro': cerebro
    }


def generate_report(all_results):
    """生成回测报告"""
    report = []
    report.append("
" + "="*80)
    report.append("期货策略回测综合报告")
    report.append("="*80)
    report.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 汇总表
    report.append("## 回测结果汇总")
    report.append("")
    report.append("| 策略 | 总收益率 | 夏普比率 | 最大回撤 |")
    report.append("|------|---------|---------|---------|")
    
    for result in all_results:
        sharpe = result.get('sharpe_ratio', 'N/A')
        max_dd = result.get('max_drawdown', 'N/A')
        report.append(f"| {result['strategy_name']} | {result['total_return']:.2f}% | {sharpe} | {max_dd}% |")
    
    report.append("")
    
    # 策略对比
    report.append("## 策略对比分析")
    report.append("")
    best_return = max(all_results, key=lambda x: x['total_return'])
    lowest_dd = min(all_results, key=lambda x: x.get('max_drawdown', 999))
    report.append(f"- **最高收益**: {best_return['strategy_name']} ({best_return['total_return']:.2f}%)")
    report.append(f"- **最低回撤**: {lowest_dd['strategy_name']} ({lowest_dd['max_drawdown']}%)")
    report.append("")
    
    # 优化建议
    report.append("## 优化建议")
    report.append("")
    report.append("### 波动率自适应策略优化")
    report.append("- 波动率计算窗口可进一步优化")
    report.append("- 建议引入凯利公式进行仓位优化")
    report.append("")
    report.append("### 日内交易策略优化")
    report.append("- 可使用ATR突破替代固定百分比突破")
    report.append("- 增加趋势过滤器避免逆势交易")
    report.append("")
    report.append("### 多周期缠论策略优化")
    report.append("- 引入更严格的笔/线段识别算法")
    report.append("- 结合多指标共振确认背驰")
    report.append("")
    report.append("## 风险提示")
    report.append("")
    report.append("1. 回测结果不代表未来表现")
    report.append("2. 需考虑滑点和实际成交价差")
    report.append("3. 建议进行样本外测试验证策略有效性")
    report.append("")
    
    return "
".join(report)


def generate_mock_data(symbol):
    """生成模拟数据"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='B')
    
    if symbol in ['CU', 'CU2006']:
        base_price = 50000
        volatility = 0.02
    elif symbol in ['RB', 'RB2005']:
        base_price = 3500
        volatility = 0.025
    else:
        base_price = 4000
        volatility = 0.015
    
    returns = np.random.normal(0, volatility, len(dates))
    close_prices = base_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'date': dates,
        'open': close_prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        'high': close_prices * (1 + np.random.uniform(0, 0.015, len(dates))),
        'low': close_prices * (1 - np.random.uniform(0, 0.015, len(dates))),
        'close': close_prices,
        'volume': np.random.randint(100000, 1000000, len(dates)),
        'open_interest': np.random.randint(100000, 500000, len(dates)),
    })
    
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


def main():
    """主函数"""
    print("="*80)
    print("期货策略回测分析系统")
    print("="*80)
    
    from data_manager import load_from_csv
    
    test_symbols = ['IF2006', 'RB2005', 'CU2006']
    all_results = []
    
    for symbol in test_symbols:
        print(f"

{'#'*80}")
        print(f"# Symbol: {symbol}")
        print(f"{'#'*80}")
        
        data = load_from_csv(symbol)
        
        if data is None or data.empty:
            print(f"Generating mock data for {symbol}...")
            data = generate_mock_data(symbol)
        
        data = data.dropna(subset=['date', 'open', 'high', 'low', 'close'])
        data = data.sort_values('date').reset_index(drop=True)
        
        # 策略回测
        try:
            result1 = run_backtest(data, TrendStrategyOptimized, "Trend Strategy",
                fast_period=10, slow_period=30, atr_period=14)
            all_results.append(result1)
        except Exception as e:
            print(f"Trend strategy error: {e}")
        
        try:
            result2 = run_backtest(data, VolatilityAdaptiveTrendStrategy, "Volatility Adaptive",
                fast_period=10, slow_period=30, volatility_lookback=20)
            all_results.append(result2)
        except Exception as e:
            print(f"Vol adaptive error: {e}")
        
        try:
            result3 = run_backtest(data, IntradayStrategy, "Intraday",
                breakout_range=0.002, stop_loss_atr=2.0)
            all_results.append(result3)
        except Exception as e:
            print(f"Intraday error: {e}")
        
        try:
            result4 = run_backtest(data, MultiTimeframeChanTheoryStrategy, "Multi-Timeframe Chan",
                big_period=60, mid_period=30, small_period=10)
            all_results.append(result4)
        except Exception as e:
            print(f"Chan theory error: {e}")
    
    if all_results:
        report = generate_report(all_results)
        print(report)
        
        with open('/home/zjc/.openclaw/workspace/backtest_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        print("
Report saved: /home/zjc/.openclaw/workspace/backtest_report.md")


if __name__ == "__main__":
    main()
