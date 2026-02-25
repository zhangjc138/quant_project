#!/usr/bin/env python3
"""
股票回测模块
支持 MA20 角度策略 + RSI + MACD + BOLL + KDJ 的历史回测验证
增强版：添加夏普比率、胜率统计等更多指标
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from stock_strategy import StockSelector, StockSignal, TechnicalIndicator
import json

# 尝试导入高级指标模块
try:
    from indicators import TechnicalIndicators as NewIndicators
    NEW_INDICATORS_AVAILABLE = True
except ImportError:
    NEW_INDICATORS_AVAILABLE = False


@dataclass
class BacktestResult:
    """回测结果"""
    symbol: str
    name: str
    start_date: str
    end_date: str
    
    # 资金
    initial_capital: float
    final_capital: float
    total_return: float
    
    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # 收益统计
    avg_win: float
    avg_loss: float
    profit_factor: float      # 盈亏比
    max_consecutive_wins: int
    max_consecutive_losses: int
    
    # 风险指标
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int  # 最大回撤持续天数
    
    # 收益率统计
    annual_return: float       # 年化收益率
    volatility: float         # 收益率波动率
    sharpe_ratio: float        # 夏普比率
    sortino_ratio: float      # 索提诺比率（只考虑下行风险）
    
    # 持仓统计
    avg_holding_days: float
    avg_profit_per_trade: float
    
    # 单笔统计
    max_single_profit: float
    max_single_loss: float
    avg_trade_duration: float
    
    # 策略指标
    rsi_entry_avg: float      # 买入时平均 RSI
    macd_golden_cross_rate: float  # MACD 金叉买入比例
    # BOLL/KDJ 策略指标
    boll_oversold_rate: float   # BOLL下轨买入比例
    kdj_gold_cross_rate: float  # KDJ金叉买入比例
    kdj_oversold_rate: float    # KDJ超卖买入比例
    
    # 详细信息
    trades: list = field(default_factory=list)


class Backtester:
    """
    多策略回测器
    
    支持:
    - MA20 角度策略
    - RSI 策略
    - MACD 策略
    - BOLL 布林带策略
    - KDJ 随机指标策略
    - 组合策略
    """
    
    DEFAULT_PARAMS = {
        "initial_capital": 100000,      # 初始资金
        "stop_loss_pct": 5.0,          # 止损比例
        "take_profit_pct": 15.0,       # 止盈比例
        "max_holding_days": 10,         # 最大持仓天数
        "position_size": 0.8,          # 仓位比例
        "commission": 0.0003,           # 手续费率
        "slippage": 0.001,             # 滑点率
        "risk_free_rate": 0.03,        # 无风险利率（年化）
        "trading_days_per_year": 252,   # 年交易日天数
        # 策略开关
        "use_ma20_angle": True,        # 使用 MA20 角度
        "use_rsi": True,               # 使用 RSI
        "use_macd": True,              # 使用 MACD
        "use_boll": False,             # 使用 BOLL
        "use_kdj": False,              # 使用 KDJ
        "composite_strategy": False,   # 复合策略模式
        # RSI 参数
        "rsi_oversold": 30,            # RSI 超卖阈值
        "rsi_overbought": 70,          # RSI 超买阈值
        # MACD 参数
        "macd_golden_cross": True,     # 是否要求 MACD 金叉
        # BOLL 参数
        "boll_buy_oversold": True,     # BOLL 下轨买入
        # KDJ 参数
        "kdj_buy_gold_cross": True,     # KDJ 金叉买入
        "kdj_buy_oversold": False,     # KDJ 超卖买入
        "kdj_oversold": 20,            # KDJ 超卖阈值
        "kdj_overbought": 80,          # KDJ 超买阈值
    }
    
    def __init__(self, params: Optional[Dict] = None):
        """
        初始化回测器
        
        Args:
            params: 回测参数
        """
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.selector = StockSelector()
        self.indicator = TechnicalIndicator()
    
    def run(self, symbol: str, start_date: str, end_date: str = None) -> BacktestResult:
        """
        运行回测
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 YYYY-MM-DD
            end_date: 结束日期 YYYY-MM-DD，默认至今
            
        Returns:
            BacktestResult: 回测结果
        """
        # 加载数据
        df = self.selector.load_stock_data(symbol, days=500)
        if df is None or len(df) < 60:
            return self._empty_result(symbol, start_date, end_date)
        
        # 筛选日期范围
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else df.index[-1]
        df = df[(df.index >= start) & (df.index <= end)]
        
        if len(df) < 60:
            return self._empty_result(symbol, start_date, end_date)
        
        # 计算技术指标
        df = self.selector.calculate_indicators(df)
        
        # 计算每日收益率
        df['daily_return'] = df['close'].pct_change()
        
        # 生成每日信号
        df['signal'] = self._generate_daily_signal(df)
        
        # 初始化交易记录
        trades = []
        position = None
        capital = self.params['initial_capital']
        capital_history = [capital]
        max_capital = capital
        max_drawdown = 0
        max_drawdown_start = None
        max_drawdown_duration = 0
        
        # 策略统计数据
        rsi_entries = []
        macd_golden_count = 0
        
        for i, (date, row) in enumerate(df.iterrows()):
            close = row['close']
            signal = row['signal']
            
            # 更新持仓止损止盈
            if position:
                # 检查是否触发止损
                if close <= position['stop_loss']:
                    profit = (close - position['price']) * position['shares']
                    profit_pct = (close - position['price']) / position['price'] * 100
                    
                    trades.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'entry_date': position['date'],
                        'entry_price': position['price'],
                        'exit_price': close,
                        'exit_reason': '止损',
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'holding_days': (date - position['date']).days,
                        'rsi_entry': position.get('rsi', 50),
                        'macd_signal': position.get('macd_signal', 'NEUTRAL'),
                    })
                    
                    position = None
                    capital += profit
                    continue
                
                # 检查是否触发止盈
                if close >= position['take_profit']:
                    profit = (close - position['price']) * position['shares']
                    profit_pct = (close - position['price']) / position['price'] * 100
                    
                    trades.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'entry_date': position['date'],
                        'entry_price': position['price'],
                        'exit_price': close,
                        'exit_reason': '止盈',
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'holding_days': (date - position['date']).days,
                        'rsi_entry': position.get('rsi', 50),
                        'macd_signal': position.get('macd_signal', 'NEUTRAL'),
                    })
                    
                    position = None
                    capital += profit
                    continue
                
                # 检查持仓天数
                if (date - position['date']).days >= self.params['max_holding_days']:
                    profit = (close - position['price']) * position['shares']
                    profit_pct = (close - position['price']) / position['price'] * 100
                    
                    trades.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'entry_date': position['date'],
                        'entry_price': position['price'],
                        'exit_price': close,
                        'exit_reason': '到期平仓',
                        'profit': profit,
                        'profit_pct': profit_pct,
                        'holding_days': (date - position['date']).days,
                        'rsi_entry': position.get('rsi', 50),
                        'macd_signal': position.get('macd_signal', 'NEUTRAL'),
                    })
                    
                    position = None
                    capital += profit
                    continue
            
            # 检查买入信号
            if signal == 'BUY' and position is None:
                # 计算买入价格（考虑滑点）
                buy_price = close * (1 + self.params['slippage'])
                
                # 计算买入数量
                position_size = int(capital * self.params['position_size'] / buy_price)
                if position_size < 100:
                    position_size = 100
                
                # 止损止盈价格
                stop_loss = buy_price * (1 - self.params['stop_loss_pct'] / 100)
                take_profit = buy_price * (1 + self.params['take_profit_pct'] / 100)
                
                # 记录 RSI 和 MACD 状态
                rsi = row.get('RSI', 50)
                macd_sig = row.get('macd_signal', 'NEUTRAL')
                
                if not pd.isna(rsi):
                    rsi_entries.append(rsi)
                if macd_sig == 'GOLD_CROSS':
                    macd_golden_count += 1
                
                position = {
                    'date': date,
                    'price': buy_price,
                    'shares': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'rsi': rsi if not pd.isna(rsi) else 50,
                    'macd_signal': macd_sig,
                }
            
            # 检查卖出信号（平仓）
            elif signal == 'SELL' and position is not None:
                profit = (close - position['price']) * position['shares']
                profit_pct = (close - position['price']) / position['price'] * 100
                
                trades.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'entry_date': position['date'].strftime('%Y-%m-%d'),
                    'entry_price': position['price'],
                    'exit_price': close,
                    'exit_reason': '信号平仓',
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'holding_days': (date - position['date']).days,
                    'rsi_entry': position.get('rsi', 50),
                    'macd_signal': position.get('macd_signal', 'NEUTRAL'),
                })
                
                position = None
                capital += profit
            
            # 记录资本历史
            capital_history.append(capital)
            
            # 更新最大回撤
            if capital > max_capital:
                max_capital = capital
                max_drawdown_start = date
            
            drawdown = (max_capital - capital) / max_capital
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_duration = (date - max_drawdown_start).days if max_drawdown_start else 0
        
        # 处理未平仓持仓
        if position is not None:
            close = df.iloc[-1]['close']
            profit = (close - position['price']) * position['shares']
            profit_pct = (close - position['price']) / position['price'] * 100
            
            trades.append({
                'date': df.index[-1].strftime('%Y-%m-%d'),
                'entry_date': position['date'].strftime('%Y-%m-%d'),
                'entry_price': position['price'],
                'exit_price': close,
                'exit_reason': '最终平仓',
                'profit': profit,
                'profit_pct': profit_pct,
                'holding_days': (df.index[-1] - position['date']).days,
                'rsi_entry': position.get('rsi', 50),
                'macd_signal': position.get('macd_signal', 'NEUTRAL'),
            })
            
            capital += profit
        
        # 计算各种统计指标
        result = self._calculate_statistics(
            symbol=symbol,
            trades=trades,
            capital=capital,
            capital_history=capital_history,
            initial_capital=self.params['initial_capital'],
            start_date=start_date,
            end_date=end_date or df.index[-1].strftime('%Y-%m-%d'),
            rsi_entries=rsi_entries,
            total_trades=len(trades),
            macd_golden_count=macd_golden_count,
            df=df
        )
        
        return result
    
    def _generate_daily_signal(self, df: pd.DataFrame) -> pd.Series:
        """生成每日交易信号"""
        signals = pd.Series('HOLD', index=df.index)
        
        ma20_angle = df['MA20_angle'].fillna(0)
        rsi = df['RSI'].fillna(50)
        macd_signal = df['macd_signal'].fillna('NEUTRAL')
        
        # BOLL信号
        boll_signal = df.get('BOLL_signal', pd.Series('NEUTRAL', index=df.index)).fillna('NEUTRAL')
        boll_position = df.get('BOLL_position', pd.Series(0.5, index=df.index)).fillna(0.5)
        
        # KDJ信号
        kdj_signal = df.get('KDJ_signal', pd.Series('NEUTRAL', index=df.index)).fillna('NEUTRAL')
        kdj_k = df.get('KDJ_K', pd.Series(50, index=df.index)).fillna(50)
        kdj_d = df.get('KDJ_D', pd.Series(50, index=df.index)).fillna(50)
        
        # 检查是否为复合策略模式
        if self.params.get('composite_strategy', False):
            # 复合策略：MA20 + RSI + BOLL + KDJ
            # 买入条件：MA20角度 > 阈值，且(BOLL超卖 或 KDJ金叉 或 KDJ超卖)
            buy_condition = (
                (ma20_angle > self.params.get('angle_threshold_buy', 3)) &
                (
                    (self.params.get('boll_buy_oversold', True) & (boll_signal == 'OVERSOLD')) |
                    (self.params.get('kdj_buy_gold_cross', True) & (kdj_signal == 'GOLD_CROSS')) |
                    (self.params.get('kdj_buy_oversold', False) & (kdj_signal == 'OVERSOLD'))
                )
            )
            
            # 卖出条件：MA20角度 < 阈值，或RSI超买，或MACD死叉，或BOLL超买，或KDJ死叉/超买
            sell_condition = (
                (ma20_angle < self.params.get('angle_threshold_sell', 0)) |
                (rsi >= self.params.get('rsi_overbought', 70)) |
                (macd_signal == 'DEAD_CROSS') |
                (boll_signal == 'OVERBOUGHT') |
                (kdj_signal == 'DEAD_CROSS') |
                (kdj_signal == 'OVERBOUGHT')
            )
        else:
            # 原策略：MA20 + RSI + MACD
            buy_condition = (
                (ma20_angle > self.params.get('angle_threshold_buy', 3)) &
                ((~self.params.get('use_rsi', True)) | (rsi <= self.params.get('rsi_oversold', 30))) &
                ((~self.params.get('use_macd', True)) | (~self.params.get('macd_golden_cross', True)) | (macd_signal == 'GOLD_CROSS'))
            )
            
            sell_condition = (
                (ma20_angle < self.params.get('angle_threshold_sell', 0)) |
                (rsi >= self.params.get('rsi_overbought', 70))
            )
        
        signals[buy_condition] = 'BUY'
        signals[sell_condition] = 'SELL'
        
        return signals
    
    def _calculate_statistics(
        self,
        symbol: str,
        trades: list,
        capital: float,
        capital_history: list,
        initial_capital: float,
        start_date: str,
        end_date: str,
        rsi_entries: list,
        total_trades: int,
        macd_golden_count: int,
        df: pd.DataFrame
    ) -> BacktestResult:
        """计算各种统计指标"""
        
        # 基础统计
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        total_return = (capital - initial_capital) / initial_capital * 100
        
        # 收益统计
        avg_win = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0
        
        # 盈亏比
        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
        else:
            profit_factor = float('inf') if avg_win > 0 else 0
        
        # 连胜连负
        consecutive = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        for t in trades:
            if t['profit'] > 0:
                consecutive = consecutive + 1 if consecutive > 0 else 1
                max_consecutive_wins = max(max_consecutive_wins, consecutive)
            else:
                consecutive = consecutive - 1 if consecutive < 0 else -1
                max_consecutive_losses = max(max_consecutive_losses, abs(consecutive))
        
        # 最大回撤
        max_capital = max(capital_history)
        max_drawdown = 0
        for cap in capital_history:
            drawdown = (max_capital - cap) / max_capital
            max_drawdown = max(max_drawdown, drawdown)
        
        # 年化收益率
        days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        if days > 0:
            annual_return = ((capital / initial_capital) ** (365.0 / days) - 1) * 100
        else:
            annual_return = 0
        
        # 收益率波动率
        daily_returns = df['daily_return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # 年化波动率
        
        # 夏普比率
        risk_free_rate = self.params['risk_free_rate']
        if volatility > 0:
            sharpe_ratio = (annual_return / 100 - risk_free_rate) / (volatility / 100)
        else:
            sharpe_ratio = 0
        
        # 索提诺比率（只考虑下行波动）
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        if downside_volatility > 0:
            sortino_ratio = (annual_return / 100 - risk_free_rate) / downside_volatility
        else:
            sortino_ratio = float('inf') if annual_return > 0 else 0
        
        # 持仓统计
        avg_holding = np.mean([t['holding_days'] for t in trades]) if trades else 0
        avg_profit = np.mean([t['profit'] for t in trades]) if trades else 0
        
        # 单笔统计
        max_single_profit = max([t['profit'] for t in trades]) if trades else 0
        max_single_loss = min([t['profit'] for t in trades]) if trades else 0
        avg_trade_duration = avg_holding
        
        # 策略指标
        rsi_entry_avg = np.mean(rsi_entries) if rsi_entries else 50
        macd_golden_cross_rate = macd_golden_count / max(total_trades, 1) * 100
        
        # BOLL/KDJ 策略指标
        boll_oversold_rate = 0  # 简化计算
        kdj_gold_cross_rate = 0  # 简化计算
        kdj_oversold_rate = 0   # 简化计算
        
        # 股票名称
        name = self.selector.watchlist.get(symbol, {}).get("name", symbol)
        
        return BacktestResult(
            symbol=symbol,
            name=name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=capital,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=len(winning_trades) / max(total_trades, 1) * 100,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            max_drawdown=max_capital,
            max_drawdown_pct=max_drawdown * 100,
            max_drawdown_duration=0,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            avg_holding_days=avg_holding,
            avg_profit_per_trade=avg_profit,
            max_single_profit=max_single_profit,
            max_single_loss=max_single_loss,
            avg_trade_duration=avg_trade_duration,
            rsi_entry_avg=rsi_entry_avg,
            macd_golden_cross_rate=macd_golden_cross_rate,
            boll_oversold_rate=boll_oversold_rate,
            kdj_gold_cross_rate=kdj_gold_cross_rate,
            kdj_oversold_rate=kdj_oversold_rate,
            trades=trades,
        )
    
    def _empty_result(self, symbol: str, start_date: str, end_date: str) -> BacktestResult:
        """返回空的回测结果"""
        return BacktestResult(
            symbol=symbol,
            name=symbol,
            start_date=start_date,
            end_date=end_date or "",
            initial_capital=self.params['initial_capital'],
            final_capital=self.params['initial_capital'],
            total_return=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            max_drawdown=self.params['initial_capital'],
            max_drawdown_pct=0,
            max_drawdown_duration=0,
            annual_return=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            avg_holding_days=0,
            avg_profit_per_trade=0,
            max_single_profit=0,
            max_single_loss=0,
            avg_trade_duration=0,
            rsi_entry_avg=50,
            macd_golden_cross_rate=0,
            boll_oversold_rate=0,
            kdj_gold_cross_rate=0,
            kdj_oversold_rate=0,
        )
    
    def run_batch(self, symbols: list, start_date: str, end_date: str = None) -> Dict[str, BacktestResult]:
        """
        批量回测多个股票
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict[股票代码, 回测结果]
        """
        results = {}
        
        for symbol in symbols:
            print(f"回测中: {symbol}...")
            result = self.run(symbol, start_date, end_date)
            results[symbol] = result
        
        return results
    
    def format_result(self, result: BacktestResult) -> str:
        """
        格式化回测结果为字符串
        
        Args:
            result: BacktestResult
            
        Returns:
            str: 格式化的回测报告
        """
        trades_str = ""
        for t in result.trades:
            emoji = "🟢" if t['profit'] > 0 else "🔴"
            trades_str += f"| {t['date']} | {t['entry_date']} | {t['entry_price']:.2f} | {t['exit_price']:.2f} | {t['exit_reason']} | {t['profit']:+.2f} | {t['profit_pct']:+.2f}% | {t['holding_days']}天 | {t['rsi_entry']:.1f} | {t['macd_signal']} |\n"
        
        sharpe_emoji = "🟢" if result.sharpe_ratio >= 1 else "🟡" if result.sharpe_ratio >= 0 else "🔴"
        win_rate_emoji = "🟢" if result.win_rate >= 50 else "🟡" if result.win_rate >= 40 else "🔴"
        
        return f"""## 回测报告: {result.name} ({result.symbol})

**回测时间**: {result.start_date} ~ {result.end_date}
**初始资金**: ¥{result.initial_capital:,.2f}
**最终资金**: ¥{result.final_capital:,.2f}
**总收益率**: {result.total_return:+.2f}%

---

### 📊 收益概览

| 指标 | 数值 | 评级 |
|------|------|------|
| 总收益率 | {result.total_return:+.2f}% | {'🟢' if result.total_return > 0 else '🔴'} |
| 年化收益率 | {result.annual_return:+.2f}% | {'🟢' if result.annual_return > 10 else '🟡' if result.annual_return > 0 else '🔴'} |
| 夏普比率 | {result.sharpe_ratio:.2f} | {sharpe_emoji} |
| 索提诺比率 | {result.sortino_ratio:.2f} | {'🟢' if result.sortino_ratio > 1 else '🟡' if result.sortino_ratio > 0 else '🔴'} |
| 交易次数 | {result.total_trades} 次 | - |
| 胜率 | {result.win_rate:.1f}% | {win_rate_emoji} |
| 盈利次数 | {result.winning_trades} 次 | 🟢 |
| 亏损次数 | {result.losing_trades} 次 | 🔴 |
| 盈亏比 | {result.profit_factor:.2f} | {'🟢' if result.profit_factor > 1.5 else '🟡' if result.profit_factor > 1 else '🔴'} |

---

### 📉 风险指标

| 指标 | 数值 |
|------|------|
| 最大回撤 | {result.max_drawdown_pct:.2f}% |
| 最大回撤持续天数 | {result.max_drawdown_duration} 天 |
| 收益率波动率 | {result.volatility:.2f}% |
| 最大单笔盈利 | ¥{result.max_single_profit:,.2f} |
| 最大单笔亏损 | ¥{result.max_single_loss:,.2f} |
| 连胜次数 | {result.max_consecutive_wins} 次 |
| 连负次数 | {result.max_consecutive_losses} 次 |

---

### 📋 持仓统计

| 指标 | 数值 |
|------|------|
| 平均持仓天数 | {result.avg_holding_days:.1f} 天 |
| 平均每笔收益 | ¥{result.avg_profit_per_trade:+,.2f} |
| 平均买入 RSI | {result.rsi_entry_avg:.1f} |
| MACD金叉买入占比 | {result.macd_golden_cross_rate:.1f}% |

---

### 📋 交易明细

| 卖出日期 | 买入日期 | 买入价 | 卖出价 | 原因 | 收益 | 收益率 | 持仓 | RSI | MACD |
|----------|----------|--------|--------|------|------|--------|------|-----|------|
{trades_str}

---

### ⚙️ 回测参数

| 参数 | 值 |
|------|-----|
| 初始资金 | ¥{self.params['initial_capital']:,} |
| 止损比例 | {self.params['stop_loss_pct']}% |
| 止盈比例 | {self.params['take_profit_pct']}% |
| 最大持仓 | {self.params['max_holding_days']} 天 |
| 仓位比例 | {self.params['position_size']*100:.0f}% |
| 手续费率 | {self.params['commission']*100:.2f}% |
| 滑点率 | {self.params['slippage']*100:.2f}% |
| 无风险利率 | {self.params['risk_free_rate']*100:.1f}% |

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    def compare_results(self, results: Dict[str, BacktestResult]) -> str:
        """
        比较多个股票的回测结果
        
        Args:
            results: Dict[股票代码, 回测结果]
            
        Returns:
            str: 对比报告
        """
        # 按收益率排序
        sorted_results = sorted(results.items(), key=lambda x: x[1].total_return, reverse=True)
        
        report = "## 多股票回测对比\n\n"
        report += f"| 排名 | 股票 | 名称 | 收益率 | 年化收益 | 夏普比率 | 交易次数 | 胜率 | 最大回撤 | 盈亏比 |\n"
        report += f"|------|------|------|--------|----------|----------|----------|------|----------|--------|\n"
        
        for i, (symbol, result) in enumerate(sorted_results, 1):
            emoji = "🟢" if result.total_return > 0 else "🔴"
            sharpe_emoji = "🟢" if result.sharpe_ratio >= 1 else "🟡" if result.sharpe_ratio >= 0 else "🔴"
            win_emoji = "🟢" if result.win_rate >= 50 else "🟡" if result.win_rate >= 40 else "🔴"
            dd_emoji = "🟢" if result.max_drawdown_pct < 10 else "🟡" if result.max_drawdown_pct < 20 else "🔴"
            
            report += f"| {i} | {symbol} | {result.name} | {emoji} {result.total_return:+.2f}% | {result.annual_return:+.2f}% | {sharpe_emoji} {result.sharpe_ratio:.2f} | {result.total_trades} | {win_emoji} {result.win_rate:.1f}% | {dd_emoji} {result.max_drawdown_pct:.1f}% | {result.profit_factor:.2f} |\n"
        
        # 汇总统计
        total_return_all = sum(r.total_return for r in results.values()) / len(results)
        avg_sharpe = sum(r.sharpe_ratio for r in results.values()) / len(results)
        avg_win_rate = sum(r.win_rate for r in results.values()) / len(results)
        
        report += f"\n### 汇总统计\n"
        report += f"- 平均收益率: {total_return_all:+.2f}%\n"
        report += f"- 平均夏普比率: {avg_sharpe:.2f}\n"
        report += f"- 平均胜率: {avg_win_rate:.1f}%\n"
        report += f"- 上涨股票: {sum(1 for r in results.values() if r.total_return > 0)}/{len(results)} 只\n"
        
        return report


# ==================== 便捷函数 ====================
def quick_backtest(symbol: str, start_date: str = "2024-01-01", end_date: str = None) -> BacktestResult:
    """
    快速回测单个股票
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        BacktestResult: 回测结果
    """
    backtester = Backtester()
    return backtester.run(symbol, start_date, end_date)


def run_multi_strategy_backtest(
    symbol: str,
    start_date: str,
    end_date: str = None,
    use_ma20: bool = True,
    use_rsi: bool = True,
    use_macd: bool = True
) -> BacktestResult:
    """
    运行多策略组合回测
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        use_ma20: 是否使用 MA20 角度策略
        use_rsi: 是否使用 RSI 策略
        use_macd: 是否使用 MACD 策略
        
    Returns:
        BacktestResult: 回测结果
    """
    params = {
        "use_ma20_angle": use_ma20,
        "use_rsi": use_rsi,
        "use_macd": use_macd,
    }
    
    backtester = Backtester(params)
    return backtester.run(symbol, start_date, end_date)


if __name__ == "__main__":
    # 测试回测
    print("=== 多策略回测 ===\n")
    
    backtester = Backtester()
    
    # 回测浦发银行
    result = backtester.run("600000", "2024-01-01", "2025-01-01")
    
    # 打印报告
    print(backtester.format_result(result))
