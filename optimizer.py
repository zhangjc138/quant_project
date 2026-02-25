#!/usr/bin/env python3
"""
参数优化器模块

提供策略参数网格搜索和优化功能
支持 MA/ RSI / MACD / BOLL / KDJ 等参数优化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from itertools import product
import akshare as ak
from datetime import datetime, timedelta


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict          # 最佳参数
    best_score: float          # 最佳得分
    all_results: List[Dict]    # 所有结果
    total_combinations: int    # 总组合数
    elapsed_seconds: float    # 耗时（秒）


class ParameterOptimizer:
    """参数优化器"""
    
    def __init__(
        self,
        symbol: str = "600519",
        start_date: str = None,
        end_date: str = None,
        initial_capital: float = 100000
    ):
        """
        初始化优化器
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
        """
        self.symbol = symbol
        self.start_date = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        
        # 加载数据
        self.df = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        """加载股票数据"""
        try:
            # 使用 akshare 获取数据
            if self.symbol.startswith('6') or self.symbol.startswith('5'):
                # 上海交易所
                df = ak.stock_zh_a_hist(
                    symbol=self.symbol,
                    period="daily",
                    start_date=self.start_date,
                    end_date=self.end_date,
                    adjust="qfq"
                )
            else:
                # 深圳交易所
                df = ak.stock_zh_a_hist(
                    symbol=self.symbol,
                    period="daily",
                    start_date=self.start_date,
                    end_date=self.end_date,
                    adjust="qfq"
                )
            
            if df is None or df.empty:
                # 返回空DataFrame
                return pd.DataFrame()
            
            # 标准化列名
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()
            
            return df
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def _calculate_ma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """计算移动平均线"""
        return df['close'].rolling(window=period).mean()
    
    def _calculate_ma_angle(self, df: pd.DataFrame, period: int) -> pd.Series:
        """计算MA角度"""
        ma = self._calculate_ma(df, period)
        angle = np.arctan((ma - ma.shift(1)) / ma.shift(1)) * 180 / np.pi
        return angle
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD"""
        ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal_period, adjust=False).mean()
        macd = (dif - dea) * 2
        
        return dif, dea, macd
    
    def _calculate_boll(
        self,
        df: pd.DataFrame,
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        ma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        
        return upper, lower, ma
    
    def _calculate_kdj(
        self,
        df: pd.DataFrame,
        n: int = 9,
        m1: int = 3,
        m2: int = 3
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算KDJ"""
        low_min = df['low'].rolling(window=n).min()
        high_max = df['high'].rolling(window=n).max()
        
        rsv = (df['close'] - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50)
        
        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def _backtest(
        self,
        df: pd.DataFrame,
        ma_period: int = 20,
        ma_angle_threshold: float = 3.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        boll_period: int = 20,
        kdj_n: int = 9,
        kdj_m1: int = 3,
        kdj_m2: int = 3,
        stop_loss: float = 0.05,
        take_profit: float = 0.15
    ) -> Dict:
        """
        策略回测
        
        Args:
            df: 股票数据
            ma_period: MA周期
            ma_angle_threshold: MA角度阈值
            rsi_period: RSI周期
            rsi_oversold: RSI超卖阈值
            rsi_overbought: RSI超买阈值
            macd_fast: MACD快速EMA周期
            macd_slow: MACD慢速EMA周期
            macd_signal: MACD信号线周期
            boll_period: BOLL周期
            kdj_n: KDJ N周期
            kdj_m1: KDJ M1周期
            kdj_m2: KDJ M2周期
            stop_loss: 止损比例
            take_profit: 止盈比例
            
        Returns:
            Dict: 回测结果
        """
        if len(df) < 50:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'trade_count': 0
            }
        
        # 计算指标
        df = df.copy()
        df['ma'] = self._calculate_ma(df, ma_period)
        df['ma_angle'] = self._calculate_ma_angle(df, ma_period)
        df['rsi'] = self._calculate_rsi(df, rsi_period)
        df['macd_dif'], df['macd_dea'], df['macd'] = self._calculate_macd(
            df, macd_fast, macd_slow, macd_signal
        )
        df['boll_upper'], df['boll_lower'], df['boll_mid'] = self._calculate_boll(df, boll_period)
        df['kdj_k'], df['kdj_d'], df['kdj_j'] = self._calculate_kdj(df, kdj_n, kdj_m1, kdj_m2)
        
        # 生成信号
        df['signal'] = 0
        df.loc[
            (df['ma_angle'] > ma_angle_threshold) &
            (df['rsi'] > rsi_oversold) &
            (df['rsi'] < rsi_overbought),
            'signal'
        ] = 1  # 买入信号
        
        df['signal'] = df['signal'].shift(1)
        df['signal'] = df['signal'].fillna(0)
        
        # 计算持仓
        df['position'] = df['signal'].cumsum()
        df['position'] = df['position'].clip(0, 1)
        
        # 计算收益率
        df['daily_return'] = df['close'].pct_change()
        df['strategy_return'] = df['daily_return'] * df['position'].shift(1)
        
        # 去除NaN
        df = df.dropna()
        
        if len(df) == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'trade_count': 0
            }
        
        # 计算指标
        total_return = (1 + df['strategy_return']).prod() - 1
        
        # 夏普比率
        strategy_returns = df['strategy_return']
        if strategy_returns.std() != 0:
            sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # 最大回撤
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 交易次数
        df['trade'] = df['signal'].diff().fillna(0)
        trade_count = (df['trade'] != 0).sum()
        
        # 胜率
        winning_trades = (df['strategy_return'] > 0).sum()
        if trade_count > 0:
            win_rate = winning_trades / trade_count
        else:
            win_rate = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trade_count': trade_count
        }
    
    def optimize_ma(
        self,
        periods: List[int] = [10, 20, 30, 60, 120],
        angle_thresholds: List[float] = [1.0, 2.0, 3.0, 5.0, 7.0],
        score_metric: str = 'sharpe_ratio'
    ) -> OptimizationResult:
        """
        优化MA参数
        
        Args:
            periods: MA周期列表
            angle_thresholds: 角度阈值列表
            score_metric: 评分指标 ('total_return', 'sharpe_ratio', 'win_rate')
            
        Returns:
            OptimizationResult: 优化结果
        """
        start_time = datetime.now()
        
        results = []
        
        for period, threshold in product(periods, angle_thresholds):
            result = self._backtest(
                self.df,
                ma_period=period,
                ma_angle_threshold=threshold
            )
            
            result['params'] = {
                'ma_period': period,
                'ma_angle_threshold': threshold
            }
            result['score'] = result.get(score_metric, 0)
            
            results.append(result)
        
        # 找最佳参数
        best = max(results, key=lambda x: x['score'])
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best['params'],
            best_score=best['score'],
            all_results=results,
            total_combinations=len(results),
            elapsed_seconds=elapsed
        )
    
    def optimize_rsi(
        self,
        periods: List[int] = [6, 9, 14, 21],
        oversold_values: List[float] = [20, 25, 30, 35],
        overbought_values: List[float] = [65, 70, 75, 80],
        score_metric: str = 'sharpe_ratio'
    ) -> OptimizationResult:
        """
        优化RSI参数
        
        Args:
            periods: RSI周期列表
            oversold_values: 超卖值列表
            overbought_values: 超买值列表
            score_metric: 评分指标
            
        Returns:
            OptimizationResult: 优化结果
        """
        start_time = datetime.now()
        
        results = []
        
        for period, oversold, overbought in product(periods, oversold_values, overbought_values):
            result = self._backtest(
                self.df,
                rsi_period=period,
                rsi_oversold=oversold,
                rsi_overbought=overbought
            )
            
            result['params'] = {
                'rsi_period': period,
                'rsi_oversold': oversold,
                'rsi_overbought': overbought
            }
            result['score'] = result.get(score_metric, 0)
            
            results.append(result)
        
        best = max(results, key=lambda x: x['score'])
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best['params'],
            best_score=best['score'],
            all_results=results,
            total_combinations=len(results),
            elapsed_seconds=elapsed
        )
    
    def optimize_macd(
        self,
        fast_periods: List[int] = [8, 12, 16],
        slow_periods: List[int] = [22, 26, 30],
        signal_periods: List[int] = [7, 9, 11],
        score_metric: str = 'sharpe_ratio'
    ) -> OptimizationResult:
        """
        优化MACD参数
        
        Args:
            fast_periods: 快速EMA周期列表
            slow_periods: 慢速EMA周期列表
            signal_periods: 信号线周期列表
            score_metric: 评分指标
            
        Returns:
            OptimizationResult: 优化结果
        """
        start_time = datetime.now()
        
        results = []
        
        for fast, slow, signal in product(fast_periods, slow_periods, signal_periods):
            if fast >= slow:
                continue  # 快速周期必须小于慢速周期
            
            result = self._backtest(
                self.df,
                macd_fast=fast,
                macd_slow=slow,
                macd_signal=signal
            )
            
            result['params'] = {
                'macd_fast': fast,
                'macd_slow': slow,
                'macd_signal': signal
            }
            result['score'] = result.get(score_metric, 0)
            
            results.append(result)
        
        best = max(results, key=lambda x: x['score'])
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best['params'],
            best_score=best['score'],
            all_results=results,
            total_combinations=len(results),
            elapsed_seconds=elapsed
        )
    
    def optimize_combined(
        self,
        ma_periods: List[int] = [10, 20, 30],
        ma_angles: List[float] = [2.0, 3.0, 5.0],
        rsi_periods: List[int] = [9, 14],
        rsi_oversolds: List[float] = [25, 30, 35],
        rsi_overboughts: List[float] = [65, 70, 75],
        score_metric: str = 'total_return'
    ) -> OptimizationResult:
        """
        综合优化MA和RSI参数
        
        Args:
            ma_periods: MA周期列表
            ma_angles: MA角度阈值列表
            rsi_periods: RSI周期列表
            rsi_oversolds: RSI超卖值列表
            rsi_overboughts: RSI超买值列表
            score_metric: 评分指标
            
        Returns:
            OptimizationResult: 优化结果
        """
        start_time = datetime.now()
        
        results = []
        
        total = len(ma_periods) * len(ma_angles) * len(rsi_periods) * len(rsi_oversolds) * len(rsi_overboughts)
        print(f"开始综合优化，共 {total} 种组合...")
        
        count = 0
        for ma_period, ma_angle, rsi_period, rsi_oversold, rsi_overbought in product(
            ma_periods, ma_angles, rsi_periods, rsi_oversolds, rsi_overboughts
        ):
            count += 1
            if count % 10 == 0:
                print(f"进度: {count}/{total} ({count/total*100:.1f}%)")
            
            result = self._backtest(
                self.df,
                ma_period=ma_period,
                ma_angle_threshold=ma_angle,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought
            )
            
            result['params'] = {
                'ma_period': ma_period,
                'ma_angle_threshold': ma_angle,
                'rsi_period': rsi_period,
                'rsi_oversold': rsi_oversold,
                'rsi_overbought': rsi_overbought
            }
            result['score'] = result.get(score_metric, 0)
            
            results.append(result)
        
        best = max(results, key=lambda x: x['score'])
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"优化完成，耗时 {elapsed:.1f} 秒")
        
        return OptimizationResult(
            best_params=best['params'],
            best_score=best['score'],
            all_results=results,
            total_combinations=len(results),
            elapsed_seconds=elapsed
        )
    
    def print_results(self, result: OptimizationResult):
        """打印优化结果"""
        print("\n" + "="*60)
        print("📊 参数优化结果")
        print("="*60)
        print(f"总组合数: {result.total_combinations}")
        print(f"最佳得分: {result.best_score:.4f}")
        print(f"最佳参数: {result.best_params}")
        print(f"耗时: {result.elapsed_seconds:.2f} 秒")
        
        print("\n" + "-"*60)
        print("TOP 10 结果:")
        print("-"*60)
        
        sorted_results = sorted(result.all_results, key=lambda x: x['score'], reverse=True)
        
        for i, r in enumerate(sorted_results[:10], 1):
            print(f"{i}. Score={r['score']:.4f}, "
                  f"Return={r['total_return']*100:.2f}%, "
                  f"Sharpe={r['sharpe_ratio']:.3f}, "
                  f"WinRate={r['win_rate']*100:.1f}%, "
                  f"Params={r['params']}")


if __name__ == "__main__":
    # 测试优化器
    print("参数优化器测试")
    
    optimizer = ParameterOptimizer(symbol="600519")
    
    if not optimizer.df.empty:
        # 优化MA参数
        print("\n优化MA参数...")
        result = optimizer.optimize_ma(
            periods=[10, 20, 30, 60],
            angle_thresholds=[2.0, 3.0, 5.0]
        )
        optimizer.print_results(result)
    else:
        print("加载数据失败")
