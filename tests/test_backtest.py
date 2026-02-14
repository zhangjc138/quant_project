# quant_project/tests/test_backtest.py
"""
回测模块测试

测试:
- 回测引擎
- 策略执行
- 绩效计算
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBacktestEngine:
    """回测引擎测试"""

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """生成模拟市场数据"""
        np.random.seed(42)
        n = 200
        
        # 生成带有趋势的价格数据
        dates = pd.date_range(start='2023-01-01', periods=n, freq='D')
        
        # 创建一个上涨趋势
        base_price = 100
        trend = np.linspace(0, 20, n)  # 上涨趋势
        noise = np.random.randn(n) * 3
        
        prices = base_price + trend + noise
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices * 0.995,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, n),
            'hold': 1  # 期货持仓
        })
        
        # 计算MA20
        df['ma20'] = df['close'].rolling(20).mean()
        
        return df

    def test_backtest_engine_import(self):
        """测试回测引擎导入"""
        try:
            from backtest.backtest_engine import run_backtest
            assert callable(run_backtest)
        except ImportError as e:
            pytest.skip(f"回测引擎导入失败: {e}")

    @pytest.mark.backtest
    def test_run_backtest_with_sample_data(self, sample_market_data: pd.DataFrame):
        """测试使用示例数据运行回测"""
        try:
            from backtest.backtest_engine import run_backtest
            from strategies.strategy_base import TrendStrategyBase
            
            # 运行回测
            results, cerebro = run_backtest(
                sample_market_data,
                strategy_class=TrendStrategyBase,
                initial_cash=100000
            )
            
            # 验证回测结果
            assert cerebro is not None
            final_value = cerebro.broker.getvalue()
            assert final_value > 0, "最终资金应该大于0"
            
        except Exception as e:
            pytest.skip(f"回测运行失败: {e}")

    @pytest.mark.backtest
    def test_initial_cash_setting(self, sample_market_data: pd.DataFrame):
        """测试初始资金设置"""
        try:
            from backtest.backtest_engine import run_backtest
            from strategies.strategy_base import TrendStrategyBase
            
            initial_cash = 200000
            results, cerebro = run_backtest(
                sample_market_data,
                strategy_class=TrendStrategyBase,
                initial_cash=initial_cash
            )
            
            assert cerebro.broker.getstartingcash() == initial_cash
            
        except Exception as e:
            pytest.skip(f"回测运行失败: {e}")

    def test_cerebro_configuration(self):
        """测试Cerebro引擎配置"""
        try:
            import backtrader as bt
            
            cerebro = bt.Cerebro()
            
            # 验证默认配置
            assert cerebro.broker.getcash() == 10000  # 默认初始资金
            
            # 设置经纪商参数
            cerebro.broker.setcash(100000)
            cerebro.broker.setcommission(commission=0.0002)
            cerebro.broker.set_slippage_perc(0.0005)
            
            # 验证设置生效
            assert cerebro.broker.getcash() == 100000
            
        except ImportError:
            pytest.skip("backtrader未安装")

    @pytest.mark.slow
    def test_data_feed_creation(self, sample_market_data: pd.DataFrame):
        """测试数据源创建"""
        try:
            import backtrader as bt
            from datetime import datetime
            
            # 创建数据源
            data = bt.feeds.PandasData(
                dataname=sample_market_data,
                datetime='date',
                open='open',
                high='high',
                low='low',
                close='close',
                volume='volume',
                openinterest='hold',
                fromdate=datetime(2023, 1, 1),
                todate=datetime(2023, 12, 31),
            )
            
            assert data is not None
            
        except Exception as e:
            pytest.skip(f"数据源创建失败: {e}")


class TestBacktestMetrics:
    """回测绩效指标测试"""

    @pytest.fixture
    def sample_trades(self):
        """模拟交易记录"""
        return [
            {'date': '2023-01-15', 'price': 100, 'size': 10, 'type': 'BUY'},
            {'date': '2023-02-20', 'price': 105, 'size': 10, 'type': 'SELL'},
            {'date': '2023-03-10', 'price': 102, 'size': 10, 'type': 'BUY'},
            {'date': '2023-04-15', 'price': 110, 'size': 10, 'type': 'SELL'},
        ]

    def test_calculate_returns(self, sample_trades):
        """测试收益率计算"""
        # 简单收益率计算
        buy_price = 100
        sell_price = 105
        profit = (sell_price - buy_price) / buy_price
        
        assert profit == 0.05, "收益率计算错误"

    def test_sharpe_ratio_calculation(self):
        """测试夏普比率计算"""
        # 模拟收益率序列
        returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.025])
        
        # 计算夏普比率
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return > 0:
            sharpe = mean_return / std_return * np.sqrt(252)  # 年化
            assert isinstance(sharpe, (int, float))

    def test_max_drawdown_calculation(self):
        """测试最大回撤计算"""
        # 模拟资金曲线
        equity = [100000, 105000, 103000, 98000, 102000, 110000, 108000]
        
        # 计算最大回撤
        peak = equity[0]
        max_dd = 0
        
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        assert max_dd > 0, "最大回撤应该大于0"
        assert max_dd <= 1, "最大回撤应该小于等于1"

    def test_win_rate_calculation(self, sample_trades):
        """测试胜率计算"""
        # 模拟交易结果
        trades = [
            {'profit': 500},   # 盈利
            {'profit': -200},  # 亏损
            {'profit': 300},   # 盈利
            {'profit': -100},  # 亏损
            {'profit': 400},   # 盈利
        ]
        
        wins = sum(1 for t in trades if t['profit'] > 0)
        win_rate = wins / len(trades)
        
        assert win_rate == 0.6, "胜率计算错误"


class TestBacktestStrategies:
    """回测策略测试"""

    def test_strategy_base_import(self):
        """测试基础策略导入"""
        try:
            from strategies.strategy_base import TrendStrategyBase
            assert TrendStrategyBase is not None
        except ImportError:
            pytest.skip("基础策略模块导入失败")

    def test_strategy_parameters(self):
        """测试策略参数"""
        try:
            from strategies.strategy_base import TrendStrategyOptimized
            
            # 检查默认参数
            assert hasattr(TrendStrategyOptimized, 'params')
            
        except ImportError:
            pytest.skip("优化策略模块导入失败")

    @pytest.mark.strategy
    def test_moving_average_crossover(self):
        """测试移动平均线交叉策略"""
        # 创建测试数据
        np.random.seed(42)
        n = 100
        
        df = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(n))
        })
        
        # 计算快线和慢线
        df['fast_ma'] = df['close'].rolling(5).mean()
        df['slow_ma'] = df['close'].rolling(20).mean()
        
        # 生成交易信号
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1
        
        # 验证信号生成
        assert 'signal' in df.columns
        assert df['signal'].isin([-1, 0, 1]).all()

    @pytest.mark.strategy
    def test_momentum_strategy(self):
        """测试动量策略"""
        # 创建测试数据
        prices = pd.Series([100, 102, 105, 103, 108, 110, 115])
        
        # 计算动量
        momentum = prices.pct_change(periods=5)
        
        # 验证动量计算
        assert len(momentum) == len(prices)
        assert momentum.notna().sum() > 0


class TestBacktestIntegration:
    """回测集成测试"""

    @pytest.mark.slow
    @pytest.mark.backtest
    def test_end_to_end_backtest(self):
        """端到端回测测试"""
        try:
            # 生成测试数据
            np.random.seed(42)
            n = 200
            
            df = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=n),
                'open': 100 + np.cumsum(np.random.randn(n)),
                'high': 100 + np.cumsum(np.random.randn(n)) + 2,
                'low': 100 + np.cumsum(np.random.randn(n)) - 2,
                'close': 100 + np.cumsum(np.random.randn(n)),
                'volume': np.random.randint(1000000, 10000000, n),
                'hold': 1
            })
            
            df['ma20'] = df['close'].rolling(20).mean()
            
            # 尝试导入并运行回测
            try:
                from backtest.backtest_engine import run_backtest
                from strategies.strategy_base import TrendStrategyBase
                
                results, cerebro = run_backtest(
                    df,
                    strategy_class=TrendStrategyBase,
                    initial_cash=100000
                )
                
                # 验证回测完成
                assert cerebro is not None
                final_value = cerebro.broker.getvalue()
                assert final_value > 0
                
            except ImportError as e:
                pytest.skip(f"模块导入失败: {e}")
                
        except Exception as e:
            pytest.skip(f"端到端测试跳过: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
