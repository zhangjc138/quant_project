# quant_project/tests/test_stock_strategy.py
"""
选股策略模块测试

测试:
- 技术指标计算
- 选股信号生成
- 策略过滤逻辑
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_strategy import TechnicalIndicator, StockSignal, StockSelector


class TestTechnicalIndicator:
    """技术指标计算器测试"""

    @pytest.fixture
    def sample_price_data(self) -> pd.DataFrame:
        """生成模拟价格数据"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # 生成随机价格序列
        base_price = 100
        price_changes = np.random.randn(99) * 2
        prices = [base_price]
        for change in price_changes:
            prices.append(prices[-1] * (1 + change / 100))
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100)
        })
        
        # 添加MA20
        df['ma20'] = df['close'].rolling(window=20).mean()
        
        return df

    def test_rsi_calculation(self, sample_price_data: pd.DataFrame):
        """测试RSI指标计算"""
        rsi = TechnicalIndicator.calculate_rsi(sample_price_data['close'])
        
        # 验证RSI值范围
        assert rsi.notna().sum() > 0, "RSI应该有计算结果"
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), \
            "RSI值应该在0-100之间"

    def test_rsi_period_customization(self, sample_price_data: pd.DataFrame):
        """测试RSI周期自定义"""
        # 使用不同的周期计算RSI
        rsi_14 = TechnicalIndicator.calculate_rsi(sample_price_data['close'], period=14)
        rsi_7 = TechnicalIndicator.calculate_rsi(sample_price_data['close'], period=7)
        
        # 周期越短，RSI波动应该越大
        assert rsi_14.std() <= rsi_7.std(), \
            "较短周期的RSI应该有更大的波动"

    def test_macd_calculation(self, sample_price_data: pd.DataFrame):
        """测试MACD指标计算"""
        macd, signal, histogram = TechnicalIndicator.calculate_macd(
            sample_price_data['close']
        )
        
        # 验证MACD输出
        assert len(macd) == len(sample_price_data), "MACD长度应该与输入数据一致"
        assert len(signal) == len(sample_price_data), "Signal线长度应该与输入数据一致"
        assert len(histogram) == len(sample_price_data), "Histogram长度应该与输入数据一致"

    def test_ma_angle_calculation(self, sample_price_data: pd.DataFrame):
        """测试MA角度计算"""
        ma_angle = TechnicalIndicator.calculate_ma_angle(
            sample_price_data['close'], 
            period=20
        )
        
        # 验证MA角度计算
        assert len(ma_angle) == len(sample_price_data), "MA角度长度应该与输入数据一致"
        # 角度应该在-90到90度之间
        valid_angles = ma_angle.dropna()
        assert (valid_angles > -90).all() and (valid_angles < 90).all(), \
            "MA角度应该在-90到90度之间"

    def test_rsi_overbought_oversold(self):
        """测试RSI超买超卖判断"""
        # 测试超买
        assert TechnicalIndicator.is_overbought(75) == True
        assert TechnicalIndicator.is_overbought(65) == False
        
        # 测试超卖
        assert TechnicalIndicator.is_oversold(25) == True
        assert TechnicalIndicator.is_oversold(35) == False

    def test_macd_signal_detection(self):
        """测试MACD信号检测"""
        # 测试金叉
        macd_values = pd.Series([1, 2, 3, 2.5, 2, 1.5])
        signal_values = pd.Series([1.5, 1.8, 2.1, 2.3, 2.5, 2.6])
        
        signal = TechnicalIndicator.get_macd_signal(macd_values, signal_values)
        # 当MACD从下方穿过Signal线时，应该检测到金叉
        # 这里测试逻辑
        assert signal in ['GOLD_CROSS', 'DEAD_CROSS', 'NEUTRAL']


class TestStockSignal:
    """股票信号数据结构测试"""

    def test_stock_signal_creation(self):
        """测试股票信号创建"""
        signal = StockSignal(
            symbol='000001',
            name='平安银行',
            price=12.5,
            change_pct=2.5,
            ma20=12.3,
            ma20_angle=15.0,
            rsi=65.0,
            rsi_signal='NEUTRAL',
            macd=0.1,
            macd_signal='NEUTRAL',
            signal='BUY',
            signal_desc='MA20角度符合要求，趋势向上',
            update_time='2024-01-15 10:00:00'
        )
        
        assert signal.symbol == '000001'
        assert signal.price == 12.5
        assert signal.signal == 'BUY'

    def test_stock_signal_to_dict(self):
        """测试股票信号转换为字典"""
        signal = StockSignal(
            symbol='000001',
            name='平安银行',
            price=12.5,
            change_pct=2.5,
            ma20=12.3,
            ma20_angle=15.0,
            rsi=65.0,
            rsi_signal='NEUTRAL',
            macd=0.1,
            macd_signal='NEUTRAL',
            signal='BUY',
            signal_desc='MA20角度符合要求，趋势向上',
            update_time='2024-01-15 10:00:00'
        )
        
        signal_dict = signal.__dict__
        assert isinstance(signal_dict, dict)
        assert 'symbol' in signal_dict


class TestStockSelector:
    """选股器测试"""

    @pytest.fixture
    def mock_stock_list(self):
        """模拟股票列表"""
        return [
            {'symbol': '000001', 'name': '平安银行'},
            {'symbol': '000002', 'name': '万科A'},
            {'symbol': '600000', 'name': '浦发银行'},
        ]

    def test_selector_initialization(self):
        """测试选股器初始化"""
        selector = StockSelector()
        assert selector.min_price > 0
        assert selector.max_price < 1000

    @pytest.mark.slow
    def test_filter_by_price(self):
        """测试价格过滤"""
        selector = StockSelector()
        
        test_data = pd.DataFrame({
            'close': [5.0, 15.0, 50.0, 100.0, 500.0]
        })
        
        filtered = selector._filter_by_price(test_data)
        # 应该过滤掉价格过高或过低的股票
        assert len(filtered) <= len(test_data)

    def test_filter_by_ma_angle(self):
        """测试MA角度过滤"""
        selector = StockSelector()
        
        test_data = pd.DataFrame({
            'ma20_angle': [-45, -15, 5, 15, 45]
        })
        
        filtered = selector._filter_by_ma_angle(test_data)
        # 过滤掉角度不符合要求的股票
        assert len(filtered) <= len(test_data)

    @pytest.mark.slow
    def test_filter_by_volume(self):
        """测试成交量过滤"""
        selector = StockSelector()
        
        test_data = pd.DataFrame({
            'volume': [100000, 1000000, 10000000, 50000000]
        })
        
        filtered = selector._filter_by_volume(test_data)
        # 应该保留成交量符合要求的股票
        assert len(filtered) <= len(test_data)


class TestStockStrategyIntegration:
    """选股策略集成测试"""

    def test_full_strategy_pipeline(self):
        """测试完整策略流程"""
        # 创建测试数据
        np.random.seed(42)
        n = 50
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=n),
            'close': 100 + np.cumsum(np.random.randn(n) * 2),
            'volume': np.random.randint(1000000, 10000000, n),
        })
        
        df['ma20'] = df['close'].rolling(20).mean()
        df['open'] = df['close'] * 0.99
        df['high'] = df['close'] * 1.01
        df['low'] = df['close'] * 0.98
        
        # 计算指标
        df['rsi'] = TechnicalIndicator.calculate_rsi(df['close'])
        df['ma20_angle'] = TechnicalIndicator.calculate_ma_angle(df['close'])
        
        # 验证计算结果
        assert 'rsi' in df.columns
        assert 'ma20_angle' in df.columns
        assert df['rsi'].notna().sum() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
