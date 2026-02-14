#!/usr/bin/env python3
"""
技术指标单元测试
测试 BOLL、KDJ、OBV、ATR 指标计算
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestBOLL:
    """BOLL布林带测试类"""
    
    @pytest.fixture
    def sample_price_data(self):
        """生成测试用的价格数据"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
        # 生成一个有趋势的价格序列
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        return pd.Series(close, index=dates, name='close')
    
    def test_calculate_boll_basic(self, sample_price_data):
        """测试BOLL基本计算"""
        from indicators import TechnicalIndicators
        
        upper, middle, lower = TechnicalIndicators.calculate_boll(
            sample_price_data, period=20, std_dev=2
        )
        
        # 检查返回类型
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        
        # 检查长度
        assert len(upper) == len(sample_price_data)
        assert len(middle) == len(sample_price_data)
        assert len(lower) == len(sample_price_data)
        
        # 检查前19个值应该是NaN
        assert upper.iloc[:19].isna().all()
        assert middle.iloc[:19].isna().all()
        assert lower.iloc[:19].isna().all()
    
    def test_calculate_boll_values(self, sample_price_data):
        """测试BOLL数值关系"""
        from indicators import TechnicalIndicators
        
        upper, middle, lower = TechnicalIndicators.calculate_boll(
            sample_price_data, period=20, std_dev=2
        )
        
        # 检查数值关系：上轨 > 中轨 > 下轨
        valid_idx = 19  # 第一个有效值
        assert upper.iloc[valid_idx] > middle.iloc[valid_idx]
        assert middle.iloc[valid_idx] > lower.iloc[valid_idx]
    
    def test_calculate_boll_with_position(self, sample_price_data):
        """测试BOLL位置计算"""
        from indicators import TechnicalIndicators
        
        upper, middle, lower, width, position = \
            TechnicalIndicators.calculate_boll_with_position(
                sample_price_data, period=20, std_dev=2
            )
        
        # 位置应该在0-1之间
        valid_position = position.iloc[19:].dropna()
        assert (valid_position >= 0).all()
        assert (valid_position <= 1).all()
        
        # 宽度应该是上轨减下轨
        expected_width = upper - lower
        pd.testing.assert_series_equal(width, expected_width)
    
    def test_detect_boll_signal(self):
        """测试BOLL信号检测"""
        from indicators import TechnicalIndicators
        
        # 超买信号
        signal = TechnicalIndicators.detect_boll_signal(
            price=12.0, upper=11.0, lower=9.0, position=0.9
        )
        assert signal == "OVERBOUGHT"
        
        # 超卖信号
        signal = TechnicalIndicators.detect_boll_signal(
            price=8.0, upper=11.0, lower=9.0, position=0.1
        )
        assert signal == "OVERSOLD"
        
        # 中性信号
        signal = TechnicalIndicators.detect_boll_signal(
            price=10.0, upper=11.0, lower=9.0, position=0.5
        )
        assert signal == "NEUTRAL"
        
        # 触及上轨
        signal = TechnicalIndicators.detect_boll_signal(
            price=11.0, upper=11.0, lower=9.0, position=1.0
        )
        assert signal == "OVERBOUGHT"
        
        # 触及下轨
        signal = TechnicalIndicators.detect_boll_signal(
            price=9.0, upper=11.0, lower=9.0, position=0.0
        )
        assert signal == "OVERSOLD"
    
    def test_calculate_boll_short_period(self):
        """测试BOLL短周期计算"""
        from indicators import TechnicalIndicators
        
        # 短周期数据
        prices = pd.Series([1, 2, 3, 4, 5])
        upper, middle, lower = TechnicalIndicators.calculate_boll(
            prices, period=20, std_dev=2
        )
        
        # 所有值应该是NaN（数据不足）
        assert upper.isna().all()
        assert middle.isna().all()
        assert lower.isna().all()


class TestKDJ:
    """KDJ随机指标测试类"""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """生成测试用的OHLC数据"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.random.rand(100)
        low = close - np.random.rand(100)
        
        return pd.DataFrame({
            'close': close,
            'high': high,
            'low': low
        }, index=dates)
    
    def test_calculate_kdj_basic(self, sample_ohlc_data):
        """测试KDJ基本计算"""
        from indicators import TechnicalIndicators
        
        k, d, j = TechnicalIndicators.calculate_kdj(
            sample_ohlc_data['high'],
            sample_ohlc_data['low'],
            sample_ohlc_data['close'],
            n=9, m1=3, m2=3
        )
        
        # 检查返回类型
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        assert isinstance(j, pd.Series)
        
        # 检查长度
        assert len(k) == len(sample_ohlc_data)
        assert len(d) == len(sample_ohlc_data)
        assert len(j) == len(sample_ohlc_data)
    
    def test_calculate_kdj_values_range(self, sample_ohlc_data):
        """测试KDJ值范围"""
        from indicators import TechnicalIndicators
        
        k, d, j = TechnicalIndicators.calculate_kdj(
            sample_ohlc_data['high'],
            sample_ohlc_data['low'],
            sample_ohlc_data['close'],
            n=9, m1=3, m2=3
        )
        
        # K、D值应该在0-100之间（大部分情况）
        valid_idx = 8  # 第一个有效值
        valid_k = k.iloc[valid_idx:].dropna()
        valid_d = d.iloc[valid_idx:].dropna()
        
        # K、D值应该大部分在0-100之间
        assert (valid_k >= 0).all() and (valid_k <= 100).all()
        assert (valid_d >= 0).all() and (valid_d <= 100).all()
    
    def test_detect_kdj_signal(self):
        """测试KDJ信号检测"""
        from indicators import TechnicalIndicators
        
        # 超买信号
        signal = TechnicalIndicators.detect_kdj_signal(
            k=85, d=82, k_prev=80, d_prev=79
        )
        assert signal == "OVERBOUGHT"
        
        # 超卖信号
        signal = TechnicalIndicators.detect_kdj_signal(
            k=18, d=17, k_prev=20, d_prev=21
        )
        assert signal == "OVERSOLD"
        
        # 金叉信号
        signal = TechnicalIndicators.detect_kdj_signal(
            k=55, d=50, k_prev=48, d_prev=49
        )
        assert signal == "GOLD_CROSS"
        
        # 死叉信号
        signal = TechnicalIndicators.detect_kdj_signal(
            k=45, d=50, k_prev=52, d_prev=49
        )
        assert signal == "DEAD_CROSS"
        
        # 多头信号（K在D上方）
        signal = TechnicalIndicators.detect_kdj_signal(
            k=60, d=50, k_prev=58, d_prev=48
        )
        assert signal == "BULLISH"
        
        # 空头信号（K在D下方）
        signal = TechnicalIndicators.detect_kdj_signal(
            k=40, d=50, k_prev=42, d_prev=48
        )
        assert signal == "BEARISH"
    
    def test_detect_kdj_cross(self, sample_ohlc_data):
        """测试KDJ金叉死叉检测"""
        from indicators import TechnicalIndicators
        
        k, d, j = TechnicalIndicators.calculate_kdj(
            sample_ohlc_data['high'],
            sample_ohlc_data['low'],
            sample_ohlc_data['close'],
            n=9, m1=3, m2=3
        )
        
        signals = TechnicalIndicators.detect_kdj_cross(k, d)
        
        # 检查返回类型
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(k)
        
        # 检查信号类型
        valid_signals = signals.iloc[9:].dropna()
        assert all(s in ["NEUTRAL", "GOLD_CROSS", "DEAD_CROSS", "OVERBOUGHT", "OVERSOLD"] 
                   for s in valid_signals)


class TestOBV:
    """OBV能量潮测试类"""
    
    @pytest.fixture
    def sample_ohlcv_data(self):
        """生成测试用的OHLCV数据"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        volume = np.random.randint(1000000, 10000000, 100)
        
        return pd.DataFrame({
            'close': close,
            'volume': volume
        }, index=dates)
    
    def test_calculate_obv_basic(self, sample_ohlcv_data):
        """测试OBV基本计算"""
        from indicators import TechnicalIndicators
        
        obv, obv_ma = TechnicalIndicators.calculate_obv(
            sample_ohlcv_data['close'],
            sample_ohlcv_data['volume'],
            ma_period=10
        )
        
        # 检查返回类型
        assert isinstance(obv, pd.Series)
        assert isinstance(obv_ma, pd.Series)
        
        # 检查长度
        assert len(obv) == len(sample_ohlcv_data)
        assert len(obv_ma) == len(sample_ohlcv_data)
    
    def test_calculate_obv_trend(self):
        """测试OBV趋势"""
        from indicators import TechnicalIndicators
        
        # 持续上涨的股票
        close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        volume = pd.Series([1000] * 10)
        
        obv, _ = TechnicalIndicators.calculate_obv(close, volume)
        
        # 持续上涨时，OBV应该持续增加
        assert obv.iloc[-1] > obv.iloc[0]
        
        # 持续下跌的股票
        close = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        volume = pd.Series([1000] * 10)
        
        obv, _ = TechnicalIndicators.calculate_obv(close, volume)
        
        # 持续下跌时，OBV应该持续减少
        assert obv.iloc[-1] < obv.iloc[0]
    
    def test_detect_obv_signal(self):
        """测试OBV信号检测"""
        from indicators import TechnicalIndicators
        
        # OBV在均线上方
        signal = TechnicalIndicators.detect_obv_signal(
            obv=1000, obv_ma=900, obv_prev=950, obv_ma_prev=880
        )
        assert signal == "BULLISH"
        
        # OBV在均线下方
        signal = TechnicalIndicators.detect_obv_signal(
            obv=800, obv_ma=900, obv_prev=850, obv_ma_prev=880
        )
        assert signal == "BEARISH"


class TestATR:
    """ATR真实波幅测试类"""
    
    @pytest.fixture
    def sample_ohlc_data(self):
        """生成测试用的OHLC数据"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        
        return pd.DataFrame({
            'close': close,
            'high': high,
            'low': low
        }, index=dates)
    
    def test_calculate_atr_basic(self, sample_ohlc_data):
        """测试ATR基本计算"""
        from indicators import TechnicalIndicators
        
        atr, atr_ma = TechnicalIndicators.calculate_atr(
            sample_ohlc_data['high'],
            sample_ohlc_data['low'],
            sample_ohlc_data['close'],
            period=14
        )
        
        # 检查返回类型
        assert isinstance(atr, pd.Series)
        assert isinstance(atr_ma, pd.Series)
        
        # 检查长度
        assert len(atr) == len(sample_ohlc_data)
        assert len(atr_ma) == len(sample_ohlc_data)
    
    def test_calculate_atr_positive(self, sample_ohlc_data):
        """测试ATR值为正"""
        from indicators import TechnicalIndicators
        
        atr, _ = TechnicalIndicators.calculate_atr(
            sample_ohlc_data['high'],
            sample_ohlc_data['low'],
            sample_ohlc_data['close'],
            period=14
        )
        
        # ATR应该为正
        valid_atr = atr.iloc[13:].dropna()
        assert (valid_atr >= 0).all()
    
    def test_calculate_atr_percent(self, sample_ohlc_data):
        """测试ATR百分比计算"""
        from indicators import TechnicalIndicators
        
        atr_percent, atr_percent_ma = \
            TechnicalIndicators.calculate_atr_percent(
                sample_ohlc_data['high'],
                sample_ohlc_data['low'],
                sample_ohlc_data['close'],
                period=14
            )
        
        # ATR百分比应该在合理范围内
        valid_percent = atr_percent.iloc[13:].dropna()
        assert (valid_percent > 0).all()
        # 通常ATR百分比在0.5%-5%之间
        assert (valid_percent < 10).all()
    
    def test_detect_atr_signal(self):
        """测试ATR信号检测"""
        from indicators import TechnicalIndicators
        
        # 高波动率
        signal = TechnicalIndicators.detect_atr_signal(
            atr=5.0, atr_percent=4.0, atr_prev=4.5
        )
        assert signal == "HIGH_VOLATILITY"
        
        # 低波动率
        signal = TechnicalIndicators.detect_atr_signal(
            atr=0.5, atr_percent=0.5, atr_prev=0.6
        )
        assert signal == "LOW_VOLATILITY"
        
        # 正常波动率
        signal = TechnicalIndicators.detect_atr_signal(
            atr=1.5, atr_percent=2.0, atr_prev=1.4
        )
        assert signal == "NORMAL_VOLATILITY"


class TestAllIndicators:
    """综合指标计算测试类"""
    
    @pytest.fixture
    def full_ohlcv_data(self):
        """生成完整的OHLCV数据"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        volume = np.random.randint(1000000, 10000000, 100)
        
        return pd.DataFrame({
            'close': close,
            'high': high,
            'low': low,
            'volume': volume
        }, index=dates)
    
    def test_calculate_all_indicators(self, full_ohlcv_data):
        """测试计算所有指标"""
        from indicators import TechnicalIndicators
        
        result = TechnicalIndicators.calculate_all_indicators(full_ohlcv_data)
        
        # 检查所有必需的列
        required_cols = [
            'BOLL_upper', 'BOLL_middle', 'BOLL_lower',
            'BOLL_width', 'BOLL_position', 'BOLL_signal',
            'KDJ_K', 'KDJ_D', 'KDJ_J', 'KDJ_signal',
            'OBV', 'OBV_MA', 'OBV_signal',
            'ATR', 'ATR_MA', 'ATR_percent', 'ATR_signal'
        ]
        
        for col in required_cols:
            assert col in result.columns, f"缺少列: {col}"
    
    def test_get_latest_boll(self, full_ohlcv_data):
        """测试获取最新BOLL数据"""
        from indicators import TechnicalIndicators
        
        result = TechnicalIndicators.calculate_all_indicators(full_ohlcv_data)
        boll_result = TechnicalIndicators.get_latest_boll(result)
        
        assert boll_result is not None
        assert hasattr(boll_result, 'upper')
        assert hasattr(boll_result, 'middle')
        assert hasattr(boll_result, 'lower')
        assert hasattr(boll_result, 'width')
        assert hasattr(boll_result, 'position')
    
    def test_get_latest_kdj(self, full_ohlcv_data):
        """测试获取最新KDJ数据"""
        from indicators import TechnicalIndicators
        
        result = TechnicalIndicators.calculate_all_indicators(full_ohlcv_data)
        kdj_result = TechnicalIndicators.get_latest_kdj(result)
        
        assert kdj_result is not None
        assert hasattr(kdj_result, 'k')
        assert hasattr(kdj_result, 'd')
        assert hasattr(kdj_result, 'j')
        assert hasattr(kdj_result, 'signal')
    
    def test_get_latest_obv(self, full_ohlcv_data):
        """测试获取最新OBV数据"""
        from indicators import TechnicalIndicators
        
        result = TechnicalIndicators.calculate_all_indicators(full_ohlcv_data)
        obv_result = TechnicalIndicators.get_latest_obv(result)
        
        assert obv_result is not None
        assert hasattr(obv_result, 'obv')
        assert hasattr(obv_result, 'obv_ma')
        assert hasattr(obv_result, 'signal')
    
    def test_get_latest_atr(self, full_ohlcv_data):
        """测试获取最新ATR数据"""
        from indicators import TechnicalIndicators
        
        result = TechnicalIndicators.calculate_all_indicators(full_ohlcv_data)
        atr_result = TechnicalIndicators.get_latest_atr(result)
        
        assert atr_result is not None
        assert hasattr(atr_result, 'atr')
        assert hasattr(atr_result, 'atr_ma')
        assert hasattr(atr_result, 'signal')


class TestConvenienceFunctions:
    """便捷函数测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
        close = 10 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.random.rand(100)
        low = close - np.random.rand(100)
        volume = np.random.randint(1000000, 10000000, 100)
        
        return {
            'close': pd.Series(close, index=dates),
            'high': pd.Series(high, index=dates),
            'low': pd.Series(low, index=dates),
            'volume': pd.Series(volume, index=dates)
        }
    
    def test_calculate_boll_function(self, sample_data):
        """测试BOLL便捷函数"""
        from indicators import calculate_boll
        
        upper, middle, lower = calculate_boll(sample_data['close'])
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
    
    def test_calculate_kdj_function(self, sample_data):
        """测试KDJ便捷函数"""
        from indicators import calculate_kdj
        
        k, d, j = calculate_kdj(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)
        assert isinstance(j, pd.Series)
    
    def test_calculate_obv_function(self, sample_data):
        """测试OBV便捷函数"""
        from indicators import calculate_obv
        
        obv, obv_ma = calculate_obv(
            sample_data['close'],
            sample_data['volume']
        )
        assert isinstance(obv, pd.Series)
        assert isinstance(obv_ma, pd.Series)
    
    def test_calculate_atr_function(self, sample_data):
        """测试ATR便捷函数"""
        from indicators import calculate_atr
        
        atr, atr_ma = calculate_atr(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )
        assert isinstance(atr, pd.Series)
        assert isinstance(atr_ma, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
