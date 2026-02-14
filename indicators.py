#!/usr/bin/env python3
"""
技术指标计算模块

支持以下技术指标:
- BOLL (布林带)
- KDJ (随机指标)
- OBV (能量潮)
- ATR (真实波幅)
- RSI (相对强弱指数) - 已有
- MACD (移动平均收敛 divergence) - 已有
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class BOLLResult:
    """布林带计算结果"""
    upper: float      # 上轨
    middle: float     # 中轨 (MA)
    lower: float      # 下轨
    width: float      # 布林带宽度
    position: float   # 价格在布林带中的位置 (0-1)


@dataclass
class KDJResult:
    """KDJ指标计算结果"""
    rsv: float       # RSV值
    k: float         # K值
    d: float         # D值
    j: float         # J值
    signal: str      # 信号类型


@dataclass
class OBVResult:
    """OBV计算结果"""
    obv: float       # OBV值
    obv_ma: float    # OBV移动平均
    signal: str      # 信号类型


@dataclass
class ATRResult:
    """ATR计算结果"""
    atr: float       # ATR值
    atr_ma: float    # ATR移动平均
    signal: str      # 信号类型


class TechnicalIndicators:
    """
    技术指标计算器
    
    支持:
    - BOLL (布林带)
    - KDJ (随机指标)
    - OBV (能量潮)
    - ATR (真实波幅)
    """
    
    # BOLL参数
    BOLL_PERIOD = 20
    BOLL_STD_DEV = 2
    
    # KDJ参数
    KDJ_N = 9          # RSV计算周期
    KDJ_M1 = 3         # K值平滑因子
    KDJ_M2 = 3         # D值平滑因子
    
    # KDJ阈值
    KDJ_OVERBOUGHT = 80
    KDJ_OVERSOLD = 20
    
    # ATR参数
    ATR_PERIOD = 14
    
    # OBV参数
    OBV_MA_PERIOD = 10
    
    # ==================== BOLL 布林带 ====================
    
    @staticmethod
    def calculate_boll(
        close_prices: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算布林带
        
        Args:
            close_prices: 收盘价序列
            period: 计算周期 (默认20)
            std_dev: 标准差倍数 (默认2)
            
        Returns:
            Tuple[上轨, 中轨, 下轨]
        """
        if len(close_prices) < period:
            nans = pd.Series([np.nan] * len(close_prices), index=close_prices.index)
            return nans, nans, nans
        
        # 中轨 = N日简单移动平均
        middle = close_prices.rolling(window=period).mean()
        
        # 标准差
        std = close_prices.rolling(window=period).std()
        
        # 上轨 = 中轨 + K * 标准差
        upper = middle + std_dev * std
        
        # 下轨 = 中轨 - K * 标准差
        lower = middle - std_dev * std
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_boll_with_position(
        close_prices: pd.Series,
        period: int = 20,
        std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        计算布林带（含位置信息）
        
        Returns:
            Tuple[上轨, 中轨, 下轨, 带宽, 位置]
        """
        upper, middle, lower = TechnicalIndicators.calculate_boll(
            close_prices, period, std_dev
        )
        
        # 布林带宽度
        width = upper - lower
        
        # 价格在布林带中的位置 (0 = 下轨, 1 = 上轨)
        # 避免除以0
        bandwidth = upper - lower
        bandwidth = bandwidth.replace(0, np.nan)
        position = (close_prices - lower) / bandwidth
        
        return upper, middle, lower, width, position
    
    @staticmethod
    def detect_boll_signal(
        price: float,
        upper: float,
        lower: float,
        position: float
    ) -> str:
        """
        检测布林带信号
        
        Args:
            price: 当前价格
            upper: 上轨
            lower: 下轨
            position: 位置 (0-1)
            
        Returns:
            信号类型: OVERBOUGHT/OVERSOLD/NEUTRAL
        """
        if pd.isna(upper) or pd.isna(lower) or pd.isna(position):
            return "NEUTRAL"
        
        if price >= upper:
            return "OVERBOUGHT"  # 触及上轨，可能回调
        elif price <= lower:
            return "OVERSOLD"   # 触及下轨，可能反弹
        elif position > 0.8:
            return "OVERBOUGHT"  # 接近上轨
        elif position < 0.2:
            return "OVERSOLD"   # 接近下轨
        else:
            return "NEUTRAL"     # 区间震荡
    
    # ==================== KDJ 随机指标 ====================
    
    @staticmethod
    def calculate_kdj(
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        n: int = 9,
        m1: int = 3,
        m2: int = 3
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        计算KDJ随机指标
        
        Args:
            high_prices: 最高价序列
            low_prices: 最低价序列
            close_prices: 收盘价序列
            n: RSV计算周期 (默认9)
            m1: K值平滑因子 (默认3)
            m2: D值平滑因子 (默认3)
            
        Returns:
            Tuple[K值, D值, J值]
        """
        if len(close_prices) < n:
            nans = pd.Series([np.nan] * len(close_prices), index=close_prices.index)
            return nans, nans, nans
        
        # 计算RSV
        lowest_low = low_prices.rolling(window=n).min()
        highest_high = high_prices.rolling(window=n).max()
        
        # 避免除以0
        denominator = highest_high - lowest_low
        denominator = denominator.replace(0, np.nan)
        rsv = ((close_prices - lowest_low) / denominator * 100).fillna(50)
        
        # 计算K、D、J值
        # K = 2/3 * 前一日K + 1/3 * RSV
        # D = 2/3 * 前一日D + 1/3 * K
        # J = 3 * K - 2 * D
        
        k = pd.Series(index=close_prices.index, dtype=float)
        d = pd.Series(index=close_prices.index, dtype=float)
        j = pd.Series(index=close_prices.index, dtype=float)
        
        # 初始化K、D值
        k.iloc[0] = 50
        d.iloc[0] = 50
        
        for i in range(1, len(close_prices)):
            k.iloc[i] = (2/3) * k.iloc[i-1] + (1/3) * rsv.iloc[i]
            d.iloc[i] = (2/3) * d.iloc[i-1] + (1/3) * k.iloc[i]
        
        # J值 = 3*K - 2*D
        j = 3 * k - 2 * d
        
        return k, d, j
    
    @staticmethod
    def detect_kdj_signal(
        k: float,
        d: float,
        k_prev: float = None,
        d_prev: float = None
    ) -> str:
        """
        检测KDJ信号
        
        Args:
            k: 当前K值
            d: 当前D值
            k_prev: 前一日K值 (可选，用于检测金叉死叉)
            d_prev: 前一日D值 (可选)
            
        Returns:
            信号类型
        """
        if pd.isna(k) or pd.isna(d):
            return "NEUTRAL"
        
        # 超买超卖
        if k >= TechnicalIndicators.KDJ_OVERBOUGHT and d >= TechnicalIndicators.KDJ_OVERBOUGHT:
            return "OVERBOUGHT"  # 超买
        elif k <= TechnicalIndicators.KDJ_OVERSOLD and d <= TechnicalIndicators.KDJ_OVERSOLD:
            return "OVERSOLD"   # 超卖
        
        # 金叉死叉检测
        if k_prev is not None and d_prev is not None:
            if not pd.isna(k_prev) and not pd.isna(d_prev):
                # 金叉: K从下方穿过D
                if k_prev <= d_prev and k > d:
                    return "GOLD_CROSS"
                # 死叉: K从上方穿过D
                elif k_prev >= d_prev and k < d:
                    return "DEAD_CROSS"
        
        # 多头/空头
        if k > d:
            return "BULLISH"  # K在D上方，多头
        else:
            return "BEARISH"  # K在D下方，空头
    
    @staticmethod
    def detect_kdj_cross(
        k_series: pd.Series,
        d_series: pd.Series
    ) -> pd.Series:
        """
        检测KDJ金叉死叉信号序列
        
        Returns:
            信号序列
        """
        signals = pd.Series("NEUTRAL", index=k_series.index)
        
        for i in range(1, len(k_series)):
            k = k_series.iloc[i]
            d = d_series.iloc[i]
            k_prev = k_series.iloc[i-1]
            d_prev = d_series.iloc[i-1]
            
            if pd.isna(k) or pd.isna(d) or pd.isna(k_prev) or pd.isna(d_prev):
                continue
            
            # 金叉
            if k_prev <= d_prev and k > d:
                signals.iloc[i] = "GOLD_CROSS"
            # 死叉
            elif k_prev >= d_prev and k < d:
                signals.iloc[i] = "DEAD_CROSS"
            # 超买
            elif k >= TechnicalIndicators.KDJ_OVERBOUGHT and d >= TechnicalIndicators.KDJ_OVERBOUGHT:
                signals.iloc[i] = "OVERBOUGHT"
            # 超卖
            elif k <= TechnicalIndicators.KDJ_OVERSOLD and d <= TechnicalIndicators.KDJ_OVERSOLD:
                signals.iloc[i] = "OVERSOLD"
        
        return signals
    
    # ==================== OBV 能量潮 ====================
    
    @staticmethod
    def calculate_obv(
        close_prices: pd.Series,
        volume: pd.Series,
        ma_period: int = 10
    ) -> Tuple[pd.Series, pd.Series]:
        """
        计算OBV能量潮指标
        
        Args:
            close_prices: 收盘价序列
            volume: 成交量序列
            ma_period: OBV移动平均周期
            
        Returns:
            Tuple[OBV, OBV_MA]
        """
        # 计算价格变化
        price_change = close_prices.diff()
        
        # 初始化OBV
        obv = pd.Series(0, index=close_prices.index)
        
        # OBV计算逻辑
        for i in range(1, len(close_prices)):
            if price_change.iloc[i] > 0:
                # 上涨，OBV累加
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                # 下跌，OBV递减
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                # 持平，OBV不变
                obv.iloc[i] = obv.iloc[i-1]
        
        # 计算OBV移动平均
        obv_ma = obv.rolling(window=ma_period).mean()
        
        return obv, obv_ma
    
    @staticmethod
    def detect_obv_signal(
        obv: float,
        obv_ma: float,
        obv_prev: float = None,
        obv_ma_prev: float = None
    ) -> str:
        """
        检测OBV信号
        
        Args:
            obv: 当前OBV值
            obv_ma: 当前OBV均线值
            obv_prev: 前一日OBV值
            obv_ma_prev: 前一日OBV均线值
            
        Returns:
            信号类型
        """
        if pd.isna(obv) or pd.isna(obv_ma):
            return "NEUTRAL"
        
        # OBV在均线上方，上涨趋势
        if obv > obv_ma:
            return "BULLISH"
        # OBV在均线下方，下跌趋势
        elif obv < obv_ma:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    # ==================== ATR 真实波幅 ====================
    
    @staticmethod
    def calculate_atr(
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """
        计算ATR真实波幅指标
        
        Args:
            high_prices: 最高价序列
            low_prices: 最低价序列
            close_prices: 收盘价序列
            period: ATR计算周期
            
        Returns:
            Tuple[ATR, ATR_MA]
        """
        # 计算真实波幅 (True Range)
        tr1 = high_prices - low_prices                    # 今日高低
        tr2 = abs(high_prices - close_prices.shift(1))   # 今日最高与昨日收盘
        tr3 = abs(low_prices - close_prices.shift(1))    # 今日最低与昨日收盘
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR (简单移动平均)
        atr = tr.rolling(window=period).mean()
        
        # ATR移动平均
        atr_ma = atr.rolling(window=period).mean()
        
        return atr, atr_ma
    
    @staticmethod
    def calculate_atr_percent(
        high_prices: pd.Series,
        low_prices: pd.Series,
        close_prices: pd.Series,
        period: int = 14
    ) -> Tuple[pd.Series, pd.Series]:
        """
        计算ATR百分比（波动率）
        
        Returns:
            Tuple[ATR_Percent, ATR_Percent_MA]
        """
        atr, atr_ma = TechnicalIndicators.calculate_atr(
            high_prices, low_prices, close_prices, period
        )
        
        # ATR百分比 = ATR / 收盘价 * 100
        atr_percent = (atr / close_prices) * 100
        atr_percent_ma = (atr_ma / close_prices) * 100
        
        return atr_percent, atr_percent_ma
    
    @staticmethod
    def detect_atr_signal(
        atr: float,
        atr_percent: float,
        atr_prev: float = None
    ) -> str:
        """
        检测ATR信号（波动率状态）
        
        Args:
            atr: 当前ATR值
            atr_percent: ATR百分比
            atr_prev: 前一日ATR值
            
        Returns:
            信号类型: HIGH_VOLATILITY/NORMAL_VOLATILITY/LOW_VOLATILITY
        """
        if pd.isna(atr) or pd.isna(atr_percent):
            return "NORMAL_VOLATILITY"
        
        # ATR百分比大于3%表示高波动率
        if atr_percent > 3.0:
            return "HIGH_VOLATILITY"
        # ATR百分比小于1%表示低波动率
        elif atr_percent < 1.0:
            return "LOW_VOLATILITY"
        else:
            return "NORMAL_VOLATILITY"
    
    # ==================== 综合指标计算 ====================
    
    @classmethod
    def calculate_all_indicators(
        cls,
        df: pd.DataFrame,
        boll_period: int = 20,
        boll_std: int = 2,
        kdj_n: int = 9,
        kdj_m1: int = 3,
        kdj_m2: int = 3,
        atr_period: int = 14,
        obv_period: int = 10
    ) -> pd.DataFrame:
        """
        计算所有技术指标
        
        Args:
            df: 包含 high, low, close, volume 列的DataFrame
            boll_period: BOLL周期
            boll_std: BOLL标准差倍数
            kdj_n: KDJ计算周期
            kdj_m1: K值平滑因子
            kdj_m2: D值平滑因子
            atr_period: ATR周期
            obv_period: OBV均线周期
            
        Returns:
            包含所有指标的DataFrame
        """
        result = df.copy()
        
        # BOLL
        result['BOLL_upper'], result['BOLL_middle'], result['BOLL_lower'] = \
            cls.calculate_boll(result['close'], boll_period, boll_std)
        
        result['BOLL_width'], result['BOLL_position'] = \
            cls.calculate_boll_with_position(
                result['close'], boll_period, boll_std
            )[3:]
        
        result['BOLL_signal'] = result.apply(
            lambda x: cls.detect_boll_signal(
                x['close'], x['BOLL_upper'], x['BOLL_lower'], x['BOLL_position']
            ), axis=1
        )
        
        # KDJ
        result['KDJ_K'], result['KDJ_D'], result['KDJ_J'] = \
            cls.calculate_kdj(
                result['high'], result['low'], result['close'],
                kdj_n, kdj_m1, kdj_m2
            )
        
        # KDJ信号（检测金叉死叉）
        kdj_signals = cls.detect_kdj_cross(
            result['KDJ_K'], result['KDJ_D']
        )
        result['KDJ_signal'] = kdj_signals
        
        # OBV
        result['OBV'], result['OBV_MA'] = cls.calculate_obv(
            result['close'], result['volume'], obv_period
        )
        
        result['OBV_signal'] = result.apply(
            lambda x: cls.detect_obv_signal(
                x['OBV'], x['OBV_MA']
            ), axis=1
        )
        
        # ATR
        result['ATR'], result['ATR_MA'] = cls.calculate_atr(
            result['high'], result['low'], result['close'], atr_period
        )
        
        result['ATR_percent'], _ = cls.calculate_atr_percent(
            result['high'], result['low'], result['close'], atr_period
        )
        
        result['ATR_signal'] = result.apply(
            lambda x: cls.detect_atr_signal(
                x['ATR'], x['ATR_percent']
            ), axis=1
        )
        
        return result
    
    # ==================== 便捷函数 ====================
    
    @staticmethod
    def get_latest_boll(df: pd.DataFrame) -> Optional[BOLLResult]:
        """获取最新布林带数据"""
        if df is None or len(df) < 20:
            return None
        
        upper = df['BOLL_upper'].iloc[-1] if 'BOLL_upper' in df else None
        middle = df['BOLL_middle'].iloc[-1] if 'BOLL_middle' in df else None
        lower = df['BOLL_lower'].iloc[-1] if 'BOLL_lower' in df else None
        width = df['BOLL_width'].iloc[-1] if 'BOLL_width' in df else None
        position = df['BOLL_position'].iloc[-1] if 'BOLL_position' in df else None
        
        if pd.isna(upper):
            return None
        
        return BOLLResult(
            upper=upper,
            middle=middle,
            lower=lower,
            width=width,
            position=position
        )
    
    @staticmethod
    def get_latest_kdj(df: pd.DataFrame) -> Optional[KDJResult]:
        """获取最新KDJ数据"""
        if df is None or len(df) < 9:
            return None
        
        k = df['KDJ_K'].iloc[-1] if 'KDJ_K' in df else None
        d = df['KDJ_D'].iloc[-1] if 'KDJ_D' in df else None
        j = df['KDJ_J'].iloc[-1] if 'KDJ_J' in df else None
        signal = df['KDJ_signal'].iloc[-1] if 'KDJ_signal' in df else "NEUTRAL"
        
        if pd.isna(k):
            return None
        
        # 获取前一日K、D值用于检测
        k_prev = df['KDJ_K'].iloc[-2] if len(df) >= 2 else None
        d_prev = df['KDJ_D'].iloc[-2] if len(df) >= 2 else None
        
        if signal == "NEUTRAL":
            signal = TechnicalIndicators.detect_kdj_signal(k, d, k_prev, d_prev)
        
        return KDJResult(
            rsv=0,  # 不返回具体RSV值
            k=k,
            d=d,
            j=j,
            signal=signal
        )
    
    @staticmethod
    def get_latest_obv(df: pd.DataFrame) -> Optional[OBVResult]:
        """获取最新OBV数据"""
        if df is None or len(df) < 10:
            return None
        
        obv = df['OBV'].iloc[-1] if 'OBV' in df else None
        obv_ma = df['OBV_MA'].iloc[-1] if 'OBV_MA' in df else None
        signal = df['OBV_signal'].iloc[-1] if 'OBV_signal' in df else "NEUTRAL"
        
        if pd.isna(obv):
            return None
        
        return OBVResult(
            obv=obv,
            obv_ma=obv_ma,
            signal=signal
        )
    
    @staticmethod
    def get_latest_atr(df: pd.DataFrame) -> Optional[ATRResult]:
        """获取最新ATR数据"""
        if df is None or len(df) < 14:
            return None
        
        atr = df['ATR'].iloc[-1] if 'ATR' in df else None
        atr_ma = df['ATR_MA'].iloc[-1] if 'ATR_MA' in df else None
        signal = df['ATR_signal'].iloc[-1] if 'ATR_signal' in df else "NORMAL_VOLATILITY"
        
        if pd.isna(atr):
            return None
        
        return ATRResult(
            atr=atr,
            atr_ma=atr_ma,
            signal=signal
        )


# ==================== 便捷函数 ====================

def calculate_boll(
    close_prices: pd.Series,
    period: int = 20,
    std_dev: int = 2
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算布林带便捷函数
    
    Returns:
        Tuple[上轨, 中轨, 下轨]
    """
    return TechnicalIndicators.calculate_boll(close_prices, period, std_dev)


def calculate_kdj(
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算KDJ指标便捷函数
    
    Returns:
        Tuple[K值, D值, J值]
    """
    return TechnicalIndicators.calculate_kdj(
        high_prices, low_prices, close_prices, n, m1, m2
    )


def calculate_obv(
    close_prices: pd.Series,
    volume: pd.Series,
    period: int = 10
) -> Tuple[pd.Series, pd.Series]:
    """
    计算OBV便捷函数
    
    Returns:
        Tuple[OBV, OBV_MA]
    """
    return TechnicalIndicators.calculate_obv(close_prices, volume, period)


def calculate_atr(
    high_prices: pd.Series,
    low_prices: pd.Series,
    close_prices: pd.Series,
    period: int = 14
) -> Tuple[pd.Series, pd.Series]:
    """
    计算ATR便捷函数
    
    Returns:
        Tuple[ATR, ATR_MA]
    """
    return TechnicalIndicators.calculate_atr(
        high_prices, low_prices, close_prices, period
    )


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    np.random.seed(42)
    
    # 生成测试数据
    dates = pd.date_range(start="2024-01-01", periods=100, freq='D')
    close = 10 + np.cumsum(np.random.randn(100) * 0.5)
    high = close + np.random.rand(100)
    low = close - np.random.rand(100)
    volume = np.random.randint(1000000, 10000000, 100)
    
    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
        'volume': volume
    }, index=dates)
    
    # 计算指标
    ind = TechnicalIndicators()
    
    # BOLL
    upper, middle, lower = ind.calculate_boll(df['close'])
    print(f"BOLL: 上轨={upper.iloc[-1]:.2f}, 中轨={middle.iloc[-1]:.2f}, 下轨={lower.iloc[-1]:.2f}")
    
    # KDJ
    k, d, j = ind.calculate_kdj(df['high'], df['low'], df['close'])
    print(f"KDJ: K={k.iloc[-1]:.2f}, D={d.iloc[-1]:.2f}, J={j.iloc[-1]:.2f}")
    
    # OBV
    obv, obv_ma = ind.calculate_obv(df['close'], df['volume'])
    print(f"OBV: {obv.iloc[-1]:.0f}, MA={obv_ma.iloc[-1]:.0f}")
    
    # ATR
    atr, atr_ma = ind.calculate_atr(df['high'], df['low'], df['close'])
    print(f"ATR: {atr.iloc[-1]:.2f}, MA={atr_ma.iloc[-1]:.2f}")
