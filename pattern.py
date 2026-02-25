#!/usr/bin/env python3
"""
K线形态识别模块

提供常见K线形态的识别功能
支持底部形态、顶部形态、突破形态等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class PatternType(Enum):
    """形态类型"""
    # 底部反转
    HAMMER = "hammer"           # 锤子线
    INVERSE_HAMMER = "inv_hammer"  # 倒锤子线
    MORNING_STAR = "morning_star"  # 晨星
    THREE_WHITE_SOLDIERS = "three_white"  # 三白兵
    BULLISH_ENGULFING = "bull_engulf"  # 阳包阴
    PIERCING_LINE = "piercing"   # 刺透形态
    
    # 顶部反转
    SHOOTING_STAR = "shooting_star"  # 流星线
    HANGING_MAN = "hanging_man"  # 吊颈线
    EVENING_STAR = "evening_star"  # 暮星
    THREE_BLACK_CROWS = "three_black"  # 三黑鸦
    BEARISH_ENGULFING = "bear_engulf"  # 阴包阳
    DARK_CLOUD_COVER = "dark_cloud"  # 乌云盖顶
    
    # 持续形态
    THREE_METHODS = "three_methods"  # 三法形态
    RISING_THREE = "rising_three"  # 上升三法
    FALLING_THREE = "falling_three"  # 下降三法
    
    # 突破形态
    BREAKOUT_UP = "breakout_up"   # 向上突破
    BREAKOUT_DOWN = "breakout_down"  # 向下突破


@dataclass
class PatternSignal:
    """形态信号"""
    pattern_type: PatternType
    pattern_name: str
    confidence: float          # 置信度 0-1
    direction: str            # "bullish", "bearish", "neutral"
    description: str          # 描述


class CandlePatternRecognizer:
    """K线形态识别器"""
    
    def __init__(self, body_ratio: float = 0.3, shadow_ratio: float = 0.1):
        """
        初始化识别器
        
        Args:
            body_ratio: 实体占比阈值（用于判断大小）
            shadow_ratio: 影线占比阈值
        """
        self.body_ratio = body_ratio
        self.shadow_ratio = shadow_ratio
    
    def recognize(self, df: pd.DataFrame) -> List[PatternSignal]:
        """
        识别K线形态
        
        Args:
            df: 包含OHLC数据的DataFrame
            
        Returns:
            List[PatternSignal]: 形态信号列表
        """
        if len(df) < 5:
            return []
        
        signals = []
        
        # 单根K线形态
        for i in range(1, len(df)):
            candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            
            # 锤子线
            if self._is_hammer(candle):
                signals.append(PatternSignal(
                    pattern_type=PatternType.HAMMER,
                    pattern_name="锤子线",
                    confidence=0.7,
                    direction="bullish",
                    description="下影线较长，上影线很短或没有，实体在下部"
                ))
            
            # 倒锤子线
            if self._is_inverse_hammer(candle):
                signals.append(PatternSignal(
                    pattern_type=PatternType.INVERSE_HAMMER,
                    pattern_name="倒锤子线",
                    confidence=0.7,
                    direction="bullish",
                    description="上影线较长，下影线很短或没有，实体在上部"
                ))
            
            # 流星线
            if self._is_shooting_star(candle):
                signals.append(PatternSignal(
                    pattern_type=PatternType.SHOOTING_STAR,
                    pattern_name="流星线",
                    confidence=0.7,
                    direction="bearish",
                    description="上影线较长，下影线很短或没有，实体在下部"
                ))
            
            # 吊颈线
            if self._is_hanging_man(candle):
                signals.append(PatternSignal(
                    pattern_type=PatternType.HANGING_MAN,
                    pattern_name="吊颈线",
                    confidence=0.7,
                    direction="bearish",
                    description="下影线较长，上影线很短或没有，实体在上部"
                ))
        
        # 多根K线形态
        if len(df) >= 3:
            # 晨星
            if self._is_morning_star(df.iloc[-3:]):
                signals.append(PatternSignal(
                    pattern_type=PatternType.MORNING_STAR,
                    pattern_name="晨星",
                    confidence=0.8,
                    direction="bullish",
                    description="下跌趋势后，出现小阴小阳星，随后出现大阳线"
                ))
            
            # 暮星
            if self._is_evening_star(df.iloc[-3:]):
                signals.append(PatternSignal(
                    pattern_type=PatternType.EVENING_STAR,
                    pattern_name="暮星",
                    confidence=0.8,
                    direction="bearish",
                    description="上涨趋势后，出现小阴小阳星，随后出现大阴线"
                ))
            
            # 三白兵
            if self._is_three_white_soldiers(df.iloc[-3:]):
                signals.append(PatternSignal(
                    pattern_type=PatternType.THREE_WHITE_SOLDIERS,
                    pattern_name="三白兵",
                    confidence=0.8,
                    direction="bullish",
                    description="连续三天出现中到大阳线，逐日创新高"
                ))
            
            # 三黑鸦
            if self._is_three_black_crows(df.iloc[-3:]):
                signals.append(PatternSignal(
                    pattern_type=PatternType.THREE_BLACK_CROWS,
                    pattern_name="三黑鸦",
                    confidence=0.8,
                    direction="bearish",
                    description="连续三天出现中到大阴线，逐日创新低"
                ))
            
            # 阳包阴
            if self._is_bullish_engulfing(df.iloc[-2:]):
                signals.append(PatternSignal(
                    pattern_type=PatternType.BULLISH_ENGULFING,
                    pattern_name="阳包阴",
                    confidence=0.8,
                    direction="bullish",
                    description="第二天阳线实体完全覆盖第一天阴线实体"
                ))
            
            # 阴包阳
            if self._is_bearish_engulfing(df.iloc[-2:]):
                signals.append(PatternSignal(
                    pattern_type=PatternType.BEARISH_ENGULFING,
                    pattern_name="阴包阳",
                    confidence=0.8,
                    direction="bearish",
                    description="第二天阴线实体完全覆盖第一天阳线实体"
                ))
            
            # 刺透形态
            if self._is_piercing_line(df.iloc[-2:]):
                signals.append(PatternSignal(
                    pattern_type=PatternType.PIERCING_LINE,
                    pattern_name="刺透形态",
                    confidence=0.7,
                    direction="bullish",
                    description="第二天阳线实体插入第一天阴线实体50%以上"
                ))
            
            # 乌云盖顶
            if self._is_dark_cloud_cover(df.iloc[-2:]):
                signals.append(PatternSignal(
                    pattern_type=PatternType.DARK_CLOUD_COVER,
                    pattern_name="乌云盖顶",
                    confidence=0.7,
                    direction="bearish",
                    description="第二天阴线实体覆盖第一天阳线实体50%以上"
                ))
        
        return signals
    
    def _get_candle_attributes(self, candle: pd.Series) -> Dict:
        """获取K线属性"""
        body = abs(candle['close'] - candle['open'])
        body_size = body / candle['open']
        
        upper_shadow = candle['high'] - max(candle['open'], candle['close'])
        lower_shadow = min(candle['open'], candle['close']) - candle['low']
        
        total_range = candle['high'] - candle['low']
        if total_range == 0:
            total_range = 1
        
        return {
            'body': body,
            'body_size': body_size,
            'upper_shadow': upper_shadow,
            'lower_shadow': lower_shadow,
            'upper_shadow_ratio': upper_shadow / total_range,
            'lower_shadow_ratio': lower_shadow / total_range,
            'is_bullish': candle['close'] > candle['open'],
            'is_bearish': candle['close'] < candle['open'],
            'total_range': total_range
        }
    
    def _is_hammer(self, candle: pd.Series) -> bool:
        """判断锤子线"""
        attrs = self._get_candle_attributes(candle)
        
        # 实体较小
        if attrs['body_size'] > self.body_ratio:
            return False
        
        # 下影线较长
        if attrs['lower_shadow_ratio'] < 0.5:
            return False
        
        # 上影线很短或没有
        if attrs['upper_shadow_ratio'] > 0.1:
            return False
        
        return True
    
    def _is_inverse_hammer(self, candle: pd.Series) -> bool:
        """判断倒锤子线"""
        attrs = self._get_candle_attributes(candle)
        
        # 实体较小
        if attrs['body_size'] > self.body_ratio:
            return False
        
        # 上影线较长
        if attrs['upper_shadow_ratio'] < 0.5:
            return False
        
        # 下影线很短
        if attrs['lower_shadow_ratio'] > 0.1:
            return False
        
        return True
    
    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """判断流星线"""
        attrs = self._get_candle_attributes(candle)
        
        # 实体较小
        if attrs['body_size'] > self.body_ratio:
            return False
        
        # 上影线较长
        if attrs['upper_shadow_ratio'] < 0.5:
            return False
        
        # 下影线很短
        if attrs['lower_shadow_ratio'] > 0.1:
            return False
        
        return True
    
    def _is_hanging_man(self, candle: pd.Series) -> bool:
        """判断吊颈线"""
        attrs = self._get_candle_attributes(candle)
        
        # 实体较小
        if attrs['body_size'] > self.body_ratio:
            return False
        
        # 下影线较长
        if attrs['lower_shadow_ratio'] < 0.5:
            return False
        
        # 上影线很短
        if attrs['upper_shadow_ratio'] > 0.1:
            return False
        
        return True
    
    def _is_morning_star(self, candles: pd.DataFrame) -> bool:
        """判断晨星（三根K线）"""
        if len(candles) != 3:
            return False
        
        c1, c2, c3 = candles.iloc[0], candles.iloc[1], candles.iloc[2]
        
        # 第一根：阴线（下跌趋势）
        if c3['close'] >= c3['open']:
            return False
        
        # 第二根：小实体星线
        body2 = abs(c2['close'] - c2['open'])
        body2_size = body2 / c2['open']
        if body2_size > self.body_ratio:
            return False
        
        # 第三根：阳线，收盘在第一天实体50%以上
        if c1['close'] < c1['open']:  # 第一天是阴线
            body1 = c1['open'] - c1['close']
            if body1 == 0:
                return False
            
            penetration = (c3['open'] - c1['close']) / body1
            if penetration < 0.5:
                return False
        
        return True
    
    def _is_evening_star(self, candles: pd.DataFrame) -> bool:
        """判断暮星（三根K线）"""
        if len(candles) != 3:
            return False
        
        c1, c2, c3 = candles.iloc[0], candles.iloc[1], candles.iloc[2]
        
        # 第一根：阳线（上涨趋势）
        if c3['close'] <= c3['open']:
            return False
        
        # 第二根：小实体星线
        body2 = abs(c2['close'] - c2['open'])
        body2_size = body2 / c2['open']
        if body2_size > self.body_ratio:
            return False
        
        # 第三根：阴线，收盘在第一天实体50%以上
        if c1['close'] > c1['open']:  # 第一天是阳线
            body1 = c1['close'] - c1['open']
            if body1 == 0:
                return False
            
            penetration = (c1['open'] - c3['close']) / body1
            if penetration < 0.5:
                return False
        
        return True
    
    def _is_three_white_soldiers(self, candles: pd.DataFrame) -> bool:
        """判断三白兵（三根阳线）"""
        if len(candles) != 3:
            return False
        
        for i, candle in enumerate(candles.iloc):
            # 必须是阳线
            if candle['close'] <= candle['open']:
                return False
            
            # 实体不能太小
            body = candle['close'] - candle['open']
            body_size = body / candle['open']
            if body_size < 0.01:
                return False
            
            # 影线不能太长
            upper_shadow = candle['high'] - candle['close']
            lower_shadow = candle['open'] - candle['low']
            total_range = candle['high'] - candle['low']
            if total_range == 0:
                return False
            
            if upper_shadow / total_range > 0.3 or lower_shadow / total_range > 0.3:
                return False
            
            # 逐日创新高
            if i > 0 and candle['close'] <= candles.iloc[i-1]['close']:
                return False
        
        return True
    
    def _is_three_black_crows(self, candles: pd.DataFrame) -> bool:
        """判断三黑鸦（三根阴线）"""
        if len(candles) != 3:
            return False
        
        for i, candle in enumerate(candles.iloc):
            # 必须是阴线
            if candle['close'] >= candle['open']:
                return False
            
            # 实体不能太小
            body = candle['open'] - candle['close']
            body_size = body / candle['open']
            if body_size < 0.01:
                return False
            
            # 影线不能太长
            upper_shadow = candle['high'] - candle['open']
            lower_shadow = candle['low'] - candle['close']
            total_range = candle['high'] - candle['low']
            if total_range == 0:
                return False
            
            if upper_shadow / total_range > 0.3 or lower_shadow / total_range > 0.3:
                return False
            
            # 逐日创新低
            if i > 0 and candle['close'] >= candles.iloc[i-1]['close']:
                return False
        
        return True
    
    def _is_bullish_engulfing(self, candles: pd.DataFrame) -> bool:
        """判断阳包阴"""
        if len(candles) != 2:
            return False
        
        c1, c2 = candles.iloc[0], candles.iloc[1]
        
        # 第一根阴线
        if c1['close'] >= c1['open']:
            return False
        
        # 第二根阳线
        if c2['close'] <= c2['open']:
            return False
        
        # 阳线实体覆盖阴线实体
        body1 = c1['open'] - c1['close']
        body2 = c2['close'] - c2['open']
        
        if body1 == 0 or body2 == 0:
            return False
        
        # 开盘价低于前一天收盘价，收盘价高于前一天开盘价
        if c2['open'] >= c1['close'] or c2['close'] <= c1['open']:
            return False
        
        return True
    
    def _is_bearish_engulfing(self, candles: pd.DataFrame) -> bool:
        """判断阴包阳"""
        if len(candles) != 2:
            return False
        
        c1, c2 = candles.iloc[0], candles.iloc[1]
        
        # 第一根阳线
        if c1['close'] <= c1['open']:
            return False
        
        # 第二根阴线
        if c2['close'] >= c2['open']:
            return False
        
        # 阴线实体覆盖阳线实体
        body1 = c1['close'] - c1['open']
        body2 = c2['open'] - c2['close']
        
        if body1 == 0 or body2 == 0:
            return False
        
        # 开盘价高于前一天收盘价，收盘价低于前一天开盘价
        if c2['open'] <= c1['close'] or c2['close'] >= c1['open']:
            return False
        
        return True
    
    def _is_piercing_line(self, candles: pd.DataFrame) -> bool:
        """判断刺透形态"""
        if len(candles) != 2:
            return False
        
        c1, c2 = candles.iloc[0], candles.iloc[1]
        
        # 第一根阴线
        if c1['close'] >= c1['open']:
            return False
        
        # 第二根阳线
        if c2['close'] <= c2['open']:
            return False
        
        # 阳线插入阴线实体50%以上
        body1 = c1['open'] - c1['close']
        if body1 == 0:
            return False
        
        penetration = (c2['open'] - c1['close']) / body1
        if penetration < 0.5:
            return False
        
        # 收盘价不超过第一天开盘价
        if c2['close'] > c1['open']:
            return False
        
        return True
    
    def _is_dark_cloud_cover(self, candles: pd.DataFrame) -> bool:
        """判断乌云盖顶"""
        if len(candles) != 2:
            return False
        
        c1, c2 = candles.iloc[0], candles.iloc[1]
        
        # 第一根阳线
        if c1['close'] <= c1['open']:
            return False
        
        # 第二根阴线
        if c2['close'] >= c2['open']:
            return False
        
        # 阴线覆盖阳线实体50%以上
        body1 = c1['close'] - c1['open']
        if body1 == 0:
            return False
        
        penetration = (c1['close'] - c2['open']) / body1
        if penetration < 0.5:
            return False
        
        # 收盘价超过第一天收盘价
        if c2['close'] < c1['close']:
            return False
        
        return True


def calculate_pattern_score(df: pd.DataFrame) -> Dict:
    """
    计算K线形态综合评分
    
    Args:
        df: OHLC数据
        
    Returns:
        Dict: 评分结果
    """
    recognizer = CandlePatternRecognizer()
    patterns = recognizer.recognize(df)
    
    # 统计
    bullish_count = sum(1 for p in patterns if p.direction == "bullish")
    bearish_count = sum(1 for p in patterns if p.direction == "bearish")
    
    # 计算分数
    score = 50  # 基础分
    
    # 加权计算
    for pattern in patterns:
        if pattern.direction == "bullish":
            score += pattern.confidence * 10
        else:
            score -= pattern.confidence * 10
    
    score = max(0, min(100, score))
    
    return {
        "score": round(score, 1),
        "patterns": [(p.pattern_name, p.direction, p.confidence) for p in patterns],
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "overall": "bullish" if bullish_count > bearish_count else "bearish" if bearish_count > bullish_count else "neutral"
    }


if __name__ == "__main__":
    # 测试
    print("K线形态识别模块测试")
    
    # 创建测试数据
    import numpy as np
    
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    
    # 模拟一些K线数据（包含锤子线形态）
    test_data = {
        'open': [100, 102, 101, 99, 98, 100, 102, 103, 104, 105],
        'close': [102, 101, 99, 98, 100, 103, 103, 104, 105, 106],
        'high': [103, 103, 102, 100, 100, 104, 104, 105, 106, 107],
        'low': [99, 100, 98, 97, 97, 99, 101, 102, 103, 104]
    }
    
    df = pd.DataFrame(test_data, index=dates)
    
    recognizer = CandlePatternRecognizer()
    patterns = recognizer.recognize(df)
    
    print(f"\n识别到 {len(patterns)} 个形态:")
    for p in patterns:
        print(f"  - {p.pattern_name} ({p.direction}, 置信度: {p.confidence})")
        print(f"    {p.description}")
    
    # 计算综合评分
    score_result = calculate_pattern_score(df)
    print(f"\n综合评分: {score_result['score']}")
    print(f"总体判断: {score_result['overall']}")
