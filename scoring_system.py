#!/usr/bin/env python3
"""
综合评分系统模块（开源版）

提供股票综合评分功能
基于技术指标打分，无需付费依赖
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class SignalLevel(Enum):
    """信号级别"""
    STRONG_BUY = "强力买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强力卖出"


@dataclass
class ScoreResult:
    """评分结果"""
    total_score: float          # 综合评分 (0-100)
    signal: SignalLevel        # 信号级别
    recommendation: str        # 建议
    scores: Dict[str, float]   # 各维度分数
    details: Dict[str, float]  # 详情数据
    trend_score: float         # 趋势分数
    momentum_score: float      # 动量分数
    volatility_score: float   # 波动率分数
    rsi_score: float          # RSI分数
    macd_score: float         # MACD分数


class ScoringSystem:
    """综合评分系统（开源版）"""
    
    def __init__(self):
        """初始化评分系统"""
        # 评分权重
        self.weights = {
            'trend': 0.25,      # 趋势强度
            'momentum': 0.25,   # 动量
            'volatility': 0.15,  # 波动率
            'rsi': 0.20,       # RSI
            'macd': 0.15       # MACD
        }
        
        # MA周期
        self.ma_periods = [5, 10, 20, 60]
    
    def calculate(self, df: pd.DataFrame, symbol: str = "stock") -> ScoreResult:
        """
        计算综合评分
        
        Args:
            df: 股票数据
            symbol: 股票代码
        
        Returns:
            ScoreResult: 评分结果
        """
        if df is None or len(df) < 60:
            return self._default_result()
        
        df = df.copy()
        
        # 计算各项指标
        df = self._calculate_indicators(df)
        
        latest = df.iloc[-1]
        
        # 计算各维度分数
        trend_score = self._calculate_trend_score(latest, df)
        momentum_score = self._calculate_momentum_score(latest, df)
        volatility_score = self._calculate_volatility_score(latest, df)
        rsi_score = self._calculate_rsi_score(latest)
        macd_score = self._calculate_macd_score(latest)
        
        # 计算综合评分
        total_score = (
            trend_score * self.weights['trend'] +
            momentum_score * self.weights['momentum'] +
            volatility_score * self.weights['volatility'] +
            rsi_score * self.weights['rsi'] +
            macd_score * self.weights['macd']
        )
        
        # 确定信号
        signal = self._get_signal(total_score)
        
        # 生成建议
        recommendation = self._get_recommendation(total_score, signal)
        
        # 详情
        details = {
            'ma20_angle': latest.get('ma20_angle', 0),
            'ma20': latest.get('ma20', 0),
            'rsi': latest.get('rsi', 50),
            'macd': latest.get('macd', 0),
            'momentum_5': latest.get('momentum_5', 0),
            'momentum_10': latest.get('momentum_10', 0),
            'momentum_20': latest.get('momentum_20', 0),
            'volume_ratio': latest.get('volume_ratio', 1),
            'volatility': latest.get('volatility', 0),
        }
        
        return ScoreResult(
            total_score=total_score,
            signal=signal,
            recommendation=recommendation,
            scores={
                '趋势强度': trend_score,
                '动量': momentum_score,
                '波动率': volatility_score,
                'RSI位置': rsi_score,
                'MACD状态': macd_score
            },
            details=details,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            rsi_score=rsi_score,
            macd_score=macd_score
        )
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        # 移动平均线
        for period in self.ma_periods:
            df[f'ma{period}'] = df['close'].rolling(period).mean()
        
        # MA20 角度
        df['ma20_1'] = df['ma20'].shift(1)
        df['ma20_angle'] = np.arctan(
            (df['ma20'] - df['ma20_1']) / (df['ma20_1'] + 0.001)
        ) * 180 / np.pi
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / (avg_loss + 0.001)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 动量
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # 成交量
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma5'] + 1)
        
        # 波动率
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        return df
    
    def _calculate_trend_score(self, latest: pd.Series, df: pd.DataFrame) -> float:
        """计算趋势分数"""
        score = 50  # 基础分
        
        # MA20 角度 (0-25分)
        ma20_angle = latest.get('ma20_angle', 0)
        if pd.notna(ma20_angle):
            if ma20_angle > 5:
                score += 25
            elif ma20_angle > 3:
                score += 20
            elif ma20_angle > 1:
                score += 15
            elif ma20_angle > 0:
                score += 10
            else:
                score -= 10
        
        # 价格与MA20关系 (0-25分)
        price = latest.get('close', 0)
        ma20 = latest.get('ma20', price)
        if pd.notna(ma20) and ma20 > 0:
            price_above_ma = (price - ma20) / ma20
            if price_above_ma > 0.05:
                score += 25
            elif price_above_ma > 0:
                score += 15
            elif price_above_ma > -0.05:
                score += 5
        
        return min(100, max(0, score))
    
    def _calculate_momentum_score(self, latest: pd.Series, df: pd.DataFrame) -> float:
        """计算动量分数"""
        score = 50  # 基础分
        
        # 5日动量 (0-25分)
        mom5 = latest.get('momentum_5', 0)
        if pd.notna(mom5):
            if mom5 > 0.05:
                score += 25
            elif mom5 > 0.02:
                score += 20
            elif mom5 > 0:
                score += 10
            else:
                score -= max(0, abs(mom5) * 100)
        
        # 10日动量 (0-25分)
        mom10 = latest.get('momentum_10', 0)
        if pd.notna(mom10):
            if mom10 > 0.08:
                score += 25
            elif mom10 > 0.03:
                score += 15
            elif mom10 > 0:
                score += 5
            else:
                score -= max(0, abs(mom10) * 50)
        
        return min(100, max(0, score))
    
    def _calculate_volatility_score(self, latest: pd.Series, df: pd.DataFrame) -> float:
        """计算波动率分数（低波动更好）"""
        score = 50  # 基础分
        
        volatility = latest.get('volatility', 0.02)
        if pd.notna(volatility):
            # 低波动加分
            if volatility < 0.02:
                score += 30
            elif volatility < 0.03:
                score += 20
            elif volatility < 0.05:
                score += 10
            else:
                score -= (volatility - 0.05) * 200
        
        # 成交量放大
        vol_ratio = latest.get('volume_ratio', 1)
        if pd.notna(vol_ratio):
            if 0.8 <= vol_ratio <= 2:
                score += 20
            elif vol_ratio > 3:
                score -= 20
        
        return min(100, max(0, score))
    
    def _calculate_rsi_score(self, latest: pd.Series) -> float:
        """计算RSI分数"""
        score = 50  # 基础分
        
        rsi = latest.get('rsi', 50)
        if pd.notna(rsi):
            # RSI 在 40-60 之间较好
            if 40 <= rsi <= 60:
                score += 30
            elif 30 <= rsi < 40:
                score += 20
            elif 60 < rsi <= 70:
                score += 20
            elif rsi < 30:
                score -= 20
            elif rsi > 70:
                score -= 20
        
        return min(100, max(0, score))
    
    def _calculate_macd_score(self, latest: pd.Series) -> float:
        """计算MACD分数"""
        score = 50  # 基础分
        
        macd = latest.get('macd', 0)
        macd_hist = latest.get('macd_hist', 0)
        
        if pd.notna(macd) and pd.notna(macd_hist):
            # MACD 在零轴上方且柱状图为正
            if macd > 0 and macd_hist > 0:
                score += 30
            elif macd > 0:
                score += 20
            elif macd_hist > 0:
                score += 15
            elif macd < 0 and macd_hist < 0:
                score -= 20
        
        return min(100, max(0, score))
    
    def _get_signal(self, total_score: float) -> SignalLevel:
        """根据总分确定信号"""
        if total_score >= 80:
            return SignalLevel.STRONG_BUY
        elif total_score >= 60:
            return SignalLevel.BUY
        elif total_score >= 40:
            return SignalLevel.HOLD
        elif total_score >= 25:
            return SignalLevel.SELL
        else:
            return SignalLevel.STRONG_SELL
    
    def _get_recommendation(self, total_score: float, signal: SignalLevel) -> str:
        """生成建议"""
        if signal == SignalLevel.STRONG_BUY:
            return "多个技术指标显示强势，建议重点关注，可考虑建仓"
        elif signal == SignalLevel.BUY:
            return "技术指标偏正面，可考虑逢低买入"
        elif signal == SignalLevel.HOLD:
            return "技术指标中性，建议观望或持有现有仓位"
        elif signal == SignalLevel.SELL:
            return "技术指标偏弱，建议减仓或观望"
        else:
            return "技术指标显示弱势，建议回避或止损"
    
    def _default_result(self) -> ScoreResult:
        """返回默认结果"""
        return ScoreResult(
            total_score=50,
            signal=SignalLevel.HOLD,
            recommendation="数据不足，无法评分",
            scores={
                '趋势强度': 0,
                '动量': 0,
                '波动率': 0,
                'RSI位置': 0,
                'MACD状态': 0
            },
            details={},
            trend_score=0,
            momentum_score=0,
            volatility_score=0,
            rsi_score=0,
            macd_score=0
        )


def print_score_result(result: ScoreResult):
    """打印评分结果"""
    print("\n" + "="*60)
    print("📊 综合评分结果")
    print("="*60)
    print(f"综合评分: {result.total_score:.1f}/100")
    print(f"信号: {result.signal.value}")
    print(f"建议: {result.recommendation}")
    print("-"*60)
    print("各维度分数:")
    for k, v in result.scores.items():
        print(f"  {k}: {v:.1f}")
    print("="*60)


if __name__ == "__main__":
    # 测试
    print("评分系统测试")
    
    # 创建模拟数据
    dates = pd.date_range(start="2024-01-01", periods=200, freq="D")
    np.random.seed(42)
    
    prices = 100 + np.cumsum(np.random.randn(200) * 0.3 + 0.02)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices,
        'open': prices * (1 + np.random.randn(200) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(200) * 0.02)),
        'low': prices * (1 - np.abs(np.random.randn(200) * 0.02)),
        'volume': np.random.randint(1000000, 10000000, 200),
    })
    
    # 计算评分
    scoring = ScoringSystem()
    result = scoring.calculate(df, "600519")
    
    # 打印结果
    print_score_result(result)
    
    print("\n评分详情:")
    print(f"  MA20角度: {result.details.get('ma20_angle', 0):.2f}°")
    print(f"  RSI: {result.details.get('rsi', 50):.1f}")
    print(f"  5日涨幅: {result.details.get('momentum_5', 0):.2%}")
