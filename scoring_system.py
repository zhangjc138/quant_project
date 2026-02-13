# -*- coding: utf-8 -*-
"""
评分系统模块 - 付费版专属功能
多维度综合评分系统

功能:
- 趋势强度评分
- 动量评分
- 波动率评分
- RSI位置评分
- MACD状态评分
- 综合评分0-100分
- 评分分级：强力买入(80+)、买入(60-80)、持有(40-60)、卖出(<40)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class SignalLevel(Enum):
    """信号等级"""
    STRONG_BUY = "强力买入"  # 80+
    BUY = "买入"             # 60-80
    HOLD = "持有"            # 40-60
    SELL = "卖出"            # <40


@dataclass
class ScoreResult:
    """评分结果"""
    total_score: float           # 综合评分 (0-100)
    signal: SignalLevel          # 信号等级
    trend_score: float           # 趋势评分 (0-25)
    momentum_score: float        # 动量评分 (0-25)
    volatility_score: float      # 波动率评分 (0-15)
    rsi_score: float             # RSI评分 (0-20)
    macd_score: float            # MACD评分 (0-15)
    scores: Dict[str, float]     # 各维度原始分数
    details: Dict[str, any]      # 详细信息
    recommendation: str          # 操作建议


class ScoringSystem:
    """
    多维度评分系统
    
    综合评估股票的技术面表现，输出0-100的综合评分
    """
    
    # 评分权重配置
    WEIGHTS = {
        'trend': 0.25,      # 趋势强度
        'momentum': 0.25,   # 动量
        'volatility': 0.15, # 波动率 (越低越好)
        'rsi': 0.20,        # RSI位置
        'macd': 0.15        # MACD状态
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        初始化评分系统
        
        Args:
            weights: 自定义权重配置
        """
        if weights:
            self.WEIGHTS = weights
        
        # 验证权重总和为1
        total = sum(self.WEIGHTS.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"权重总和必须为1.0，当前: {total}")
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """
        计算所有需要的指标
        
        Args:
            df: OHLCV数据
            
        Returns:
            指标字典
        """
        result = df.copy()
        
        # 均线
        result['ma5'] = result['close'].rolling(5).mean()
        result['ma10'] = result['close'].rolling(10).mean()
        result['ma20'] = result['close'].rolling(20).mean()
        result['ma60'] = result['close'].rolling(60).mean()
        
        # MA角度 (MA20)
        ma20 = result['ma20']
        result['ma20_angle'] = np.arctan(
            (ma20 - ma20.shift(1)) / (ma20.shift(1).replace(0, np.nan))
        ) * 180 / np.pi
        
        # 趋势强度 (价格与MA20的关系)
        result['price_above_ma20'] = (result['close'] > result['ma20']).astype(int)
        result['price_above_ma60'] = (result['close'] > result['ma60']).astype(int)
        
        # RSI
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = result['close'].ewm(span=12, adjust=False).mean()
        ema26 = result['close'].ewm(span=26, adjust=False).mean()
        result['macd_diff'] = ema12 - ema26
        result['macd_dea'] = result['macd_diff'].ewm(span=9, adjust=False).mean()
        result['macd_histogram'] = result['macd_diff'] - result['macd_dea']
        
        # 成交量变化
        result['volume_change'] = result['volume'].pct_change()
        result['volume_ma5'] = result['volume'].rolling(5).mean()
        result['volume_ratio'] = result['volume'] / result['volume_ma5']
        
        # 动量
        result['momentum_5'] = result['close'].pct_change(5)
        result['momentum_10'] = result['close'].pct_change(10)
        result['momentum_20'] = result['close'].pct_change(20)
        
        # 波动率
        result['daily_return'] = result['close'].pct_change()
        result['volatility_5'] = result['daily_return'].rolling(5).std()
        result['volatility_10'] = result['daily_return'].rolling(10).std()
        
        return result
    
    def _score_trend(self, latest: pd.Series, history: pd.DataFrame) -> Tuple[float, Dict]:
        """
        趋势强度评分 (0-25分)
        
        评估:
        - MA20角度
        - 价格与均线的位置关系
        - 均线多头排列
        """
        score = 0
        details = {}
        
        # MA20角度评分 (0-10分)
        ma20_angle = latest.get('ma20_angle', 0)
        if pd.notna(ma20_angle):
            if ma20_angle >= 5:
                angle_score = 10
            elif ma20_angle >= 3:
                angle_score = 8
            elif ma20_angle >= 1:
                angle_score = 6
            elif ma20_angle >= 0:
                angle_score = 4
            elif ma20_angle >= -2:
                angle_score = 2
            else:
                angle_score = 0
            score += angle_score
            details['ma20_angle'] = round(ma20_angle, 2)
            details['ma20_angle_score'] = angle_score
        
        # 价格与均线关系 (0-8分)
        price = latest.get('close', 0)
        ma20 = latest.get('ma20', 0)
        ma60 = latest.get('ma60', 0)
        
        if pd.notna(ma20) and pd.notna(ma60):
            above_ma20 = (price > ma20) * 1
            above_ma60 = (price > ma60) * 1
            ma20_above_60 = (ma20 > ma60) * 1
            
            # 价格位置得分
            price_score = (above_ma20 * 4) + (above_ma60 * 2) + (ma20_above_60 * 2)
            score += price_score
            details['price_above_ma20'] = bool(above_ma20)
            details['price_above_ma60'] = bool(above_ma60)
            details['ma20_above_ma60'] = bool(ma20_above_60)
            details['price_position_score'] = price_score
        
        # 短期均线方向 (0-7分)
        ma5 = latest.get('ma5', 0)
        ma10 = latest.get('ma10', 0)
        
        if pd.notna(ma5) and pd.notna(ma10):
            ma5_above_10 = (ma5 > ma10) * 1
            ma10_above_20 = (ma10 > ma20) * 1 if pd.notna(ma20) else 0
            
            short_trend_score = (ma5_above_10 * 3) + (ma10_above_20 * 4)
            score += short_trend_score
            details['ma5_above_ma10'] = bool(ma5_above_10)
            details['short_trend_score'] = short_trend_score
        
        # 限制分数范围
        score = min(score, 25)
        details['trend_score'] = round(score, 1)
        
        return score, details
    
    def _score_momentum(self, latest: pd.Series, history: pd.DataFrame) -> Tuple[float, Dict]:
        """
        动量评分 (0-25分)
        
        评估:
        - 短期涨幅
        - 动量持续性
        - 成交量确认
        """
        score = 0
        details = {}
        
        # 各周期涨幅
        momentum_5 = latest.get('momentum_5', 0)
        momentum_10 = latest.get('momentum_10', 0)
        momentum_20 = latest.get('momentum_20', 0)
        
        # 5日动量 (0-10分)
        if pd.notna(momentum_5):
            if momentum_5 >= 0.10:  # 10%以上
                mom5_score = 10
            elif momentum_5 >= 0.05:
                mom5_score = 8
            elif momentum_5 >= 0.02:
                mom5_score = 6
            elif momentum_5 >= 0:
                mom5_score = 4
            elif momentum_5 >= -0.03:
                mom5_score = 2
            else:
                mom5_score = 0
            score += mom5_score
            details['momentum_5'] = f"{momentum_5:.2%}"
            details['momentum_5_score'] = mom5_score
        
        # 10日动量 (0-8分)
        if pd.notna(momentum_10):
            if momentum_10 >= 0.15:
                mom10_score = 8
            elif momentum_10 >= 0.08:
                mom10_score = 6
            elif momentum_10 >= 0.03:
                mom10_score = 4
            elif momentum_10 >= 0:
                mom10_score = 2
            else:
                mom10_score = 0
            score += mom10_score
            details['momentum_10'] = f"{momentum_10:.2%}"
            details['momentum_10_score'] = mom10_score
        
        # 动量方向一致性 (0-7分)
        if pd.notna(momentum_5) and pd.notna(momentum_10) and pd.notna(momentum_20):
            consistent = sum([
                momentum_5 > 0,
                momentum_10 > 0,
                momentum_20 > 0
            ])
            consistency_score = consistent * 2 + 1  # 1-7分
            score += consistency_score
            details['momentum_consistency'] = consistent
            details['consistency_score'] = consistency_score
        
        # 成交量确认 (额外加分)
        volume_ratio = latest.get('volume_ratio', 0)
        if pd.notna(volume_ratio):
            if volume_ratio >= 2.0:
                volume_bonus = 3
            elif volume_ratio >= 1.5:
                volume_bonus = 2
            elif volume_ratio >= 1.2:
                volume_bonus = 1
            else:
                volume_bonus = 0
            score += volume_bonus
            details['volume_ratio'] = round(volume_ratio, 2)
            details['volume_confirm_score'] = volume_bonus
        
        # 限制分数范围
        score = min(score, 25)
        details['momentum_score'] = round(score, 1)
        
        return score, details
    
    def _score_volatility(self, latest: pd.Series, history: pd.DataFrame) -> Tuple[float, Dict]:
        """
        波动率评分 (0-15分)
        
        评估:
        - 波动率水平 (低波动率得分高)
        - 波动的稳定性
        """
        score = 0
        details = {}
        
        # 5日波动率 (0-10分)
        volatility_5 = latest.get('volatility_5', 0)
        volatility_10 = latest.get('volatility_10', 0)
        
        if pd.notna(volatility_5):
            # 假设合理波动率在 1%-5% 之间
            if volatility_5 <= 0.015:  # 1.5%
                vol_score = 10
            elif volatility_5 <= 0.025:
                vol_score = 8
            elif volatility_5 <= 0.035:
                vol_score = 6
            elif volatility_5 <= 0.05:
                vol_score = 4
            elif volatility_5 <= 0.08:
                vol_score = 2
            else:
                vol_score = 0
            score += vol_score
            details['volatility_5'] = f"{volatility_5:.2%}"
            details['volatility_5_score'] = vol_score
        
        # 波动率稳定性 (0-5分)
        if pd.notna(volatility_5) and pd.notna(volatility_10):
            vol_change = abs(volatility_5 - volatility_10) / volatility_10 if volatility_10 > 0 else 0
            if vol_change <= 0.2:
                stability_score = 5
            elif vol_change <= 0.4:
                stability_score = 3
            elif vol_change <= 0.6:
                stability_score = 1
            else:
                stability_score = 0
            score += stability_score
            details['volatility_stability'] = f"{vol_change:.2%}"
            details['stability_score'] = stability_score
        
        score = min(score, 15)
        details['volatility_score'] = round(score, 1)
        
        return score, details
    
    def _score_rsi(self, latest: pd.Series, history: pd.DataFrame) -> Tuple[float, Dict]:
        """
        RSI评分 (0-20分)
        
        评估:
        - RSI绝对水平
        - RSI趋势
        """
        score = 0
        details = {}
        
        rsi = latest.get('rsi', 50)
        
        if pd.notna(rsi):
            # RSI绝对位置 (0-12分)
            # 理想区间: 50-70 (强势但未超买)
            if 55 <= rsi <= 65:
                rsi_position_score = 12
            elif 50 <= rsi <= 70:
                rsi_position_score = 10
            elif 45 <= rsi <= 75:
                rsi_position_score = 8
            elif 40 <= rsi <= 80:
                rsi_position_score = 5
            elif rsi < 40:
                rsi_position_score = 3  # 超卖区域，反弹可能
            elif rsi > 80:
                rsi_position_score = 2  # 超买区域
            else:
                rsi_position_score = 6
            
            score += rsi_position_score
            details['rsi'] = round(rsi, 1)
            details['rsi_position_score'] = rsi_position_score
            
            # RSI方向 (0-8分)
            if len(history) >= 5:
                prev_rsi = history['rsi'].iloc[-5]
                if pd.notna(prev_rsi):
                    rsi_change = rsi - prev_rsi
                    if rsi_change >= 3:
                        rsi_trend_score = 8  # 明显上升趋势
                    elif rsi_change >= 1:
                        rsi_trend_score = 6
                    elif rsi_change >= -1:
                        rsi_trend_score = 4
                    elif rsi_change >= -3:
                        rsi_trend_score = 2
                    else:
                        rsi_trend_score = 0
                    score += rsi_trend_score
                    details['rsi_change'] = round(rsi_change, 2)
                    details['rsi_trend_score'] = rsi_trend_score
        
        score = min(score, 20)
        details['rsi_score'] = round(score, 1)
        
        return score, details
    
    def _score_macd(self, latest: pd.Series, history: pd.DataFrame) -> Tuple[float, Dict]:
        """
        MACD评分 (0-15分)
        
        评估:
        - MACD柱状图方向
        - DIF与DEA的关系
        - MACD金叉/死叉状态
        """
        score = 0
        details = {}
        
        macd_diff = latest.get('macd_diff', 0)
        macd_dea = latest.get('macd_dea', 0)
        macd_hist = latest.get('macd_histogram', 0)
        
        # MACD柱状图方向 (0-8分)
        if pd.notna(macd_hist):
            if macd_hist > 0:
                hist_score = 4
                # 正值大小
                if macd_hist > 0.5:
                    hist_score += 4
                elif macd_hist > 0.2:
                    hist_score += 2
                else:
                    hist_score += 1
            else:
                hist_score = 2
                if macd_hist < -0.5:
                    hist_score -= 1
            score += hist_score
            details['macd_histogram'] = round(macd_hist, 4)
            details['histogram_score'] = hist_score
        
        # DIF与DEA关系 (0-7分)
        if pd.notna(macd_diff) and pd.notna(macd_dea):
            if macd_diff > macd_dea:
                cross_score = 4
                # 金叉持续性
                if len(history) >= 3:
                    prev_diff = history['macd_diff'].iloc[-3]
                    prev_dea = history['macd_dea'].iloc[-3]
                    if pd.notna(prev_diff) and pd.notna(prev_dea):
                        if prev_diff > prev_dea:
                            cross_score += 3  # 持续金叉
                        else:
                            cross_score += 1  # 刚金叉
            else:
                cross_score = 0
            score += cross_score
            details['dif_vs_dea'] = 'DIF > DEA' if macd_diff > macd_dea else 'DIF < DEA'
            details['cross_score'] = cross_score
        
        score = min(score, 15)
        details['macd_score'] = round(score, 1)
        
        return score, details
    
    def calculate(self, df: pd.DataFrame) -> ScoreResult:
        """
        计算综合评分
        
        Args:
            df: OHLCV数据 (至少20行)
            
        Returns:
            ScoreResult: 评分结果
        """
        if len(df) < 20:
            raise ValueError("需要至少20行数据计算评分")
        
        # 计算所有指标
        data = self.calculate_indicators(df)
        latest = data.iloc[-1]
        history = data
        
        # 计算各维度分数
        trend_score, trend_details = self._score_trend(latest, history)
        momentum_score, momentum_details = self._score_momentum(latest, history)
        volatility_score, volatility_details = self._score_volatility(latest, history)
        rsi_score, rsi_details = self._score_rsi(latest, history)
        macd_score, macd_details = self._score_macd(latest, history)
        
        # 加权总分
        total_score = (
            trend_score * self.WEIGHTS['trend'] +
            momentum_score * self.WEIGHTS['momentum'] +
            volatility_score * self.WEIGHTS['volatility'] +
            rsi_score * self.WEIGHTS['rsi'] +
            macd_score * self.WEIGHTS['macd']
        )
        total_score = round(total_score, 1)
        
        # 确定信号等级
        if total_score >= 80:
            signal = SignalLevel.STRONG_BUY
        elif total_score >= 60:
            signal = SignalLevel.BUY
        elif total_score >= 40:
            signal = SignalLevel.HOLD
        else:
            signal = SignalLevel.SELL
        
        # 生成操作建议
        recommendation = self._generate_recommendation(
            total_score, trend_score, momentum_score, 
            rsi_score, macd_score, latest
        )
        
        # 收集所有详细信息
        details = {
            'ma20_angle': latest.get('ma20_angle', 0),
            'momentum_5': latest.get('momentum_5', 0),
            'momentum_10': latest.get('momentum_10', 0),
            'volatility_5': latest.get('volatility_5', 0),
            'rsi': latest.get('rsi', 50),
            'macd_histogram': latest.get('macd_histogram', 0),
            'price_above_ma20': bool(latest.get('price_above_ma20', 0)),
            'volume_ratio': latest.get('volume_ratio', 1),
            **trend_details,
            **momentum_details,
            **volatility_details,
            **rsi_details,
            **macd_details
        }
        
        scores = {
            'trend': trend_score,
            'momentum': momentum_score,
            'volatility': volatility_score,
            'rsi': rsi_score,
            'macd': macd_score,
            'total': total_score
        }
        
        return ScoreResult(
            total_score=total_score,
            signal=signal,
            trend_score=trend_score,
            momentum_score=momentum_score,
            volatility_score=volatility_score,
            rsi_score=rsi_score,
            macd_score=macd_score,
            scores=scores,
            details=details,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, total: float, trend: float, 
                                  momentum: float, rsi: float, 
                                  macd: float, latest: pd.Series) -> str:
        """生成操作建议"""
        parts = []
        
        if total >= 80:
            parts.append("✅ 强烈看涨，技术面表现优异")
        elif total >= 60:
            parts.append("🟢 温和看涨，可以关注")
        elif total >= 40:
            parts.append("🟡 建议观望，等待更明确信号")
        else:
            parts.append("🔴 建议回避或减仓")
        
        # 趋势建议
        if trend >= 20:
            parts.append("趋势强劲")
        elif trend < 10:
            parts.append("趋势偏弱")
        
        # 动量建议
        if momentum >= 20:
            parts.append("动量充足")
        elif momentum < 10:
            parts.append("动量不足")
        
        # RSI建议
        rsi_val = latest.get('rsi', 50)
        if rsi_val > 75:
            parts.append("注意RSI超买风险")
        elif rsi_val < 35:
            parts.append("RSI超卖，可能有反弹机会")
        
        return " | ".join(parts)
    
    def batch_score(self, stock_data: Dict[str, pd.DataFrame]) -> List[ScoreResult]:
        """
        批量评分
        
        Args:
            stock_data: 股票代码到数据的映射
            
        Returns:
            评分结果列表 (按分数降序)
        """
        results = []
        for symbol, df in stock_data.items():
            try:
                score_result = self.calculate(df)
                score_result.details['symbol'] = symbol
                results.append(score_result)
            except Exception as e:
                print(f"评分失败 {symbol}: {e}")
        
        # 按总分排序
        results.sort(key=lambda x: x.total_score, reverse=True)
        return results
    
    def get_top_stocks(self, stock_data: Dict[str, pd.DataFrame], 
                        top_n: int = 10, 
                        min_score: float = 50) -> List[ScoreResult]:
        """
        获取评分最高的股票
        
        Args:
            stock_data: 股票数据
            top_n: 返回前N只
            min_score: 最低评分门槛
            
        Returns:
            高分股票列表
        """
        results = self.batch_score(stock_data)
        filtered = [r for r in results if r.total_score >= min_score]
        return filtered[:top_n]


def print_score_result(result: ScoreResult, symbol: str = ""):
    """打印评分结果"""
    print(f"\n{'='*60}")
    if symbol:
        print(f"股票: {symbol}")
    print(f"{'='*60}")
    
    print(f"\n📊 综合评分: {result.total_score:.1f}/100")
    print(f"   信号: {result.signal.value}")
    
    print(f"\n📈 各维度评分:")
    print(f"   趋势强度: {result.trend_score:.1f}/25")
    print(f"   动量:     {result.momentum_score:.1f}/25")
    print(f"   波动率:   {result.volatility_score:.1f}/15")
    print(f"   RSI位置:  {result.rsi_score:.1f}/20")
    print(f"   MACD状态: {result.macd_score:.1f}/15")
    
    print(f"\n💡 操作建议: {result.recommendation}")
    
    print(f"\n📌 关键指标:")
    details = result.details
    
    # 安全解析数值的辅助函数
    def parse_number(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return 0
        if isinstance(val, str):
            return float(val.replace('%', '').replace('°', ''))
        return float(val)
    
    ma20_angle = parse_number(details.get('ma20_angle', 0))
    momentum_5 = parse_number(details.get('momentum_5', 0))
    rsi = parse_number(details.get('rsi', 50))
    macd_hist = parse_number(details.get('macd_histogram', 0))
    vol_ratio = parse_number(details.get('volume_ratio', 1))
    
    print(f"   MA20角度: {ma20_angle:.2f}°")
    print(f"   5日涨幅: {momentum_5:.2%}")
    print(f"   RSI(14): {rsi:.1f}")
    print(f"   MACD柱: {macd_hist:.4f}")
    print(f"   成交量比: {vol_ratio:.2f}")


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("评分系统测试")
    print("=" * 60)
    
    # 模拟数据
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # 创建模拟价格数据
    close = 100 + np.cumsum(np.random.randn(100) * 0.3)
    open_ = close - np.random.randn(100) * 0.1
    high = close + np.abs(np.random.randn(100) * 0.2)
    low = close - np.abs(np.random.randn(100) * 0.2)
    volume = np.random.randint(1000000, 10000000, 100)
    
    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)
    
    # 计算评分
    scoring = ScoringSystem()
    result = scoring.calculate(df)
    
    # 打印结果
    print_score_result(result, "测试股票")
    
    # 批量评分测试
    print("\n\n" + "=" * 60)
    print("批量评分测试")
    print("=" * 60)
    
    # 创建多只股票数据
    stocks = {}
    for i in range(5):
        close = 50 + i*10 + np.cumsum(np.random.randn(100) * 0.4)
        volume = np.random.randint(1000000, 10000000, 100)
        df = pd.DataFrame({
            'open': close - np.random.randn(100) * 0.1,
            'high': close + np.abs(np.random.randn(100) * 0.2),
            'low': close - np.abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': volume
        }, index=dates)
        stocks[f"60000{i}"] = df
    
    results = scoring.batch_score(stocks)
    
    print("\n🏆 评分排名:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r.details.get('symbol', 'Unknown')}: {r.total_score:.1f}分 - {r.signal.value}")
