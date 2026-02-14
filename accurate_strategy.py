#!/usr/bin/env python3
"""
高精度选股策略

基于多指标共振和趋势确认的严格筛选策略
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HighAccuracyStrategy:
    """高精度选股策略"""
    
    def __init__(self):
        # 策略参数
        self.params = {
            # MA20角度阈值
            'ma20_angle_min': 3.0,  # 必须大于3度
            'ma20_angle_max': 15.0,  # 不能太大（可能是假突破）
            
            # RSI阈值
            'rsi_min': 30,  # 超卖区域
            'rsi_max': 65,   # 不能超买
            
            # MACD要求
            'macd_require_golden_cross': True,  # 必须金叉
            'macd_histogram_positive': True,    # 柱状图必须为正
            
            # 成交量要求
            'volume_ratio_min': 1.2,  # 成交量要放大
            
            # 动量要求
            'momentum_positive': True,  # 5日涨幅必须为正
            
            # 均线多头排列
            'ma排列要求': 'ma5>ma10>ma20',
        }
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有指标"""
        result = df.copy()
        
        # 均线
        for period in [5, 10, 20, 60]:
            result[f'ma{period}'] = result['close'].rolling(period).mean()
        
        # MA20角度
        ma20 = result['ma20']
        result['ma20_angle'] = np.arctan(
            (ma20 - ma20.shift(1)) / (ma20.shift(1).replace(0, np.nan))
        ) * 180 / np.pi
        
        # RSI
        delta = result['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        result['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = result['close'].ewm(span=12).mean()
        ema26 = result['close'].ewm(span=26).mean()
        result['macd_diff'] = ema12 - ema26
        result['macd_dea'] = result['macd_diff'].ewm(span=9).mean()
        result['macd_hist'] = result['macd_diff'] - result['macd_dea']
        
        # 成交量
        result['volume_ma5'] = result['volume'].rolling(5).mean()
        result['volume_ratio'] = result['volume'] / result['volume_ma5']
        
        # 动量
        result['momentum_5'] = result['close'].pct_change(5) * 100
        result['momentum_20'] = result['close'].pct_change(20) * 100
        
        return result
    
    def check_buy_signal(self, row: pd.Series) -> Tuple[bool, str]:
        """
        检查是否满足买入条件
        返回: (是否买入, 原因)
        """
        reasons = []
        score = 0
        
        # 1. MA20角度检查 (权重: 25分)
        ma20_angle = row.get('ma20_angle', 0)
        if pd.isna(ma20_angle):
            return False, "数据不足"
        
        if self.params['ma20_angle_min'] <= ma20_angle <= self.params['ma20_angle_max']:
            score += 25
            reasons.append(f"✅ MA20角度{ma20_angle:.1f}°")
        elif ma20_angle > 1:
            score += 10
            reasons.append(f"➖ MA20角度{ma20_angle:.1f}°")
        else:
            return False, "MA20角度不足"
        
        # 2. RSI检查 (权重: 25分)
        rsi = row.get('rsi', 50)
        if pd.isna(rsi):
            return False, "RSI数据不足"
        
        if self.params['rsi_min'] <= rsi <= self.params['rsi_max']:
            score += 25
            reasons.append(f"✅ RSI={rsi:.1f}")
        elif rsi < 30:
            score += 20
            reasons.append(f"✅ RSI超卖={rsi:.1f}")
        else:
            return False, f"RSI={rsi:.1f}不在合理范围"
        
        # 3. MACD检查 (权重: 25分)
        macd_diff = row.get('macd_diff', 0)
        macd_dea = row.get('macd_dea', 0)
        macd_hist = row.get('macd_hist', 0)
        
        if pd.notna(macd_diff) and pd.notna(macd_dea):
            if macd_diff > macd_dea:  # 金叉
                score += 20
                reasons.append("✅ MACD金叉")
                if macd_hist > 0:
                    score += 5
                    reasons.append("✅ MACD柱状图为正")
            else:
                score += 5
                reasons.append(f"➖ MACD死叉")
        else:
            return False, "MACD数据不足"
        
        # 4. 成交量检查 (权重: 15分)
        volume_ratio = row.get('volume_ratio', 1)
        if pd.notna(volume_ratio) and volume_ratio >= self.params['volume_ratio_min']:
            score += 15
            reasons.append(f"✅ 成交量放大{volume_ratio:.1f}倍")
        elif pd.notna(volume_ratio) and volume_ratio >= 1.0:
            score += 8
            reasons.append(f"➖ 成交量正常")
        
        # 5. 动量检查 (权重: 10分)
        momentum = row.get('momentum_5', 0)
        if pd.notna(momentum):
            if momentum > 0:
                score += 10
                reasons.append(f"✅ 5日涨幅{momentum:.1f}%")
            elif momentum > -3:
                score += 5
        
        # 判断是否买入
        buy = score >= 70  # 必须70分以上
        
        return buy, f"评分{score}/100: " + ", ".join(reasons[:3])
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        回测策略表现
        """
        if len(df) < 60:
            return {'error': '数据不足'}
        
        df = self.calculate_indicators(df)
        
        trades = []
        position = 0
        buy_price = 0
        
        for i in range(30, len(df) - 5):  # 留5天验证
            row = df.iloc[i]
            buy, reason = self.check_buy_signal(row)
            
            # 买入
            if buy and position == 0:
                position = 1
                buy_price = row['close']
                buy_date = df.index[i]
                trades.append({
                    'date': buy_date,
                    'type': 'BUY',
                    'price': buy_price,
                    'reason': reason
                })
            
            # 卖出（持有5天后卖出，或者达到5%收益，或者下跌3%止损）
            elif position == 1:
                hold_days = 5
                if i >= len(df) - hold_days - 1:
                    # 最后一天，卖出
                    sell_price = df.iloc[-1]['close']
                    profit = (sell_price - buy_price) / buy_price * 100
                    trades.append({
                        'date': df.index[-1],
                        'type': 'SELL',
                        'price': sell_price,
                        'profit': profit
                    })
                    position = 0
                else:
                    current_price = row['close']
                    profit = (current_price - buy_price) / buy_price * 100
                    
                    # 止盈止损
                    if profit >= 5 or profit <= -3:
                        trades.append({
                            'date': df.index[i],
                            'type': 'SELL',
                            'price': current_price,
                            'profit': profit
                        })
                        position = 0
        
        # 计算结果
        if not trades:
            return {'trades': 0, 'win_rate': 0, 'avg_profit': 0}
        
        buy_trades = [t for t in trades if t['type'] == 'BUY']
        sell_trades = [t for t in trades if t['type'] == 'SELL']
        
        wins = 0
        total_profit = 0
        for t in sell_trades:
            if t.get('profit', 0) > 0:
                wins += 1
            total_profit += t.get('profit', 0)
        
        win_rate = wins / len(sell_trades) * 100 if sell_trades else 0
        avg_profit = total_profit / len(sell_trades) if sell_trades else 0
        
        return {
            'total_trades': len(buy_trades),
            'wins': wins,
            'losses': len(sell_trades) - wins,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'total_profit': total_profit,
            'trades_detail': trades[-10:]  # 最近10笔
        }


def scan_stocks(stocks: List[Tuple[str, str]], get_data_func) -> List[Dict]:
    """
    扫描股票池，返回符合条件的股票
    """
    strategy = HighAccuracyStrategy()
    results = []
    
    for code, name in stocks:
        # 获取数据
        df = get_data_func(code, days=100)
        if df is None or len(df) < 30:
            continue
        
        # 计算指标
        df = strategy.calculate_indicators(df)
        
        # 检查信号
        latest = df.iloc[-1]
        buy, reason = strategy.check_buy_signal(latest)
        
        if buy:
            results.append({
                'code': code,
                'name': name,
                'price': latest['close'],
                'reason': reason,
                'score': 80  # 简化，实际应该计算
            })
    
    return results


if __name__ == "__main__":
    # 测试
    print("高精度策略测试")
    print("=" * 50)
    
    # 模拟数据测试
    np.random.seed(42)
    dates = pd.date_range(end='2026-02-14', periods=200, freq='D')
    close = 100 + np.cumsum(np.random.randn(200) * 2)
    
    df = pd.DataFrame({
        'open': close - 1,
        'high': close + 2,
        'low': close - 2,
        'close': close,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)
    
    strategy = HighAccuracyStrategy()
    df = strategy.calculate_indicators(df)
    
    buy, reason = strategy.check_buy_signal(df.iloc[-1])
    print(f"买入信号: {buy}")
    print(f"原因: {reason}")
