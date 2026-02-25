"""
策略3: Breakout Confirmation Filter (BCF)
代号: BCF
功能: 突破买入过滤
核心: 过滤假突破
  - 真突破: 放量突破 + 波动率上升
  - 假突破: 放量突破 + 波动率下降
"""
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    生成交易信号
    
    参数:
        df: 包含 close, volume, high, low 的DataFrame
        params: 参数字典
            - lookback: 周期数 (默认20)
            - vol_multiplier: 成交量倍数 (默认1.5)
            - breakout_threshold: 突破幅度 (默认0.03 = 3%)
    
    返回:
        带signal列的DataFrame
    """
    params = params or {}
    lookback = params.get('lookback', 20)
    vol_multiplier = params.get('vol_multiplier', 1.5)
    breakout_threshold = params.get('breakout_threshold', 0.03)
    
    df = df.copy()
    
    # 20日高点
    df['high_20d'] = df['high'].rolling(lookback).max()
    df['low_20d'] = df['low'].rolling(lookback).min()
    
    # 成交量均线
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    
    # 波动率
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(20).std()
    df['volatility_ma5'] = df['volatility'].rolling(5).mean()
    # 波动率变化
    df['volatility_change'] = df['volatility'] / df['volatility'].shift(10)
    
    # 突破条件
    df['breakout_up'] = df['close'] > df['high_20d'].shift(1)  # 突破20日高点
    df['breakout_down'] = df['close'] < df['low_20d'].shift(1)  # 跌破20日低点
    
    # 突破幅度
    df['breakout_strength'] = (df['close'] - df['high_20d'].shift(1)) / df['high_20d'].shift(1)
    
    # 信号生成
    df['signal'] = 0
    
    # ======== 买入信号 ========
    # 条件1: 突破20日高点
    # 条件2: 成交量放大 > 1.5倍
    # 条件3: 波动率上升 > 1.1 (确认真突破)
    buy_condition = (
        df['breakout_up'] & 
        (df['vol_ratio'] > vol_multiplier) & 
        (df['volatility_change'] > 1.1)
    )
    df.loc[buy_condition, 'signal'] = 1
    
    # ======== 卖出信号 ========
    # 假突破: 放量突破但波动率下降
    fake_breakout = (
        df['breakout_up'] & 
        (df['vol_ratio'] > vol_multiplier) & 
        (df['volatility_change'] < 0.9)
    )
    df.loc[fake_breakout, 'signal'] = -2  # 假突破警告
    
    # 跌破20日低点 + 波动率上升 = 真下跌
    sell_condition = (
        df['breakout_down'] & 
        (df['volatility_change'] > 1.0)
    )
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def get_breakout_info(df: pd.DataFrame, current_idx: int) -> dict:
    """
    获取当前突破状态详情
    """
    if current_idx < 20:
        return {'status': '数据不足'}
    
    row = df.iloc[current_idx]
    
    info = {
        'price': row['close'],
        'high_20d': row['high_20d'],
        'breakout': row['breakout_up'],
        'vol_ratio': row['vol_ratio'],
        'volatility_change': row['volatility_change'],
        'signal': row['signal']
    }
    
    # 解读
    if row['signal'] == 1:
        info['interpretation'] = '✅ 真突破 - 放量上涨+波动率上升'
    elif row['signal'] == -2:
        info['interpretation'] = '⚠️ 假突破 - 放量但波动率下降'
    elif row['signal'] == -1:
        info['interpretation'] = '🔻 真下跌 - 跌破支撑'
    else:
        info['interpretation'] = '⏳ 观望'
    
    return info


if __name__ == '__main__':
    print("BCF 策略 - 突破确认过滤器")
    print("=" * 50)
    print("信号说明:")
    print("  1  = 真突破买入")
    print(" -1  = 真跌破卖出")
    print(" -2  = 假突破警告(放量但不涨)")
    print("  0  = 观望")
