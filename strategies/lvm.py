"""
策略2: Liquidity Volatility Model (LVM)
代号: LVM
功能: 流动性波动率模型
核心: sigma = K / sqrt(V/N), 换手率越低波动率越高
应用: 根据换手率动态调整止损宽度
"""
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    生成交易信号
    
    参数:
        df: 包含 close, volume, turnover_rate 的DataFrame
        params: 参数字典
            - k_factor: K系数 (默认0.01)
            - turnover_threshold: 换手率阈值 (默认0.02 = 2%)
    
    返回:
        带signal列的DataFrame
    """
    params = params or {}
    k_factor = params.get('k_factor', 0.01)
    turnover_threshold = params.get('turnover_threshold', 0.02)
    
    df = df.copy()
    
    # 换手率 (需要volume和流通股本,若无则用volume/close估算)
    if 'turnover_rate' in df.columns:
        df['turnover'] = df['turnover_rate']
    else:
        # 估算换手率 = 成交量 / 流通股本 (假设流通股本=成交量累计/100)
        df['turnover'] = df['volume'] / 100000000  # 简化估算
    
    # 计算预期波动率: sigma = K / sqrt(换手率)
    df['expected_volatility'] = k_factor / np.sqrt(df['turnover'].replace(0, np.nan))
    
    # 计算实际波动率
    df['returns'] = df['close'].pct_change()
    df['actual_volatility'] = df['returns'].rolling(20).std()
    
    # 波动率偏离度
    df['volatility_deviation'] = df['actual_volatility'] / df['expected_volatility']
    
    # 信号生成
    df['signal'] = 0
    
    # 买入信号: 实际波动率远高于预期 -> 超卖反弹
    # 波动率偏离度 > 2 表示实际波动是预期的2倍以上
    buy_condition = df['volatility_deviation'] > 2.0
    df.loc[buy_condition, 'signal'] = 1
    
    # 卖出信号: 实际波动率远低于预期 -> 超买回调
    sell_condition = df['volatility_deviation'] < 0.5
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def get_stop_loss(entry_price: float, df: pd.DataFrame, current_idx: int, 
                  atr_multiplier: float = 2.0) -> dict:
    """
    根据流动性动态计算止损宽度
    
    换手率低 -> 波动率高 -> 止损宽
    换手率高 -> 波动率低 -> 止损窄
    
    返回:
        dict: {'stop_loss': 止损价, 'stop_loss_pct': 止损百分比}
    """
    if current_idx < 20:
        return {'stop_loss': entry_price * 0.95, 'stop_loss_pct': 0.05}
    
    turnover = df.iloc[current_idx]['turnover']
    returns = df['returns'].iloc[:current_idx+1]
    
    # ATR计算
    high = df['high'].iloc[:current_idx+1] if 'high' in df.columns else df['close']
    low = df['low'].iloc[:current_idx+1] if 'low' in df.columns else df['close']
    close = df['close'].iloc[:current_idx+1]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    
    # 基础止损比例 = ATR倍数
    base_stop_pct = atr_multiplier * atr / entry_price
    
    # 根据换手率调整
    # 换手率 > 5% -> 止损收窄
    # 换手率 < 1% -> 止损放宽
    if turnover > 0.05:
        adjusted_stop_pct = base_stop_pct * 0.8
    elif turnover < 0.01:
        adjusted_stop_pct = base_stop_pct * 1.5
    else:
        adjusted_stop_pct = base_stop_pct
    
    # 限制范围
    adjusted_stop_pct = max(0.02, min(0.15, adjusted_stop_pct))
    
    stop_loss = entry_price * (1 - adjusted_stop_pct)
    
    return {
        'stop_loss': stop_loss,
        'stop_loss_pct': adjusted_stop_pct,
        'atr': atr,
        'turnover': turnover
    }


if __name__ == '__main__':
    print("LVM 策略 - 流动性波动率模型")
    print("=" * 50)
    print("功能: 根据换手率计算预期波动率,偏离时产生信号")
    print("买入: 实际波动率 >> 预期波动率 (超卖)")
    print("卖出: 实际波动率 << 预期波动率 (超买)")
