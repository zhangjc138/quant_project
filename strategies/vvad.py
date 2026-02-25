"""
策略1: Volatility-Volume Anomaly Detector (V-VAD)
代号: V-VAD
功能: 波动率-成交量异常检测
核心: 成交量放大时波动率应该下降,若反向上升则为异常
"""
import pandas as pd
import numpy as np

def generate_signals(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    生成交易信号
    
    参数:
        df: 包含 close, volume, turnover_rate 的DataFrame
        params: 参数字典
            - vol_threshold: 成交量倍数阈值 (默认1.5)
            - vola_change_threshold: 波动率变化阈值 (默认1.3)
    
    返回:
        带signal列的DataFrame: 1买入, -1卖出, 0持有
    """
    params = params or {}
    vol_threshold = params.get('vol_threshold', 1.5)
    vola_change_threshold = params.get('vola_change_threshold', 1.3)
    
    df = df.copy()
    
    # 计算20日成交量均线
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    # 成交量比率
    df['vol_ratio'] = df['volume'] / df['vol_ma20']
    
    # 计算20日波动率(收盘价收益率的标准差)
    df['returns'] = df['close'].pct_change()
    df['volatility_20d'] = df['returns'].rolling(20).std()
    # 波动率变化比率(当前波动率 / 10日前波动率)
    df['volatility_ratio'] = df['volatility_20d'] / df['volatility_20d'].shift(10)
    
    # 生成信号
    df['signal'] = 0
    
    # 买入信号: 放量且波动率上升 = 强趋势延续
    buy_condition = (df['vol_ratio'] > vol_threshold) & (df['volatility_ratio'] > vola_change_threshold)
    df.loc[buy_condition, 'signal'] = 1
    
    # 卖出信号: 放量但波动率下降 = 可能见顶/主力出货
    sell_condition = (df['vol_ratio'] > vol_threshold) & (df['volatility_ratio'] < 0.8)
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def get_position_size(base_size: float, df: pd.DataFrame, current_idx: int) -> float:
    """
    根据波动率动态调整仓位
    波动率高 -> 仓位小
    波动率低 -> 仓位大
    """
    if current_idx < 20:
        return base_size
    
    volatility = df.iloc[current_idx]['volatility_20d']
    avg_volatility = df.iloc[current_idx-20:current_idx]['volatility_20d'].mean()
    
    if pd.isna(volatility) or pd.isna(avg_volatility) or avg_volatility == 0:
        return base_size
    
    # 波动率比率
    vola_ratio = avg_volatility / volatility
    # 限制在0.5-2倍之间
    vola_ratio = max(0.5, min(2.0, vola_ratio))
    
    return base_size * vola_ratio


if __name__ == '__main__':
    # 简单测试
    import tushare as ts
    
    print("V-VAD 策略 - 波动率成交量异常检测")
    print("=" * 50)
    
    try:
        # 获取测试数据
        df = ts.get_k_data('600519', start='2024-01-01')
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        df['volume'] = df['vol'].astype(float)
        df['close'] = df['close'].astype(float)
        
        result = generate_signals(df)
        
        print(f"\n买入信号数: {(result['signal'] == 1).sum()}")
        print(f"卖出信号数: {(result['signal'] == -1).sum()}")
        
        # 显示最近信号
        signals = result[result['signal'] != 0].tail(10)
        if len(signals) > 0:
            print("\n最近信号:")
            print(signals[['close', 'vol_ratio', 'volatility_ratio', 'signal']])
            
    except Exception as e:
        print(f"需要tushare或提供数据: {e}")
