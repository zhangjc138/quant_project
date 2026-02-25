"""
策略4: Dynamic Position Sizer (DPS)
代号: DPS
功能: 动态仓位公式
核心: Position ∝ sqrt(V/N)
     换手率高 -> 仓位大
     换手率低 -> 仓位小
"""
import pandas as pd
import numpy as np

def calculate_position_size(base_capital: float, df: pd.DataFrame, 
                            current_idx: int, params: dict = None) -> float:
    """
    计算动态仓位
    
    公式: Position = BaseCapital * sqrt(换手率) * 风险系数
    
    参数:
        base_capital: 基础资金
        df: 股票数据
        current_idx: 当前索引
        params: 参数字典
            - max_position: 最大仓位比例 (默认0.3 = 30%)
            - min_position: 最小仓位比例 (默认0.05 = 5%)
            - risk_factor: 风险系数 (默认1.0)
    
    返回:
        建议买入金额
    """
    params = params or {}
    max_position = params.get('max_position', 0.3)
    min_position = params.get('min_position', 0.05)
    risk_factor = params.get('risk_factor', 1.0)
    
    if current_idx < 20:
        return base_capital * min_position
    
    df_slice = df.iloc[:current_idx+1]
    
    # 获取换手率
    if 'turnover_rate' in df_slice.columns:
        turnover = df_slice.iloc[-1]['turnover_rate']
    else:
        # 估算换手率
        avg_vol = df_slice['volume'].rolling(20).mean().iloc[-1]
        turnover = avg_vol / 100000000  # 简化
    
    # 换手率范围限制 (0.5% - 10%)
    turnover = max(0.005, min(0.10, turnover))
    
    # 动态仓位 = 基础仓位 * sqrt(换手率) * 风险系数
    # 换手率5% -> sqrt(0.05) = 0.224 -> 仓位22.4%
    # 换手率1% -> sqrt(0.01) = 0.1 -> 仓位10%
    # 换手率10% -> sqrt(0.10) = 0.316 -> 仓位31.6%
    position_ratio = np.sqrt(turnover) * risk_factor
    
    # 限制范围
    position_ratio = max(min_position, min(max_position, position_ratio))
    
    return base_capital * position_ratio


def generate_signals(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    生成交易信号
    
    注意: DPS本身不是独立策略,而是仓位管理模块
    这里生成辅助信号用于演示
    
    返回:
        带position_size列的DataFrame (仓位建议)
    """
    params = params or {}
    base_capital = params.get('base_capital', 100000)
    
    df = df.copy()
    
    # 计算换手率
    if 'turnover_rate' in df.columns:
        df['turnover'] = df['turnover_rate']
    else:
        df['turnover'] = df['volume'] / 100000000
    
    # 计算动态仓位
    df['position_size'] = 0.0
    
    for i in range(20, len(df)):
        df.loc[df.index[i], 'position_size'] = calculate_position_size(
            base_capital, df, i, params
        )
    
    # 仓位变化信号
    df['position_change'] = df['position_size'].pct_change()
    
    # 简单的买入/卖出信号基于仓位突变
    df['signal'] = 0
    
    # 仓位大幅增加 (>20%) -> 买入信号
    buy_condition = df['position_change'] > 0.2
    df.loc[buy_condition, 'signal'] = 1
    
    # 仓位大幅减少 (>30%) -> 卖出信号
    sell_condition = df['position_change'] < -0.3
    df.loc[sell_condition, 'signal'] = -1
    
    return df


def get_recommended_position(current_price: float, base_capital: float,
                              turnover_rate: float, volatility: float = None,
                              params: dict = None) -> dict:
    """
    快速获取推荐仓位
    
    参数:
        current_price: 当前股价
        base_capital: 基础资金
        turnover_rate: 换手率 (小数,如0.05表示5%)
        volatility: 波动率 (可选,用于风险调整)
        params: 参数
    
    返回:
        dict: {
            'shares': 股数,
            'amount': 金额,
            'position_pct': 仓位比例,
            'risk_adjusted': 是否风险调整
        }
    """
    params = params or {}
    max_position = params.get('max_position', 0.3)
    min_position = params.get('min_position', 0.05)
    risk_factor = params.get('risk_factor', 1.0)
    
    # 如果提供了波动率,调整风险系数
    if volatility is not None:
        # 波动率高 -> 降低风险系数
        if volatility > 0.03:  # 日波动 > 3%
            risk_factor *= 0.7
        elif volatility < 0.01:  # 日波动 < 1%
            risk_factor *= 1.2
    
    # 计算仓位
    turnover = max(0.005, min(0.10, turnover_rate))
    position_ratio = np.sqrt(turnover) * risk_factor
    position_ratio = max(min_position, min(max_position, position_ratio))
    
    amount = base_capital * position_ratio
    shares = int(amount / current_price)
    
    return {
        'shares': shares,
        'amount': round(amount, 2),
        'position_pct': round(position_ratio * 100, 2),
        'risk_adjusted': volatility is not None,
        'risk_factor': risk_factor
    }


if __name__ == '__main__':
    print("DPS 策略 - 动态仓位管理器")
    print("=" * 50)
    print("核心公式: Position ∝ √换手率")
    print()
    print("示例:")
    # 测试不同换手率
    for turnover in [0.01, 0.02, 0.05, 0.10]:
        result = get_recommended_position(
            current_price=10.0,
            base_capital=100000,
            turnover_rate=turnover
        )
        print(f"  换手率 {turnover*100:5.1f}% -> 仓位 {result['position_pct']:5.2f}% "
              f"(¥{result['amount']:,.0f})")
