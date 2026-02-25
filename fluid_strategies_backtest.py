"""
流体物理策略集 - 综合回测框架
Fluid Physics Trading Strategies

代号对照:
  V-VAD: Volatility-Volume Anomaly Detector (波动率-成交量异常检测)
  LVM:   Liquidity Volatility Model (流动性波动率模型)
  BCF:   Breakout Confirmation Filter (突破确认过滤器)
  DPS:   Dynamic Position Sizer (动态仓位管理器)

使用方法:
    python fluid_strategies_backtest.py --stock 600519 --start 2024-01-01
"""
import pandas as pd
import numpy as np
import argparse
import sys
from datetime import datetime

# 导入各策略
from vvad import generate_signals as vvad_signals
from lvm import generate_signals as lvm_signals, get_stop_loss as lvm_stop_loss
from bcf import generate_signals as bcf_signals, get_breakout_info
from dps import generate_signals as dps_signals, calculate_position_size


def get_stock_data(stock_code: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    获取股票数据
    优先使用baostock,其次tushare,否则生成模拟数据
    """
    # 尝试baostock
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != '0':
            raise Exception(f"baostock登录失败: {lg.error_msg}")
        
        # 格式化股票代码 (sh.600519 -> 600519)
        if '.' in stock_code:
            bs_code = stock_code
        else:
            bs_code = f"sh.{stock_code}" if stock_code.startswith('6') else f"sz.{stock_code}"
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume",
            start_date=start or '2020-01-01',
            end_date=end or datetime.now().strftime('%Y-%m-%d'),
            frequency="d",
            adjustflag="2"  # 前复权
        )
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        bs.logout()
        
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df['volume'] = df['volume'].astype(float)
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['open'] = df['open'].astype(float)
            # 估算换手率 (成交量/流通股本, 假设茅台流通股本约10亿)
            df['turnover_rate'] = df['volume'] / 1e9
            df['code'] = stock_code
            print(f"   ✅ baostock数据: {len(df)} 条 ({df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
            return df
        raise Exception("baostock无数据")
    except ImportError:
        pass
    except Exception as e:
        print(f"   ⚠️ baostock获取失败: {e}")
    
    # 尝试tushare
    try:
        import tushare as ts
        df = ts.get_k_data(stock_code, start=start or '2020-01-01', 
                           end=end or datetime.now().strftime('%Y-%m-%d'))
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        df['volume'] = df['vol'].astype(float)
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        print(f"   ✅ tushare数据: {len(df)} 条")
        return df
    except Exception as e:
        print(f"   ⚠️ tushare获取失败: {e}")
    
    # 使用模拟数据
    print(f"   ⚠️ 使用模拟数据")
    return generate_mock_data(stock_code)


def generate_mock_data(stock_code: str, days: int = 500) -> pd.DataFrame:
    """生成模拟数据用于测试"""
    np.random.seed(hash(stock_code) % 10000)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # 模拟价格走势
    returns = np.random.normal(0.001, 0.02, days)
    close = 10 * np.exp(np.cumsum(returns))
    
    # 模拟成交量 (与价格波动相关)
    volume = np.random.lognormal(15, 0.5, days) * (1 + np.abs(returns) * 10)
    
    # 模拟换手率
    turnover_rate = np.random.uniform(0.5, 8, days) / 100
    
    df = pd.DataFrame({
        'date': dates,
        'code': stock_code,
        'open': close * (1 + np.random.uniform(-0.01, 0.01, days)),
        'high': close * (1 + np.random.uniform(0, 0.03, days)),
        'low': close * (1 - np.random.uniform(0, 0.03, days)),
        'close': close,
        'vol': volume,
        'volume': volume,
        'turnover_rate': turnover_rate
    })
    df.set_index('date', inplace=True)
    
    return df


def run_backtest(df: pd.DataFrame, strategy_name: str, 
                 initial_capital: float = 100000) -> dict:
    """
    简单回测
    
    返回:
        dict: 策略绩效指标
    """
    df = df.copy()
    
    # 选择策略
    if strategy_name == 'V-VAD':
        df = vvad_signals(df)
    elif strategy_name == 'LVM':
        df = lvm_signals(df)
    elif strategy_name == 'BCF':
        df = bcf_signals(df)
    elif strategy_name == 'DPS':
        df = dps_signals(df, {'base_capital': initial_capital})
    else:
        raise ValueError(f"未知策略: {strategy_name}")
    
    # 模拟交易
    position = 0
    cash = initial_capital
    trades = []
    
    for i in range(50, len(df)):  # 前50天预热
        row = df.iloc[i]
        
        if pd.isna(row.get('signal', 0)) or row.get('signal', 0) == 0:
            continue
            
        signal = row['signal']
        price = row['close']
        
        # 买入
        if signal == 1 and position == 0:
            shares = int(cash / price * 0.95)  # 95%仓位
            if shares > 0:
                cost = shares * price
                cash -= cost
                position = shares
                trades.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': price,
                    'shares': shares,
                    'amount': cost
                })
        
        # 卖出
        elif (signal < 0) and position > 0:
            proceeds = position * price
            cash += proceeds
            trades.append({
                'date': df.index[i],
                'type': 'SELL',
                'price': price,
                'shares': position,
                'amount': proceeds
            })
            position = 0
    
    # 最终持仓
    final_value = cash + position * df.iloc[-1]['close']
    
    # 计算指标
    total_return = (final_value - initial_capital) / initial_capital * 100
    num_trades = len(trades)
    
    return {
        'strategy': strategy_name,
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return_pct': total_return,
        'num_trades': num_trades,
        'trades': trades,
        'data': df
    }


def compare_strategies(df: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
    """
    对比所有策略
    """
    strategies = ['V-VAD', 'LVM', 'BCF', 'DPS']
    results = []
    
    print("\n" + "="*60)
    print("流体物理策略集 - 回测对比")
    print("="*60)
    print(f"股票: {df.iloc[-1].get('code', 'N/A')}")
    print(f"回测期: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"初始资金: ¥{initial_capital:,.0f}")
    print("-"*60)
    
    for strategy in strategies:
        try:
            result = run_backtest(df, strategy, initial_capital)
            results.append({
                '策略代号': strategy,
                '最终价值': f"¥{result['final_value']:,.0f}",
                '收益率': f"{result['total_return_pct']:+.2f}%",
                '交易次数': result['num_trades']
            })
            print(f"  {strategy:8s} | 收益: {result['total_return_pct']:+7.2f}% | "
                  f"交易: {result['num_trades']:3d}次 | "
                  f"最终: ¥{result['final_value']:,.0f}")
        except Exception as e:
            print(f"  {strategy:8s} | 错误: {e}")
    
    # 基准收益 (买入持有)
    bh_return = (df.iloc[-1]['close'] / df.iloc[50]['close'] - 1) * 100
    print("-"*60)
    print(f"  基准(买入持有) | 收益: {bh_return:+7.2f}%")
    print("="*60)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='流体物理策略回测')
    parser.add_argument('--stock', default='600519', help='股票代码')
    parser.add_argument('--start', default='2024-01-01', help='开始日期')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    parser.add_argument('--compare', action='store_true', help='对比所有策略')
    
    args = parser.parse_args()
    
    print(f"📊 加载数据: {args.stock}")
    df = get_stock_data(args.stock, args.start)
    print(f"   数据量: {len(df)} 条")
    
    if args.compare:
        compare_strategies(df, args.capital)
    else:
        # 默认运行V-VAD
        result = run_backtest(df, 'V-VAD', args.capital)
        print(f"\n✅ V-VAD 策略结果:")
        print(f"   收益率: {result['total_return_pct']:+.2f}%")
        print(f"   交易次数: {result['num_trades']}")


if __name__ == '__main__':
    main()
