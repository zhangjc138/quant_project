#!/usr/bin/env python3
"""
分钟级数据模块

提供股票分钟级行情数据获取功能
支持1/5/15/30/60分钟K线数据
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import akshare as ak
import time


def get_stock_minute_data(
    symbol: str,
    period: str = "1",
    start_date: str = None,
    end_date: str = None,
    adjust: str = "qfq"
) -> Optional[pd.DataFrame]:
    """
    获取股票分钟级数据
    
    Args:
        symbol: 股票代码（如：600519）
        period: 分钟周期 ("1"=1分钟, "5"=5分钟, "15"=15分钟, "30"=30分钟, "60"=60分钟)
        start_date: 开始日期（格式：YYYYMMDD）
        end_date: 结束日期（格式：YYYYMMDD）
        adjust: 复权类型 ("qfq"=前复权, "hfq"=后复权, ""=不复权)
    
    Returns:
        DataFrame: 分钟K线数据，失败返回None
    """
    try:
        # 标准化股票代码
        code = symbol.replace(".", "").replace("SH", "").replace("SZ", "")
        
        # 设置默认日期范围（最近5个交易日）
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=10)).strftime('%Y%m%d')
        
        # 使用 akshare 获取分钟数据
        if adjust:
            df = ak.stock_zh_a_minute(
                symbol=code,
                period=period,
                adjust=adjust
            )
        else:
            df = ak.stock_zh_a_minute(
                symbol=code,
                period=period,
                adjust="qfq"
            )
        
        if df is None or df.empty:
            return None
        
        # 标准化列名
        df = _standardize_columns(df)
        
        return df
        
    except Exception as e:
        print(f"获取分钟数据失败: {symbol}, 错误: {e}")
        return None


def get_stock_zh_a_hist_minute(
    symbol: str,
    period: str = "5",
    start_date: str = None,
    end_date: str = None,
    adjust: str = "qfq"
) -> Optional[pd.DataFrame]:
    """
    获取股票历史分钟数据（另一种方式）
    
    Args:
        symbol: 股票代码
        period: 分钟周期
        start_date: 开始日期
        end_date: 结束日期
        adjust: 复权类型
    
    Returns:
        DataFrame: 分钟K线数据
    """
    try:
        # 标准化股票代码
        code = symbol.replace(".", "").replace("SH", "").replace("SZ", "")
        
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        
        df = ak.stock_zh_a_hist_minute(
            symbol=code,
            period=period,
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )
        
        if df is None or df.empty:
            return None
        
        # 标准化列名
        df = _standardize_columns(df)
        
        return df
        
    except Exception as e:
        print(f"获取历史分钟数据失败: {symbol}, 错误: {e}")
        return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名"""
    # 列名映射
    column_mapping = {
        '时间': 'time',
        '日期': 'date',
        '开盘': 'open',
        '最高': 'high',
        '最低': 'low',
        '收盘': 'close',
        '成交量': 'volume',
        '成交额': 'amount',
        '涨跌幅': 'change_pct',
        '涨跌额': 'change',
        '换手率': 'turnover',
    }
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    # 确保必要列存在
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 如果有 'time' 列但没有 'date' 列，使用 'time' 作为索引
    if 'time' in df.columns and 'date' not in df.columns:
        df['date'] = df['time']
    
    # 转换数据类型
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'volume' in df.columns:
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    
    return df


def calculate_minute_indicators(
    df: pd.DataFrame,
    ma_periods: List[int] = [5, 10, 20, 60],
    rsi_period: int = 14,
    boll_period: int = 20,
    boll_std: int = 2,
    kdj_period: int = 9,
    kdj_m1: int = 3,
    kdj_m2: int = 3
) -> pd.DataFrame:
    """
    计算分钟级技术指标
    
    Args:
        df: 分钟K线数据
        ma_periods: MA周期列表
        rsi_period: RSI周期
        boll_period: BOLL周期
        boll_std: BOLL标准差倍数
        kdj_period: KDJ周期
        kdj_m1: KDJ M1
        kdj_m2: KDJ M2
    
    Returns:
        DataFrame: 带技术指标的数据
    """
    if df is None or len(df) < 10:
        return df
    
    df = df.copy()
    
    # 基础指标
    df['change_pct'] = df['close'].pct_change() * 100
    
    # 移动平均线
    for period in ma_periods:
        df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi'] = df['rsi'].fillna(50)
    
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    
    df['macd_dif'] = ema12 - ema26
    df['macd_dea'] = df['macd_dif'].ewm(span=9, adjust=False).mean()
    df['macd'] = (df['macd_dif'] - df['macd_dea']) * 2
    
    # BOLL
    ma = df['close'].rolling(window=boll_period).mean()
    std = df['close'].rolling(window=boll_period).std()
    
    df['boll_upper'] = ma + (std * boll_std)
    df['boll_lower'] = ma - (std * boll_std)
    df['boll_mid'] = ma
    
    # KDJ
    low_min = df['low'].rolling(window=kdj_period).min()
    high_max = df['high'].rolling(window=kdj_period).max()
    
    rsv = (df['close'] - low_min) / (high_max - low_min) * 100
    rsv = rsv.fillna(50)
    
    df['kdj_k'] = rsv.ewm(alpha=1/kdj_m1, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(alpha=1/kdj_m2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    
    # 填充NaN
    df = df.fillna(0)
    
    return df


def get_minute_signal(df: pd.DataFrame) -> Dict:
    """
    基于分钟数据生成信号
    
    Args:
        df: 带指标的分钟数据
    
    Returns:
        Dict: 信号信息
    """
    if df is None or len(df) < 20:
        return {'signal': 'NEUTRAL', 'confidence': 0, 'reason': '数据不足'}
    
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    
    signals = []
    confidence = 0
    
    # MA 趋势信号
    if latest['close'] > latest.get('ma20', latest['close']):
        signals.append("MA趋势向上")
        confidence += 0.2
    
    # RSI 信号
    rsi = latest.get('rsi', 50)
    if rsi < 30:
        signals.append("RSI超卖")
        confidence += 0.2
    elif rsi > 70:
        signals.append("RSI超买")
        confidence -= 0.2
    else:
        signals.append("RSI中性")
        confidence += 0.1
    
    # MACD 信号
    if latest.get('macd_dif', 0) > latest.get('macd_dea', 0):
        if prev.get('macd_dif', 0) <= prev.get('macd_dea', 0):
            signals.append("MACD金叉")
            confidence += 0.3
        else:
            signals.append("MACD多头")
            confidence += 0.2
    else:
        signals.append("MACD空头")
        confidence -= 0.2
    
    # BOLL 信号
    if latest['close'] > latest.get('boll_upper', latest['close']):
        signals.append("BOLL上轨压力")
        confidence -= 0.2
    elif latest['close'] < latest.get('boll_lower', latest['close']):
        signals.append("BOLL下轨支撑")
        confidence += 0.2
    else:
        signals.append("BOLL震荡")
        confidence += 0.1
    
    # KDJ 信号
    kdj_k = latest.get('kdj_k', 50)
    kdj_d = latest.get('kdj_d', 50)
    kdj_j = latest.get('kdj_j', 50)
    
    if kdj_k < 20 and kdj_d < 20:
        signals.append("KDJ超卖")
        confidence += 0.2
    elif kdj_k > 80 and kdj_d > 80:
        signals.append("KDJ超买")
        confidence -= 0.2
    elif kdj_k > kdj_d:
        signals.append("KDJ多头")
        confidence += 0.1
    else:
        signals.append("KDJ空头")
        confidence -= 0.1
    
    # 确定最终信号
    if confidence >= 0.4:
        signal = "BUY"
    elif confidence <= -0.4:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    confidence = max(0, min(1, abs(confidence)))
    
    return {
        'signal': signal,
        'confidence': confidence,
        'reason': " | ".join(signals),
        'rsi': rsi,
        'macd_dif': latest.get('macd_dif', 0),
        'macd_dea': latest.get('macd_dea', 0),
        'boll_position': (latest['close'] - latest.get('boll_lower', latest['close'])) / 
                        (latest.get('boll_upper', latest['close']) - latest.get('boll_lower', latest['close']) + 0.001),
        'kdj_k': kdj_k,
        'kdj_d': kdj_d
    }


# 周期配置
PERIOD_CONFIG = {
    "1": {"name": "1分钟", "minutes_per_bar": 1, "max_bars": 500},
    "5": {"name": "5分钟", "minutes_per_bar": 5, "max_bars": 500},
    "15": {"name": "15分钟", "minutes_per_bar": 15, "max_bars": 500},
    "30": {"name": "30分钟", "minutes_per_bar": 30, "max_bars": 500},
    "60": {"name": "60分钟", "minutes_per_bar": 60, "max_bars": 500},
}


def get_period_name(period: str) -> str:
    """获取周期名称"""
    return PERIOD_CONFIG.get(period, {}).get("name", f"{period}分钟")


if __name__ == "__main__":
    # 测试
    print("分钟级数据模块测试")
    
    # 获取5分钟数据
    print("\n获取股票5分钟数据...")
    df = get_stock_minute_data("600519", period="5")
    
    if df is not None:
        print(f"数据量: {len(df)} 条")
        print(f"列名: {df.columns.tolist()}")
        print(f"\n前5行:")
        print(df.head())
        
        # 计算指标
        print("\n计算技术指标...")
        df_indicators = calculate_minute_indicators(df)
        
        if 'rsi' in df_indicators.columns:
            print(f"RSI范围: {df_indicators['rsi'].min():.1f} - {df_indicators['rsi'].max():.1f}")
        
        # 生成信号
        print("\n生成信号...")
        signal_info = get_minute_signal(df_indicators)
        print(f"信号: {signal_info}")
    else:
        print("获取数据失败")
