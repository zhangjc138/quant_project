#!/usr/bin/env python3
"""
数据管理模块
- akshare 获取股票/期货数据
- 本地 CSV 存储
- 统一的数据接口
"""

import os
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
from typing import Optional, Union, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据目录
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def get_data_dir() -> str:
    """获取数据目录"""
    return DATA_DIR


def get_file_path(symbol: str, market: str = "stock", timeframe: str = "daily") -> str:
    """
    生成数据文件名
    
    Args:
        symbol: 合约代码
        market: 市场类型 (stock/futures)
        timeframe: 时间周期 (daily/minute)
        
    Returns:
        str: 文件路径
    """
    return os.path.join(get_data_dir(), f"{market}_{symbol}_{timeframe}.csv")


# ==================== 股票数据 ====================

def load_stock_daily(symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    加载股票日线数据
    
    Args:
        symbol: 股票代码 (如 600000)
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        
    Returns:
        DataFrame 或 None
    """
    filepath = get_file_path(symbol, "stock", "daily")
    
    # 检查本地缓存
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 筛选日期范围
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            return df
        except Exception as e:
            logger.warning(f"加载本地数据失败: {e}")
    
    return None


def fetch_stock_daily(symbol: str, start_date: str = None, end_date: str = None, 
                      save_local: bool = True) -> Optional[pd.DataFrame]:
    """
    获取股票日线数据（自动缓存）
    
    Args:
        symbol: 股票代码 (如 600000)
        start_date: 开始日期
        end_date: 结束日期
        save_local: 是否保存到本地
        
    Returns:
        DataFrame 或 None
    """
    # 确定日期范围
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    
    try:
        # 转换代码格式
        if symbol.startswith("6"):
            symbol_ak = "sh" + symbol
        else:
            symbol_ak = "sz" + symbol
        
        # 获取数据
        df = ak.stock_zh_a_hist(
            symbol=symbol_ak,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq"
        )
        
        if df is None or df.empty:
            logger.warning(f"未获取到数据: {symbol}")
            return None
        
        # 统一格式
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '振幅': 'amplitude',
            '涨跌幅': 'change_pct',
            '涨跌额': 'change',
            '换手率': 'turnover'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        # 计算 MA20
        df['MA20'] = df['close'].rolling(window=20).mean()
        
        # 保存到本地
        if save_local:
            filepath = get_file_path(symbol, "stock", "daily")
            df.to_csv(filepath)
            logger.info(f"数据已保存: {filepath}")
        
        return df
        
    except Exception as e:
        logger.error(f"获取股票数据失败 {symbol}: {e}")
        return None


def get_stock_ma20_angle(symbol: str) -> Tuple[float, float, float]:
    """
    获取股票 MA20 角度（便捷函数）
    
    Args:
        symbol: 股票代码
        
    Returns:
        Tuple[MA20, MA20_angle, close_price]
    """
    df = fetch_stock_daily(symbol)
    
    if df is None or len(df) < 25:
        return 0.0, 0.0, 0.0
    
    # 取最近 20 个 MA20 值
    ma20_series = df['MA20'].dropna().tail(20)
    if len(ma20_series) < 20:
        return 0.0, 0.0, 0.0
    
    # 计算角度
    x = np.arange(len(ma20_series))
    y = ma20_series.values
    
    slope = np.cov(x, y)[0, 1] / np.var(x)
    angle = np.degrees(np.arctan(slope / ma20_series.mean() * 100))
    
    return df['MA20'].iloc[-1], angle, df['close'].iloc[-1]


def get_realtime_price(symbol: str) -> Optional[dict]:
    """
    获取实时行情
    
    Args:
        symbol: 股票代码
        
    Returns:
        dict 或 None
    """
    try:
        # 转换代码格式
        if symbol.startswith("6"):
            symbol_ak = "sh" + symbol
        else:
            symbol_ak = "sz" + symbol
        
        df = ak.stock_zh_a实时行情_阿里(symbol=symbol_ak)
        
        if df is None or df.empty:
            return None
        
        latest = df.iloc[0]
        
        return {
            'symbol': symbol,
            'name': latest.get('名称', symbol),
            'price': float(latest.get('最新价', 0)),
            'open': float(latest.get('今开', 0)),
            'high': float(latest.get('最高', 0)),
            'low': float(latest.get('最低', 0)),
            'volume': float(latest.get('成交量', 0)),
            'amount': float(latest.get('成交额', 0)),
            'change_pct': float(latest.get('涨跌幅', 0)),
        }
        
    except Exception as e:
        logger.error(f"获取实时行情失败 {symbol}: {e}")
        return None


# ==================== 期货数据 ====================

def fetch_futures_daily(symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """
    获取期货日线数据
    
    Args:
        symbol: 合约代码 (如 IF2006)
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        DataFrame 或 None
    """
    # 确定日期范围
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
    
    try:
        # 中金所股指期货
        if symbol.startswith("IF"):
            df = ak.futures_zh_daily_sina(symbol=symbol)
        # 其他期货
        else:
            df = ak.futures_zh_index_sina(symbol=symbol)
        
        if df is None or df.empty:
            return None
        
        # 统一格式
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '持仓量': 'open_interest'
        })
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
        
    except Exception as e:
        logger.error(f"获取期货数据失败 {symbol}: {e}")
        return None


# ==================== 辅助函数 ====================

def merge_stock_data(symbols: list, start_date: str = None, end_date: str = None) -> dict:
    """
    合并多只股票数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        dict: {symbol: DataFrame}
    """
    data = {}
    
    for symbol in symbols:
        df = fetch_stock_daily(symbol, start_date, end_date)
        if df is not None:
            data[symbol] = df
    
    return data


def calculate_ma20_angle(ma20_series: pd.Series) -> float:
    """
    计算 MA20 角度
    
    Args:
        ma20_series: MA20 序列
        
    Returns:
        float: 角度（度）
    """
    if len(ma20_series) < 20:
        return 0.0
    
    x = np.arange(len(ma20_series))
    y = ma20_series.values
    
    slope = np.cov(x, y)[0, 1] / np.var(x)
    angle = np.degrees(np.arctan(slope / np.mean(y) * 100))
    
    return angle


# ==================== 主程序测试 ====================

if __name__ == "__main__":
    print("=== 数据管理模块测试 ===\n")
    
    # 测试获取股票数据
    print("1. 获取浦发银行(600000)日线数据...")
    df = fetch_stock_daily("600000", "2024-01-01", "2025-01-01")
    if df is not None:
        print(f"   获取到 {len(df)} 条数据")
        print(f"   最新收盘: {df['close'].iloc[-1]:.2f}")
        print(f"   MA20: {df['MA20'].iloc[-1]:.2f}")
    else:
        print("   获取数据失败")
    
    # 测试 MA20 角度
    print("\n2. 计算浦发银行 MA20 角度...")
    ma20, angle, price = get_stock_ma20_angle("600000")
    print(f"   MA20: {ma20:.2f}")
    print(f"   MA20角度: {angle:.2f}°")
    print(f"   当前价格: {price:.2f}")
    
    # 测试实时行情
    print("\n3. 获取浦发银行实时行情...")
    realtime = get_realtime_price("600000")
    if realtime:
        print(f"   当前价格: {realtime['price']:.2f}")
        print(f"   涨跌幅: {realtime['change_pct']:+.2f}%")
    else:
        print("   获取实时行情失败")
    
    print("\n=== 测试完成 ===")
