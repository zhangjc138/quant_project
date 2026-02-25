#!/usr/bin/env python3
"""
股票数据管理器

功能：
- 本地缓存历史数据
- 增量更新（只获取新数据）
- 多数据源支持（akshare + baostock）
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')

# 数据存储目录
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'stocks')
os.makedirs(DATA_DIR, exist_ok=True)


def get_stock_file(symbol: str) -> str:
    """获取股票数据文件路径"""
    return os.path.join(DATA_DIR, f"{symbol}.csv")


def load_local_data(symbol: str) -> Optional[pd.DataFrame]:
    """加载本地缓存数据"""
    file_path = get_stock_file(symbol)
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            print(f"加载本地数据失败 {symbol}: {e}")
    return None


def save_local_data(symbol: str, df: pd.DataFrame):
    """保存数据到本地"""
    file_path = get_stock_file(symbol)
    try:
        df.to_csv(file_path)
        print(f"✅ 数据已缓存: {symbol} ({len(df)}条)")
    except Exception as e:
        print(f"保存本地数据失败 {symbol}: {e}")


def fetch_new_data(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
    """从网络获取新数据"""
    try:
        from stock_data import get_stock_daily
        df = get_stock_daily(symbol)
        return df
    except Exception as e:
        print(f"获取网络数据失败 {symbol}: {e}")
        return None


def get_stock_data_cached(symbol: str, days: int = 365, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """
    获取股票数据（带缓存）
    
    策略：
    1. 先尝试加载本地数据
    2. 如果本地数据超过7天没更新，则增量更新
    3. 如果本地数据不存在，则完整下载
    
    Args:
        symbol: 股票代码
        days: 需要的天数
        force_refresh: 是否强制刷新
    
    Returns:
        DataFrame 或 None
    """
    local_df = load_local_data(symbol)
    
    # 情况1: 强制刷新
    if force_refresh:
        print(f"🔄 强制刷新 {symbol}...")
        df = fetch_new_data(symbol, days)
        if df is not None and len(df) > 0:
            save_local_data(symbol, df)
            return df.tail(days)
        return local_df
    
    # 情况2: 本地无数据
    if local_df is None or len(local_df) == 0:
        print(f"📥 首次下载 {symbol}...")
        df = fetch_new_data(symbol, days)
        if df is not None and len(df) > 0:
            save_local_data(symbol, df)
            return df.tail(days)
        return None
    
    # 情况3: 检查是否需要增量更新
    last_date = local_df.index[-1]
    today = datetime.now()
    days_since_update = (today - last_date).days
    
    if days_since_update > 7:
        print(f"🔄 增量更新 {symbol} (距上次{days_since_update}天)...")
        df = fetch_new_data(symbol, days)
        if df is not None and len(df) > 0:
            # 合并数据并去重
            combined = pd.concat([local_df, df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            save_local_data(symbol, combined)
            return combined.tail(days)
        return local_df.tail(days)
    
    # 情况4: 使用本地数据
    print(f"📂 使用缓存 {symbol} ({len(local_df)}条, 更新于{last_date.strftime('%Y-%m-%d')})")
    return local_df.tail(days)


def batch_get_stocks(symbols: List[str], days: int = 365) -> Dict[str, pd.DataFrame]:
    """
    批量获取多只股票数据
    
    Args:
        symbols: 股票代码列表
        days: 天数
    
    Returns:
        dict: {symbol: DataFrame}
    """
    results = {}
    for sym in symbols:
        df = get_stock_data_cached(sym, days)
        if df is not None:
            results[sym] = df
    return results


def update_all_cached_data(symbols: List[str] = None):
    """
    更新所有缓存数据
    
    Args:
        symbols: 股票代码列表，默认从行业映射读取
    """
    if symbols is None:
        # 从默认股票池读取
        from app import INDUSTRY_STOCKS
        symbols = []
        for stocks in INDUSTRY_STOCKS.values():
            symbols.extend([s[0] for s in stocks])
        symbols = list(set(symbols))
    
    print(f"📊 开始更新 {len(symbols)} 只股票数据...")
    
    success = 0
    for sym in symbols:
        df = get_stock_data_cached(sym, force_refresh=True)
        if df is not None:
            success += 1
    
    print(f"✅ 更新完成: {success}/{len(symbols)}")


def get_cache_stats() -> Dict:
    """获取缓存统计信息"""
    stats = {
        'total_stocks': 0,
        'total_size_mb': 0,
        'oldest_data': None,
        'newest_data': None,
        'stocks': []
    }
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    stats['total_stocks'] = len(files)
    
    for f in files:
        file_path = os.path.join(DATA_DIR, f)
        size_mb = os.path.getsize(file_path) / 1024 / 1024
        stats['total_size_mb'] += size_mb
        
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            symbol = f.replace('.csv', '')
            first_date = df.index[0].strftime('%Y-%m-%d')
            last_date = df.index[-1].strftime('%Y-%m-%d')
            
            stats['stocks'].append({
                'symbol': symbol,
                'rows': len(df),
                'first_date': first_date,
                'last_date': last_date,
                'size_mb': round(size_mb, 2)
            })
        except:
            pass
    
    if stats['stocks']:
        stats['oldest_data'] = min(s['first_date'] for s in stats['stocks'])
        stats['newest_data'] = max(s['last_date'] for s in stats['stocks'])
    
    return stats


if __name__ == "__main__":
    # 测试
    print("=" * 50)
    print("股票数据管理器测试")
    print("=" * 50)
    
    # 测试获取数据
    df = get_stock_data_cached('600519')
    if df is not None:
        print(f"✅ 获取成功: {len(df)}条")
        print(f"最新: {df.iloc[-1]['close']}")
    
    # 显示缓存统计
    print("\n缓存统计:")
    stats = get_cache_stats()
    print(f"股票数量: {stats['total_stocks']}")
    print(f"总大小: {stats['total_size_mb']:.2f}MB")
