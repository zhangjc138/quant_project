#!/usr/bin/env python3
"""
股票历史数据模块

支持 akshare 和 baostock 双数据源
实时获取A股历史日线数据
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import akshare as ak
import warnings
warnings.filterwarnings('ignore')


def get_stock_daily(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    use_backup: bool = False
) -> Optional[pd.DataFrame]:
    """
    获取股票日线数据（主数据源：akshare）
    
    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        use_backup: 是否强制使用备份数据源
    
    Returns:
        DataFrame: 日线数据，失败返回None
    """
    # 默认日期范围
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    # 标准化股票代码
    code = symbol.replace(".", "").replace("SH", "").replace("SZ", "")
    
    # 如果强制使用备份或 akshare 失败
    if use_backup:
        return _get_from_baostock(code, start_date, end_date)
    
    try:
        # 尝试 akshare
        df = _get_from_akshare(code, start_date, end_date)
        if df is not None and len(df) > 0:
            return df
        
        # akshare 失败，尝试 baostock
        print(f"akshare获取失败，尝试baostock...")
        return _get_from_baostock(code, start_date, end_date)
        
    except Exception as e:
        print(f"获取数据失败: {e}")
        return _get_from_baostock(code, start_date, end_date)


def _get_from_akshare(
    symbol: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    从 akshare 获取数据
    """
    try:
        # 确定交易所
        if symbol.startswith('6') or symbol.startswith('5'):
            # 上海交易所
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
        else:
            # 深圳交易所
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"
            )
        
        if df is None or df.empty:
            return None
        
        # 标准化列名
        df.columns = [col.lower() for col in df.columns]
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
        df = df.set_index('date').sort_index()
        
        # 确保数值列
        for col in ['open', 'close', 'high', 'low', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
        
    except Exception as e:
        print(f"akshare获取失败: {e}")
        return None


def _get_from_baostock(
    symbol: str,
    start_date: str,
    end_date: str
) -> Optional[pd.DataFrame]:
    """
    从 baostock 获取数据
    """
    try:
        import baostock as bs
        
        # 登录
        lg = bs.login()
        if lg.error_code != '0':
            return None
        
        # 转换日期格式
        start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        end = f"{end_date[:4]}-{end_date[5:7]}-{end_date[8:10]}"
        
        # 获取数据
        rs = bs.query_history_k_data_plus(
            symbol=f"sh.{symbol}" if symbol.startswith('6') or symbol.startswith('5') else f"sz.{symbol}",
            fields="date,open,high,low,close,volume,amount,turn",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2"  # 前复权
        )
        
        # 登出
        bs.logout()
        
        # 解析结果
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
        
        if not data_list:
            return None
        
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        # 标准化
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        # 转换数值
        for col in ['open', 'close', 'high', 'low', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.rename(columns={
            'volume': 'volume',
            'amount': 'amount',
            'turn': 'turnover'
        })
        
        return df
        
    except Exception as e:
        print(f"baostock获取失败: {e}")
        return None


def get_a_stock_list() -> List[Dict]:
    """
    获取A股股票列表
    
    Returns:
        [{'code': xxx, 'name': xxx}, ...]
    """
    try:
        stock_df = ak.stock_info_a_code_name()
        if stock_df is not None and not stock_df.empty:
            return stock_df[['code', 'name']].to_dict('records')
    except:
        pass
    
    # 备用：返回常用股票
    return [
        {'code': '600519', 'name': '贵州茅台'},
        {'code': '000858', 'name': '五粮液'},
        {'code': '600036', 'name': '招商银行'},
        {'code': '600000', 'name': '浦发银行'},
        {'code': '601398', 'name': '工商银行'},
        {'code': '601318', 'name': '中国平安'},
        {'code': '600760', 'name': '中航沈飞'},
        {'code': '002594', 'name': '比亚迪'},
        {'code': '300750', 'name': '宁德时代'},
        {'code': '600703', 'name': '三安光电'},
    ]


def get_market_summary() -> Dict:
    """
    获取大盘指数行情
    
    Returns:
        {'上证指数': {'close': xxx, 'change': xxx}, ...}
    """
    try:
        # 上证指数
        sh_df = ak.stock_zh_index_spot()
        sh = sh_df[sh_df['代码'] == 'sh000001']
        
        # 深证成指
        sz_df = ak.stock_zh_index_spot()
        sz = sz_df[sz_df['代码'] == 'sz399001']
        
        return {
            '上证指数': {
                'close': float(sh.iloc[0]['最新价']) if not sh.empty else 0,
                'change': float(sh.iloc[0]['涨跌幅']) if not sh.empty else 0,
            },
            '深证成指': {
                'close': float(sz.iloc[0]['最新价']) if not sz.empty else 0,
                'change': float(sz.iloc[0]['涨跌幅']) if not sz.empty else 0,
            }
        }
    except:
        return {'上证指数': {'close': 0, 'change': 0}}


# 数据缓存
_data_cache = {}


def get_cached_data(
    symbol: str,
    days: int = 365
) -> Optional[pd.DataFrame]:
    """
    获取缓存的数据（用于演示）
    
    Args:
        symbol: 股票代码
        days: 数据天数
    
    Returns:
        DataFrame
    """
    cache_key = f"{symbol}_{days}"
    
    if cache_key in _data_cache:
        return _data_cache[cache_key]
    
    # 尝试获取真实数据
    df = get_stock_daily(symbol)
    
    if df is None or len(df) < days:
        # 使用模拟数据
        df = _generate_mock_data(symbol, days)
    
    _data_cache[cache_key] = df
    return df


def _generate_mock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """生成模拟数据"""
    np.random.seed(hash(symbol) % 2**32)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    base_price = 10 + hash(symbol) % 100
    
    # 生成趋势数据
    trend = np.linspace(0, 0.5, days)
    noise = np.cumsum(np.random.randn(days) * 0.02)
    prices = base_price * (1 + trend + noise * 0.1)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(days) * 0.01),
        'close': prices,
        'high': prices * (1 + np.abs(np.random.randn(days) * 0.02)),
        'low': prices * (1 - np.abs(np.random.randn(days) * 0.02)),
        'volume': np.random.randint(1000000, 10000000, days),
    }, index=dates)
    
    df['change_pct'] = df['close'].pct_change() * 100
    
    return df


if __name__ == "__main__":
    print("=" * 50)
    print("股票历史数据模块测试")
    print("=" * 50)
    
    # 测试获取真实数据
    print("\n获取股票日线数据...")
    df = get_stock_daily("600519")
    
    if df is not None and len(df) > 0:
        print(f"✅ 获取成功，共 {len(df)} 条数据")
        print(f"   时间范围: {df.index[0]} ~ {df.index[-1]}")
        print(f"   最新价: {df['close'].iloc[-1]:.2f}")
    else:
        print("❌ 获取失败，使用模拟数据")
        df = _generate_mock_data("600519")
        print(f"   模拟数据: {len(df)} 条")
    
    # 测试获取股票列表
    print("\n获取A股股票列表...")
    stocks = get_a_stock_list()[:5]
    print(f"   示例: {[s['code'] for s in stocks]}")
    
    # 测试大盘行情
    print("\n获取大盘行情...")
    summary = get_market_summary()
    print(f"   {summary}")
