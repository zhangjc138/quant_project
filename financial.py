#!/usr/bin/env python3
"""
财务因子选股模块

提供股票财务数据获取和筛选功能
支持 PE、PB、ROE、营收增速、净利润增速等因子
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import akshare as ak
from datetime import datetime


@dataclass
class FinancialMetrics:
    """财务指标数据结构"""
    symbol: str           # 股票代码
    name: str            # 股票名称
    pe: float            # 市盈率 PE
    pb: float            # 市净率 PB
    roe: float           # 净资产收益率 ROE (%)
    revenue_growth: float # 营收增速 (%)
    profit_growth: float  # 净利润增速 (%)
    gross_margin: float   # 毛利率 (%)
    debt_ratio: float    # 资产负债率 (%)
    market_cap: float    # 总市值 (亿元)
    circulating_cap: float # 流通市值 (亿元)
    report_date: str     # 报告期


def get_stock_financials(symbol: str) -> Optional[FinancialMetrics]:
    """
    获取单只股票财务指标
    
    Args:
        symbol: 股票代码 (如: 600519)
    
    Returns:
        FinancialMetrics 对象，失败返回 None
    """
    try:
        # 标准化股票代码
        code = symbol.replace(".", "").replace("SH", "").replace("SZ", "")
        
        # 使用 akshare 获取财务数据
        # 股票实时行情数据
        stock_info = ak.stock_info_a_code_name()
        if stock_info is None or stock_info.empty:
            return None
        
        # 获取当前价格和市值
        df = ak.stock_zh_a_spot_em()
        stock_data = df[df['代码'] == code]
        
        if stock_data.empty:
            return None
        
        current_price = stock_data['最新价'].values[0]
        market_cap = stock_data['总市值'].values[0] / 100000000  # 转换为亿元
        circulating_cap = stock_data['流通市值'].values[0] / 100000000
        
        # 获取财务数据
        try:
            financial = ak.stock_financial_analysis_indicator(symbol=f"sh{code}" if code.startswith('6') else f"sz{code}")
            if financial is None or financial.empty:
                # 使用默认指标
                return FinancialMetrics(
                    symbol=code,
                    name="",
                    pe=0,
                    pb=0,
                    roe=0,
                    revenue_growth=0,
                    profit_growth=0,
                    gross_margin=0,
                    debt_ratio=0,
                    market_cap=market_cap,
                    circulating_cap=circulating_cap,
                    report_date=""
                )
            
            # 取最新数据
            latest = financial.iloc[0]
            
            return FinancialMetrics(
                symbol=code,
                name=latest.get('股票名称', ''),
                pe=float(latest.get('市盈率', 0)),
                pb=float(latest.get('市净率', 0)),
                roe=float(latest.get('净资产收益率', 0)),
                revenue_growth=float(latest.get('营收增长率', 0)),
                profit_growth=float(latest.get('净利润增长率', 0)),
                gross_margin=float(latest.get('毛利率', 0)),
                debt_ratio=float(latest.get('资产负债率', 0)),
                market_cap=market_cap,
                circulating_cap=circulating_cap,
                report_date=str(latest.get('报告期', ''))
            )
        except Exception as e:
            print(f"获取财务数据失败: {symbol}, 错误: {e}")
            return None
            
    except Exception as e:
        print(f"获取财务数据异常: {symbol}, 错误: {e}")
        return None


def filter_by_pe(metrics_list: List[FinancialMetrics], min_pe: float = 0, max_pe: float = 100) -> List[FinancialMetrics]:
    """
    按 PE 市盈率筛选
    
    Args:
        metrics_list: 财务指标列表
        min_pe: 最小 PE
        max_pe: 最大 PE
    
    Returns:
        符合条件的股票列表
    """
    return [m for m in metrics_list if min_pe <= m.pe <= max_pe]


def filter_by_pb(metrics_list: List[FinancialMetrics], min_pb: float = 0, max_pb: float = 10) -> List[FinancialMetrics]:
    """
    按 PB 市净率筛选
    
    Args:
        metrics_list: 财务指标列表
        min_pb: 最小 PB
        max_pb: 最大 PB
    
    Returns:
        符合条件的股票列表
    """
    return [m for m in metrics_list if min_pb <= m.pb <= max_pb]


def filter_by_roe(metrics_list: List[FinancialMetrics], min_roe: float = 0) -> List[FinancialMetrics]:
    """
    按 ROE 净资产收益率筛选
    
    Args:
        metrics_list: 财务指标列表
        min_roe: 最小 ROE (%)
    
    Returns:
        符合条件的股票列表
    """
    return [m for m in metrics_list if m.roe >= min_roe]


def filter_by_revenue_growth(metrics_list: List[FinancialMetrics], min_growth: float = -50, max_growth: float = 100) -> List[FinancialMetrics]:
    """
    按营收增速筛选
    
    Args:
        metrics_list: 财务指标列表
        min_growth: 最小增速 (%)
        max_growth: 最大增速 (%)
    
    Returns:
        符合条件的股票列表
    """
    return [m for m in metrics_list if min_growth <= m.revenue_growth <= max_growth]


def filter_by_profit_growth(metrics_list: List[FinancialMetrics], min_growth: float = -50, max_growth: float = 100) -> List[FinancialMetrics]:
    """
    按净利润增速筛选
    
    Args:
        metrics_list: 财务指标列表
        min_growth: 最小增速 (%)
        max_growth: 最大增速 (%)
    
    Returns:
        符合条件的股票列表
    """
    return [m for m in metrics_list if min_growth <= m.profit_growth <= max_growth]


def filter_by_gross_margin(metrics_list: List[FinancialMetrics], min_margin: float = 0, max_margin: float = 100) -> List[FinancialMetrics]:
    """
    按毛利率筛选
    
    Args:
        metrics_list: 财务指标列表
        min_margin: 最小毛利率 (%)
        max_margin: 最大毛利率 (%)
    
    Returns:
        符合条件的股票列表
    """
    return [m for m in metrics_list if min_margin <= m.gross_margin <= max_margin]


def filter_by_debt_ratio(metrics_list: List[FinancialMetrics], min_ratio: float = 0, max_ratio: float = 100) -> List[FinancialMetrics]:
    """
    按资产负债率筛选
    
    Args:
        metrics_list: 财务指标列表
        min_ratio: 最小资产负债率 (%)
        max_ratio: 最大资产负债率 (%)
    
    Returns:
        符合条件的股票列表
    """
    return [m for m in metrics_list if min_ratio <= m.debt_ratio <= max_ratio]


def filter_financials(
    metrics_list: List[FinancialMetrics],
    min_pe: float = 0,
    max_pe: float = 100,
    min_pb: float = 0,
    max_pb: float = 10,
    min_roe: float = 0,
    min_revenue_growth: float = -50,
    max_revenue_growth: float = 100,
    min_profit_growth: float = -50,
    max_profit_growth: float = 100,
    min_gross_margin: float = 0,
    max_gross_margin: float = 100,
    min_debt_ratio: float = 0,
    max_debt_ratio: float = 100
) -> List[FinancialMetrics]:
    """
    综合财务因子筛选
    
    Args:
        metrics_list: 财务指标列表
        min_pe: 最小 PE
        max_pe: 最大 PE
        min_pb: 最小 PB
        max_pb: 最大 PB
        min_roe: 最小 ROE (%)
        min_revenue_growth: 最小营收增速 (%)
        max_revenue_growth: 最大营收增速 (%)
        min_profit_growth: 最小净利润增速 (%)
        max_profit_growth: 最大净利润增速 (%)
        min_gross_margin: 最小毛利率 (%)
        max_gross_margin: 最大毛利率 (%)
        min_debt_ratio: 最小资产负债率 (%)
        max_debt_ratio: 最大资产负债率 (%)
    
    Returns:
        符合条件的股票列表
    """
    result = metrics_list
    
    result = filter_by_pe(result, min_pe, max_pe)
    result = filter_by_pb(result, min_pb, max_pb)
    result = filter_by_roe(result, min_roe)
    result = filter_by_revenue_growth(result, min_revenue_growth, max_revenue_growth)
    result = filter_by_profit_growth(result, min_profit_growth, max_profit_growth)
    result = filter_by_gross_margin(result, min_gross_margin, max_gross_margin)
    result = filter_by_debt_ratio(result, min_debt_ratio, max_debt_ratio)
    
    return result


def get_a_stock_financials_batch(symbols: List[str], max_workers: int = 10) -> Dict[str, FinancialMetrics]:
    """
    批量获取多只股票财务数据
    
    Args:
        symbols: 股票代码列表
        max_workers: 并行线程数
    
    Returns:
        股票代码 -> FinancialMetrics 的映射
    """
    import concurrent.futures
    
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(get_stock_financials, symbol): symbol 
            for symbol in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                metrics = future.result()
                if metrics:
                    results[symbol] = metrics
            except Exception as e:
                print(f"处理股票 {symbol} 时出错: {e}")
    
    return results


if __name__ == "__main__":
    # 测试
    print("财务因子选股模块测试")
    
    # 测试单只股票
    test_symbols = ["600519", "000858", "600036"]
    
    print("\n获取股票财务数据:")
    for symbol in test_symbols:
        metrics = get_stock_financials(symbol)
        if metrics:
            print(f"  {symbol}: PE={metrics.pe:.2f}, PB={metrics.pb:.2f}, ROE={metrics.roe:.2f}%")
        else:
            print(f"  {symbol}: 获取失败")
