#!/usr/bin/env python3
"""
行业板块数据模块

提供A股行业板块分类数据，支持按行业筛选股票
支持 akshare 实时数据获取
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import akshare as ak


def get_stock_industry_from_akshare(symbol: str) -> str:
    """
    从 akshare 获取股票所属行业
    
    Args:
        symbol: 股票代码
    
    Returns:
        行业名称
    """
    try:
        stock_df = ak.stock_info_a_code_name()
        if stock_df is not None and not stock_df.empty:
            code = symbol.replace(".", "").replace("SH", "").replace("SZ", "")
            row = stock_df[stock_df['code'] == code]
            if not row.empty:
                return row.iloc[0].get('industry', '未知')
    except Exception as e:
        print(f"获取行业数据失败: {e}")
    return "未知"


def get_industry_list() -> List[str]:
    """
    获取行业列表（从 akshare）
    
    Returns:
        行业名称列表
    """
    try:
        stock_df = ak.stock_info_a_code_name()
        if stock_df is not None and 'industry' in stock_df.columns:
            industries = stock_df['industry'].dropna().unique().tolist()
            if industries:
                return sorted([i for i in industries if i])
    except Exception as e:
        print(f"获取行业列表失败: {e}")
    
    # 默认列表
    return [
        "科技", "消费", "医药", "金融", "地产", "周期",
        "制造", "能源", "军工", "新能源", "半导体",
        "新能源汽车", "人工智能", "云计算", "生物医药",
        "新材料", "数字经济", "智能制造", "绿色能源", "高端装备"
    ]


def get_stocks_by_industry(industry: str) -> List[Dict]:
    """
    获取某个行业的所有股票
    
    Args:
        industry: 行业名称
    
    Returns:
        股票列表 [{'code': xxx, 'name': xxx}, ...]
    """
    try:
        stock_df = ak.stock_info_a_code_name()
        if stock_df is not None and 'industry' in stock_df.columns:
            industry_stocks = stock_df[stock_df['industry'] == industry]
            if not industry_stocks.empty:
                return industry_stocks[['code', 'name']].to_dict('records')
    except Exception as e:
        print(f"获取行业股票失败: {e}")
    return []


def get_all_industry_stocks() -> Dict[str, List[Dict]]:
    """
    获取所有行业的股票
    
    Returns:
        行业 -> 股票列表 的映射
    """
    result = {}
    industries = get_industry_list()
    
    for industry in industries:
        stocks = get_stocks_by_industry(industry)
        if stocks:
            result[industry] = stocks
    
    return result


# 申万一级行业分类
INDUSTRYClassification = {
    "801010": "农林牧渔",
    "801020": "化工",
    "801030": "钢铁",
    "801040": "有色金属",
    "801050": "综合",
    "801060": "商业贸易",
    "801080": "电子",
    "801090": "食品饮料",
    "801100": "纺织服装",
    "801110": "轻工制造",
    "801120": "医药生物",
    "801130": "电力及公用事业",
    "801140": "交通运输",
    "801150": "房地产",
    "801160": "金融服务",
    "801180": "建筑材料",
    "801190": "建筑装饰",
    "801210": "家用电器",
    "801220": "汽车",
    "801230": "商贸零售",
    "801250": "休闲服务",
    "801260": "国防军工",
    "801270": "计算机",
    "801280": "传媒",
    "801290": "通信",
}

# 常用行业板块（简化版）
INDUSTRY_LIST = [
    "科技", "消费", "医药", "金融", "地产", "周期",
    "制造", "能源", "军工", "新能源", "半导体",
    "新能源汽车", "人工智能", "云计算", "生物医药",
    "新材料", "数字经济", "智能制造", "绿色能源", "高端装备"
]

# 股票代码到行业的映射（示例）
STOCK_INDUSTRY_MAPPING = {
    "600703": "半导体", "600460": "半导体", "002475": "半导体",
    "600519": "食品饮料", "000858": "食品饮料", "603288": "食品饮料",
    "600760": "国防军工", "600893": "国防军工",
    "002594": "新能源汽车", "300750": "新能源汽车",
    "601398": "银行", "601318": "保险", "600030": "非银金融",
    "000002": "房地产", "600048": "房地产",
}


def get_stock_industry(stock_code: str) -> str:
    """获取股票所属行业"""
    code = stock_code.replace(".", "").replace("SH", "").replace("SZ", "")
    if len(code) == 6:
        return STOCK_INDUSTRY_MAPPING.get(code, "未知")
    return "未知"


def get_industry_stocks(industry: str) -> list:
    """获取某个行业的所有股票代码"""
    return [
        code for code, ind in STOCK_INDUSTRY_MAPPING.items()
        if ind == industry or industry in ind
    ]


if __name__ == "__main__":
    print("=" * 50)
    print("行业板块数据模块测试")
    print("=" * 50)
    
    print("\n获取行业列表...")
    industries = get_industry_list()
    print(f"共 {len(industries)} 个行业")
    print(f"示例: {industries[:5]}")
    
    print("\n获取科技行业股票...")
    tech_stocks = get_stocks_by_industry("科技")
    print(f"科技行业共 {len(tech_stocks)} 只股票")
    for stock in tech_stocks[:5]:
        print(f"  {stock['code']} - {stock['name']}")
