#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quant_project - 量化选股助手（简化版）

核心价值：帮散户做出更理性的买入决策
页面：首页、选股、单股分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="quant_project - 量化选股助手",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 数据缓存 ====================
@st.cache_data(ttl=3600)
def get_stock_data(symbol: str, days: int = 365):
    """获取股票数据（优先缓存）"""
    try:
        from data_manager import get_stock_data_cached
        df = get_stock_data_cached(symbol, days)
        if df is not None and len(df) >= 30:
            return df
        
        from stock_data import get_stock_daily
        df = get_stock_daily(symbol)
        if df is not None and len(df) >= 30:
            return df.tail(days)
    except:
        pass
    return None


@st.cache_data(ttl=3600)
def calculate_indicators(df):
    """计算核心指标"""
    result = df.copy()
    
    # MA均线
    result['ma5'] = result['close'].rolling(5).mean()
    result['ma10'] = result['close'].rolling(10).mean()
    result['ma20'] = result['close'].rolling(20).mean()
    
    # MA20角度（核心指标）
    ma20 = result['ma20']
    result['ma20_angle'] = np.arctan(
        (ma20 - ma20.shift(1)) / (ma20.shift(1).replace(0, np.nan))
    ) * 180 / np.pi
    
    # RSI
    delta = result['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    result['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = result['close'].ewm(span=12, adjust=False).mean()
    ema26 = result['close'].ewm(span=26, adjust=False).mean()
    result['macd_diff'] = ema12 - ema26
    result['macd_dea'] = result['macd_diff'].ewm(span=9, adjust=False).mean()
    
    # 动量
    result['momentum_5'] = result['close'].pct_change(5)
    
    return result


def get_score_and_signal(row):
    """计算综合评分和信号"""
    ma20_angle = row.get('ma20_angle', 0)
    rsi = row.get('rsi', 50)
    macd_diff = row.get('macd_diff', 0)
    macd_dea = row.get('macd_dea', 0)
    
    if pd.isna(ma20_angle) or pd.isna(rsi):
        return 50, "数据不足", "neutral"
    
    # 计算评分（0-100）
    score = 50
    
    # MA20角度 (35分)
    if ma20_angle > 5:
        score += 35
    elif ma20_angle > 3:
        score += 25
    elif ma20_angle > 1:
        score += 15
    elif ma20_angle > 0:
        score += 5
    
    # RSI (30分)
    if 30 < rsi < 45:
        score += 30  # 超卖，反弹机会
    elif 45 <= rsi < 55:
        score += 20
    elif rsi < 30:
        score += 15  # 严重超卖
    
    # MACD (35分)
    if macd_diff > macd_dea:
        score += 35  # 金叉
    elif macd_diff > macd_dea * 0.8:
        score += 20
    
    score = min(score, 100)
    
    # 信号
    if score >= 70:
        signal = "🟢 买入"
    elif score >= 50:
        signal = "🟡 观望"
    else:
        signal = "🔴 卖出"
    
    return score, signal, "bullish" if score >= 60 else "bearish"


def get_signal_explain(row, score):
    """生成信号解释（说人话）"""
    ma20_angle = row.get('ma20_angle', 0)
    rsi = row.get('rsi', 50)
    macd_diff = row.get('macd_diff', 0)
    macd_dea = row.get('macd_dea', 0)
    
    reasons = []
    
    # MA20分析
    if ma20_angle > 3:
        reasons.append("✅ 均线向上，趋势走强")
    elif ma20_angle < -2:
        reasons.append("⚠️ 均线下行，趋势走弱")
    
    # RSI分析
    if rsi < 30:
        reasons.append("✅ RSI超卖，可能反弹")
    elif rsi > 70:
        reasons.append("⚠️ RSI超买，注意风险")
    
    # MACD分析
    if macd_diff > macd_dea:
        reasons.append("✅ MACD金叉，看涨")
    elif macd_diff < macd_dea:
        reasons.append("⚠️ MACD死叉，看跌")
    
    if not reasons:
        reasons.append("➖ 趋势不明朗，建议观望")
    
    return reasons


# ==================== 股票名称映射 ====================
STOCK_NAMES = {
    '600519': '贵州茅台', '000001': '平安银行', '601398': '工商银行',
    '600036': '招商银行', '600760': '中航沈飞', '002519': '银河电子',
    '600789': '鲁抗医药', '002498': '汉缆股份', '000858': '五粮液',
    '000651': '格力电器', '300750': '宁德时代', '002594': '比亚迪',
    '601318': '中国平安', '600030': '中信证券', '603259': '药明康德',
    '300760': '迈瑞医疗', '002410': '广联达', '300025': '华鹏飞',
    '600756': '浪潮软件', '300188': '美亚柏科', '000034': '神州数码',
    '600588': '用友网络', '002475': '长盈精密', '000063': '中兴通讯',
    '002415': '海康威视', '300124': '汇川技术', '600703': '三安光电',
}

# 热门股票池
HOT_STOCKS = [
    ('600519', '贵州茅台'), ('000001', '平安银行'), ('601398', '工商银行'),
    ('600036', '招商银行'), ('600760', '中航沈飞'), ('002519', '银河电子'),
    ('600789', '鲁抗医药'), ('000858', '五粮液'), ('000651', '格力电器'),
    ('300750', '宁德时代'),
]


# ==================== 页面函数 ====================

def show_home():
    """首页 - 今日推荐"""
    st.markdown("""
    <style>
    .big-score { font-size: 48px; font-weight: bold; text-align: center; }
    .signal-buy { color: #22c55e; }
    .signal-sell { color: #ef4444; }
    .signal-hold { color: #f59e0b; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<p class="main-header">📈 量化选股助手</p>', unsafe_allow_html=True)
    
    # 快速搜索
    col_search, col_btn = st.columns([3, 1])
    with col_search:
        search_code = st.text_input("🔍 输入股票代码搜索", placeholder="如: 600519")
    with col_btn:
        if st.button("搜索", type="primary"):
            if search_code:
                st.session_state['target_stock'] = search_code
                st.rerun()
    
    st.markdown("---")
    
    # 今日推荐
    st.subheader("🎯 今日推荐")
    
    # 批量获取推荐股票数据
    recommendations = []
    progress_bar = st.progress(0)
    
    for i, (code, name) in enumerate(HOT_STOCKS[:6]):
        progress_bar.progress((i+1)/6)
        
        df = get_stock_data(code, days=100)
        if df is not None and len(df) >= 20:
            df = calculate_indicators(df)
            latest = df.iloc[-1]
            score, signal, _ = get_score_and_signal(latest)
            
            recommendations.append({
                'code': code,
                'name': name,
                'score': score,
                'signal': signal,
                'price': latest['close'],
                'change': latest.get('momentum_5', 0) * 100
            })
    
    progress_bar.empty()
    
    # 显示推荐卡片
    if recommendations:
        # 按评分排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        cols = st.columns(3)
        for i, rec in enumerate(recommendations[:3]):
            with cols[i]:
                signal_color = "#22c55e" if "买入" in rec['signal'] else "#f59e0b" if "观望" in rec['signal'] else "#ef4444"
                
                st.markdown(f"""
                <div style="background: #1e293b; padding: 20px; border-radius: 10px; margin: 10px 0;">
                    <h3 style="margin: 0;">{rec['code']}</h3>
                    <p style="color: #94a3b8; margin: 5px 0;">{rec['name']}</p>
                    <div class="big-score">{rec['score']}</div>
                    <p style="color: {signal_color}; font-size: 18px; margin: 10px 0;">{rec['signal']}</p>
                    <p style="color: #64748b;">现价: {rec['price']:.2f} | 5日: {rec['change']:+.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"查看详情 →", key=f"view_{rec['code']}"):
                    st.session_state['target_stock'] = rec['code']
                    st.rerun()
    
    # 更多推荐
    st.markdown("---")
    st.subheader("📊 热门股票榜")
    
    other_recs = recommendations[3:] if len(recommendations) > 3 else []
    if other_recs:
        data = []
        for r in other_recs:
            signal_emoji = "🟢" if "买入" in r['signal'] else "🟡" if "观望" in r['signal'] else "🔴"
            data.append({
                "代码": r['code'],
                "名称": r['name'],
                "评分": r['score'],
                "信号": f"{signal_emoji} {r['signal']}",
                "现价": f"{r['price']:.2f}",
                "5日涨跌": f"{r['change']:+.1f}%"
            })
        
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)


def show_selector():
    """选股页面"""
    st.markdown('<p class="main-header">🔍 智能选股</p>', unsafe_allow_html=True)
    
    # 筛选条件
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score = st.selectbox("最低评分", [0, 30, 50, 60, 70, 80], index=4)
    with col2:
        signal_filter = st.selectbox("信号筛选", ["全部", "买入", "观望", "卖出"])
    with col3:
        if st.button("🔍 开始选股", type="primary"):
            st.rerun()
    
    st.markdown("---")
    
    # 获取所有热门股票数据
    all_stocks = []
    progress_bar = st.progress(0)
    
    for i, (code, name) in enumerate(HOT_STOCKS):
        progress_bar.progress((i+1)/len(HOT_STOCKS))
        
        df = get_stock_data(code, days=100)
        if df is not None and len(df) >= 20:
            df = calculate_indicators(df)
            latest = df.iloc[-1]
            score, signal, _ = get_score_and_signal(latest)
            
            # 筛选
            if score < min_score:
                continue
            if signal_filter != "全部" and signal_filter not in signal:
                continue
            
            all_stocks.append({
                'code': code,
                'name': name,
                'score': score,
                'signal': signal,
                'price': latest['close'],
                'ma20_angle': latest.get('ma20_angle', 0),
                'rsi': latest.get('rsi', 50),
                'momentum': latest.get('momentum_5', 0) * 100
            })
    
    progress_bar.empty()
    
    # 排序显示
    all_stocks.sort(key=lambda x: x['score'], reverse=True)
    
    st.subheader(f"📊 选股结果 ({len(all_stocks)}只)")
    
    if all_stocks:
        # 表格显示
        data = []
        for s in all_stocks:
            signal_emoji = "🟢" if "买入" in s['signal'] else "🟡" if "观望" in s['signal'] else "🔴"
            data.append({
                "代码": s['code'],
                "名称": s['name'],
                "评分": s['score'],
                "MA20角度": f"{s['ma20_angle']:.1f}°",
                "RSI": f"{s['rsi']:.0f}",
                "5日涨跌": f"{s['momentum']:+.1f}%",
                "信号": f"{signal_emoji} {s['signal']}",
            })
        
        st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)
    else:
        st.info("没有符合条件的股票")


def show_stock_detail(code):
    """单股分析页面"""
    name = STOCK_NAMES.get(code, code)
    
    st.markdown(f'<p class="main-header">📈 {code} {name}</p>', unsafe_allow_html=True)
    
    # 获取数据
    df = get_stock_data(code, days=200)
    if df is None or len(df) < 20:
        st.error(f"无法获取 {code} 的数据")
        return
    
    df = calculate_indicators(df)
    latest = df.iloc[-1]
    score, signal, trend = get_score_and_signal(latest)
    explanations = get_signal_explain(latest, score)
    
    # 综合评分（大字展示）
    st.markdown("---")
    
    col_score, col_signal = st.columns([1, 2])
    
    with col_score:
        signal_color = "#22c55e" if "买入" in signal else "#f59e0b" if "观望" in signal else "#ef4444"
        
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: #1e293b; border-radius: 15px;">
            <p style="color: #94a3b8; font-size: 18px;">综合评分</p>
            <div style="font-size: 72px; font-weight: bold; color: {signal_color};">{score}</div>
            <div style="font-size: 24px; color: {signal_color}; margin-top: 10px;">{signal}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_signal:
        st.subheader("📋 信号解读")
        
        for exp in explanations:
            st.write(exp)
        
        # 技术指标详情
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ma20_angle = latest.get('ma20_angle', 0)
            st.metric("MA20角度", f"{ma20_angle:.2f}°", 
                      "↑" if ma20_angle > 0 else "↓")
        
        with col2:
            rsi = latest.get('rsi', 50)
            rsi_status = "超买" if rsi > 70 else "超卖" if rsi < 30 else "正常"
            st.metric("RSI", f"{rsi:.1f}", rsi_status)
        
        with col3:
            macd_diff = latest.get('macd_diff', 0)
            macd_dea = latest.get('macd_dea', 0)
            macd_status = "金叉" if macd_diff > macd_dea else "死叉"
            st.metric("MACD", f"{macd_diff:.2f}", macd_status)
    
    # K线图
    st.markdown("---")
    st.subheader("📊 K线走势")
    
    fig = go.Figure()
    
    # K线
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线'
    ))
    
    # MA20
    fig.add_trace(go.Scatter(
        x=df.index, y=df['ma20'],
        mode='lines', name='MA20',
        line=dict(color='#2196F3', width=2)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 快捷操作
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ 加入自选", type="primary", use_container_width=True):
            st.success(f"已添加 {code} {name}")
    
    with col2:
        if st.button("📊 回测验证", use_container_width=True):
            st.session_state['page'] = 'backtest'
            st.session_state['target_stock'] = code
            st.rerun()
    
    with col3:
        if st.button("🔮 LSTM预测", use_container_width=True):
            st.session_state['page'] = 'lstm'
            st.session_state['target_stock'] = code
            st.rerun()


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 侧边栏
    st.sidebar.title("📈 quant_project")
    st.sidebar.markdown("---")
    
    # 导航
    page = st.sidebar.radio(
        "导航",
        ["首页", "选股", "单股分析"]
    )
    
    # 如果有目标股票，切换到单股分析
    if 'target_stock' in st.session_state and st.session_state['target_stock']:
        page = "单股分析"
        code = st.session_state['target_stock']
    
    # 系统信息
    st.sidebar.markdown("---")
    st.sidebar.subheader("ℹ️ 系统信息")
    
    try:
        test_df = get_stock_data('600519')
        data_status = "📈 真实数据" if test_df is not None else "📊 数据异常"
    except:
        data_status = "📊 未知"
    
    st.sidebar.info(f"版本: v2.0\n数据: {data_status}")
    
    # 根据导航显示页面
    if page == "首页":
        show_home()
    elif page == "选股":
        show_selector()
    elif page == "单股分析":
        # 默认显示贵州茅台
        default_code = st.session_state.get('target_stock', '600519')
        
        col_search, _ = st.columns([3, 1])
        with col_search:
            code_input = st.text_input("输入股票代码", value=default_code)
            if code_input:
                show_stock_detail(code_input)
            else:
                show_stock_detail('600519')


if __name__ == "__main__":
    main()
