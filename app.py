#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quant_project - Streamlit Web界面
交互式Web界面，让用户无需命令行即可使用量化选股工具

功能页面:
- 📈 选股页面：输入股票代码/批量扫描、显示MA20角度、RSI、MACD、信号
- 📊 回测页面：选择股票、时间范围、回测参数、显示收益曲线、统计指标
- 🤖 ML预测页面：选择模型、显示预测结果、特征重要性
- ⭐ 评分系统页面：综合评分、各维度得分、可视化
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入付费版模块
try:
    from scoring_system import ScoringSystem, ScoreResult, SignalLevel, print_score_result
    PREMIUM_FEATURES = True
except ImportError:
    PREMIUM_FEATURES = False

try:
    from ml_selector import MLSelector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from smart_stock_picker import SmartStockPicker, A_SHARE_POOL
    PICKER_AVAILABLE = True
except ImportError:
    PICKER_AVAILABLE = False

# 尝试导入开源版模块
try:
    from stock_strategy import StockSelector, calculate_rsi, calculate_macd
    OPEN_SOURCE_AVAILABLE = True
except ImportError:
    OPEN_SOURCE_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="quant_project - 量化选股工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 5px;
    }
    .buy-signal {
        color: #28a745;
        font-weight: bold;
    }
    .sell-signal {
        color: #dc3545;
        font-weight: bold;
    }
    .hold-signal {
        color: #ffc107;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ==================== 辅助函数 ====================

@st.cache_data(ttl=3600)
def generate_mock_data(symbol, days=200):
    """生成模拟数据用于演示"""
    np.random.seed(hash(symbol) % 2**32)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # 基于股票代码生成不同的走势
    base_price = 10 + (hash(symbol) % 100)
    trend = (hash(symbol) % 20 - 10) * 0.001
    close = base_price + np.cumsum(np.random.randn(days) * 2 + trend)
    
    df = pd.DataFrame({
        'open': close - np.random.uniform(-0.5, 0.5, days),
        'high': close + np.random.uniform(0, 2, days),
        'low': close - np.random.uniform(0, 2, days),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, days)
    }, index=dates)
    
    return df


@st.cache_data(ttl=3600)
def calculate_indicators(df):
    """计算技术指标"""
    result = df.copy()
    
    # 均线
    result['ma5'] = result['close'].rolling(5).mean()
    result['ma10'] = result['close'].rolling(10).mean()
    result['ma20'] = result['close'].rolling(20).mean()
    result['ma60'] = result['close'].rolling(60).mean()
    
    # MA20角度
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
    result['macd_hist'] = result['macd_diff'] - result['macd_dea']
    
    # 动量
    result['momentum_5'] = result['close'].pct_change(5)
    result['momentum_10'] = result['close'].pct_change(10)
    
    # 成交量
    result['volume_ma5'] = result['volume'].rolling(5).mean()
    result['volume_ratio'] = result['volume'] / result['volume_ma5']
    
    # BOLL 布林带
    boll_middle = result['close'].rolling(window=20).mean()
    boll_std = result['close'].rolling(window=20).std()
    result['boll_upper'] = boll_middle + 2 * boll_std
    result['boll_lower'] = boll_middle - 2 * boll_std
    result['boll_width'] = result['boll_upper'] - result['boll_lower']
    result['boll_position'] = (result['close'] - result['boll_lower']) / \
        result['boll_width'].replace(0, np.nan)
    
    # KDJ 随机指标
    low_min = result['low'].rolling(window=9).min()
    high_max = result['high'].rolling(window=9).max()
    rsv = ((result['close'] - low_min) / (high_max - low_min).replace(0, np.nan) * 100).fillna(50)
    result['kdj_k'] = rsv.rolling(window=3).mean()
    result['kdj_d'] = result['kdj_k'].rolling(window=3).mean()
    result['kdj_j'] = 3 * result['kdj_k'] - 2 * result['kdj_d']
    
    return result


def get_signal_from_indicators(row):
    """根据指标生成信号"""
    ma20_angle = row.get('ma20_angle', 0)
    rsi = row.get('rsi', 50)
    macd_diff = row.get('macd_diff', 0)
    macd_dea = row.get('macd_dea', 0)
    boll_position = row.get('boll_position', 0.5)
    kdj_k = row.get('kdj_k', 50)
    kdj_d = row.get('kdj_d', 50)
    
    if pd.isna(ma20_angle) or pd.isna(rsi):
        return "HOLD", "数据不足"
    
    # MA20角度判断
    if ma20_angle > 3:
        trend_signal = "BUY"
    elif ma20_angle < 0:
        trend_signal = "SELL"
    else:
        trend_signal = "HOLD"
    
    # RSI判断
    if rsi > 70:
        rsi_signal = "超买"
    elif rsi < 30:
        rsi_signal = "超卖"
    else:
        rsi_signal = "中性"
    
    # MACD判断
    if macd_diff > macd_dea:
        macd_signal = "金叉"
    elif macd_diff < macd_dea:
        macd_signal = "死叉"
    else:
        macd_signal = "中性"
    
    # BOLL判断
    if pd.isna(boll_position):
        boll_signal = "中性"
    elif boll_position >= 0.9:
        boll_signal = "超买"
    elif boll_position <= 0.1:
        boll_signal = "超卖"
    else:
        boll_signal = "中性"
    
    # KDJ判断
    kdj_prev_k = row.get('kdj_k', 50) if 'kdj_k' in row else 50
    kdj_prev_d = row.get('kdj_d', 50) if 'kdj_d' in row else 50
    
    if pd.isna(kdj_k) or pd.isna(kdj_d):
        kdj_signal = "中性"
    elif kdj_k >= 80 and kdj_d >= 80:
        kdj_signal = "超买"
    elif kdj_k <= 20 and kdj_d <= 20:
        kdj_signal = "超卖"
    elif kdj_prev_k <= kdj_prev_d and kdj_k > kdj_d:
        kdj_signal = "金叉"
    elif kdj_prev_k >= kdj_prev_d and kdj_k < kdj_d:
        kdj_signal = "死叉"
    elif kdj_k > kdj_d:
        kdj_signal = "多头"
    else:
        kdj_signal = "空头"
    
    # 综合信号
    if trend_signal == "BUY" and macd_signal == "金叉":
        signal = "🟢 强力买入"
    elif trend_signal == "BUY":
        signal = "🟢 买入"
    elif trend_signal == "SELL":
        signal = "🔴 卖出"
    else:
        signal = "🟡 持有"
    
    details = f"{trend_signal} | {rsi_signal} | {macd_signal} | {boll_signal} | {kdj_signal}"
    
    return signal, details


def plot_candlestick_with_indicators(df, symbol="股票"):
    """绘制K线图和指标"""
    if df is None or len(df) < 20:
        return None
    
    # 创建子图
    fig = go.Figure()
    
    # K线图
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # MA均线
    if 'ma20' in df.columns and df['ma20'].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df['ma20'],
            mode='lines', name='MA20',
            line=dict(color='#2196F3', width=1.5)
        ))
    
    if 'ma60' in df.columns and df['ma60'].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df['ma60'],
            mode='lines', name='MA60',
            line=dict(color='#FF9800', width=1.5)
        ))
    
    # 布局设置
    fig.update_layout(
        title=f'{symbol} K线图',
        xaxis_title='日期',
        yaxis_title='价格',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    return fig


# ==================== 页面函数 ====================

def show_stock_selector():
    """选股页面"""
    st.markdown('<p class="main-header">📈 智能选股</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("选股参数")
        
        input_method = st.radio("输入方式", ["单只股票", "批量扫描"])
        
        if input_method == "单只股票":
            symbol = st.text_input("股票代码", value="600519", help="如: 600519 (贵州茅台)")
            symbols = [symbol]
        else:
            # 使用预设股票池
            stock_pool = list(A_SHARE_POOL.items()) if PICKER_AVAILABLE else [
                ('600519', '贵州茅台'), ('600036', '招商银行'), ('000858', '五粮液'),
                ('000651', '格力电器'), ('600030', '中信证券')
            ]
            selected = st.multiselect(
                "选择股票",
                options=[s[0] for s in stock_pool],
                default=[s[0] for s in stock_pool[:5]],
                format_func=lambda x: f"{x} - {dict(stock_pool).get(x, '')}"
            )
            symbols = selected if selected else [stock_pool[0][0]]
        
        # 筛选参数
        # 行业筛选
        industry_options = ["全部"] + sorted([
            "科技", "消费", "医药", "金融", "地产", "周期", 
            "制造", "能源", "军工", "新能源", "半导体",
            "新能源汽车", "人工智能", "云计算", "生物医药",
            "新材料", "数字经济", "智能制造", "绿色能源", "高端装备"
        ])
        selected_industry = st.selectbox("行业板块", industry_options, help="选择行业进行筛选")
        
        # 定义行业股票映射（示例数据）
        INDUSTRY_STOCKS = {
            "科技": ["600703", "002475", "000063", "002415", "300124", "002410"],
            "消费": ["600519", "000858", "603288", "000651", "000333"],
            "医药": ["603259", "300760", "002252", "600085"],
            "金融": ["601398", "601318", "600030", "600000"],
            "地产": ["000002", "600048", "600383"],
            "新能源": ["002594", "300750", "600438", "002466"],
            "半导体": ["600703", "600460", "002475", "688008"],
            "军工": ["600893", "600760", "600150"],
            "人工智能": ["002415", "300124", "002410", "300033"],
            "云计算": ["002410", "300025", "600756", "300188"],
        }
        
        # 根据选择调整股票池
        if selected_industry != "全部" and selected_industry in INDUSTRY_STOCKS:
            # 只显示选中行业的股票
            industry_stocks = INDUSTRY_STOCKS[selected_industry]
            stock_pool = [(s, s) for s in industry_stocks]
        
        with st.expander("基本面筛选", expanded=False):
            # PE 市盈率
            col_pe1, col_pe2 = st.columns(2)
            with col_pe1:
                pe_min = st.number_input("PE最小", value=0, min_value=0, key="pe_min")
            with col_pe2:
                pe_max = st.number_input("PE最大", value=100, min_value=0, key="pe_max")
            
            # PB 市净率
            col_pb1, col_pb2 = st.columns(2)
            with col_pb1:
                pb_min = st.number_input("PB最小", value=0, min_value=0, key="pb_min")
            with col_pb2:
                pb_max = st.number_input("PB最大", value=10, min_value=0, key="pb_max")
            
            # ROE 净资产收益率
            min_roe = st.number_input("最小ROE (%)", value=0, min_value=0, max_value=100)
            
            # 营收增速
            col_rev1, col_rev2 = st.columns(2)
            with col_rev1:
                rev_growth_min = st.number_input("最小营收增速 (%)", value=-50, key="rev_min")
            with col_rev2:
                rev_growth_max = st.number_input("最大营收增速 (%)", value=100, key="rev_max")
            
            # 净利润增速
            col_pro1, col_pro2 = st.columns(2)
            with col_pro1:
                profit_growth_min = st.number_input("最小净利润增速 (%)", value=-50, key="profit_min")
            with col_pro2:
                profit_growth_max = st.number_input("最大净利润增速 (%)", value=100, key="profit_max")
        
        # 推送设置
        with st.expander("🔔 推送设置", expanded=False):
            enable_push = st.toggle("启用推送通知", value=False, help="开启后将通过配置的渠道发送信号通知")
            
            if enable_push:
                push_channel = st.selectbox("推送渠道", ["飞书", "微信"], help="选择推送方式")
                
                if push_channel == "飞书":
                    webhook_url = st.text_input("飞书Webhook URL", type="password", help="填入飞书群机器人Webhook地址")
                elif push_channel == "微信":
                    push_method = st.selectbox("微信推送方式", ["Server酱", "酷推"], help="选择微信推送方式")
                    if push_method == "Server酱":
                        wechat_key = st.text_input("Server酱 SCKEY", type="password", help="填入Server酱的SCKEY")
                    else:
                        wechat_key = st.text_input("酷推 Skey", type="password", help="填入酷推的Skey")
        
        scan_button = st.button("🔍 开始选股", type="primary")
    
    with col2:
        if scan_button or input_method == "单只股票":
            results = []
            
            for sym in symbols:
                # 生成/加载数据
                df = generate_mock_data(sym)
                df = calculate_indicators(df)
                
                if len(df) >= 20:
                    latest = df.iloc[-1]
                    signal, desc = get_signal_from_indicators(latest)
                    
                    # 计算简单评分
                    ma20_angle = latest.get('ma20_angle', 0)
                    rsi = latest.get('rsi', 50)
                    momentum = latest.get('momentum_5', 0) * 100
                    
                    # 简单评分 (0-100)
                    score = 50
                    if ma20_angle > 3:
                        score += min(ma20_angle * 3, 20)
                    if 30 < rsi < 70:
                        score += 10
                    if momentum > 0:
                        score += min(momentum * 2, 20)
                    score = min(score, 100)
                    
                    name = dict(A_SHARE_POOL).get(sym, sym) if PICKER_AVAILABLE else sym
                    results.append({
                        '代码': sym,
                        '名称': name,
                        '评分': round(score, 1),
                        'MA20角度': round(ma20_angle, 2) if pd.notna(ma20_angle) else 0,
                        'RSI': round(rsi, 1) if pd.notna(rsi) else 50,
                        '5日涨幅': f"{momentum:.2f}%",
                        '信号': signal,
                        '详情': desc,
                        '数据': df
                    })
            
            # 显示K线图
            if input_method == "单只股票" and symbols:
                sym = symbols[0]
                name = dict(A_SHARE_POOL).get(sym, sym) if PICKER_AVAILABLE else sym
                df = results[0]['数据'] if results else generate_mock_data(sym)
                fig = plot_candlestick_with_indicators(df, f"{sym} - {name}")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # 显示结果表格
            if results:
                st.subheader("选股结果")
                
                # 财务因子筛选
                if 'pe_min' in dir() or 'pe_min' in locals():
                    filtered_results = []
                    for r in results:
                        # 获取财务因子（模拟数据或真实数据）
                        pe = r.get('pe', 0)
                        pb = r.get('pb', 0)
                        roe = r.get('roe', 0)
                        revenue_growth = r.get('revenue_growth', 0)
                        profit_growth = r.get('profit_growth', 0)
                        
                        # 筛选条件
                        if (pe_min <= pe <= pe_max and 
                            pb_min <= pb <= pb_max and 
                            roe >= min_roe and
                            rev_growth_min <= revenue_growth <= rev_growth_max and
                            profit_growth_min <= profit_growth <= profit_growth_max):
                            filtered_results.append(r)
                    
                    results = filtered_results
                
                # 格式化显示
                display_df = pd.DataFrame([{
                    '代码': r['代码'],
                    '名称': r['名称'],
                    '评分': r['评分'],
                    'MA20角度': f"{r['MA20角度']:.2f}°",
                    'RSI': r['RSI'],
                    '5日涨幅': r['5日涨幅'],
                    '信号': r['信号']
                } for r in results])
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # 信号统计
                signal_counts = display_df['信号'].value_counts()
                st.write("📊 信号统计:", signal_counts.to_dict())


def show_backtest():
    """回测页面"""
    st.markdown('<p class="main-header">📊 策略回测</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("回测参数")
        
        symbol = st.text_input("股票代码", value="600519")
        
        date_range = st.date_input(
            "时间范围",
            value=(datetime.now() - timedelta(days=365), datetime.now()),
            help="选择回测的时间范围"
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
        
        # 回测参数
        initial_capital = st.number_input("初始资金", value=100000, step=10000)
        stop_loss = st.slider("止损比例", 0, 20, 5) / 100
        take_profit = st.slider("止盈比例", 0, 50, 15) / 100
        
        st.subheader("策略选择")
        use_ma20 = st.checkbox("MA20角度策略", value=True)
        use_rsi = st.checkbox("RSI策略", value=True)
        use_macd = st.checkbox("MACD策略", value=True)
        
        run_button = st.button("🚀 运行回测", type="primary")
    
    with col2:
        if run_button or True:  # 始终显示结果区域
            if not run_button:
                st.info("👈 点击'运行回测'开始分析")
            
            # 生成模拟数据
            df = generate_mock_data(symbol, days=1000)
            df = calculate_indicators(df)
            
            # 筛选日期范围
            df = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
            
            if len(df) < 50:
                st.error("数据不足，无法进行回测")
                return
            
            # 模拟回测逻辑
            cash = initial_capital
            position = 0
            shares = 0
            trades = []
            equity_curve = []
            
            for i in range(1, len(df)):
                row = df.iloc[i]
                prev_row = df.iloc[i-1]
                
                # 买入信号
                signal = []
                if use_ma20:
                    ma20_angle = row.get('ma20_angle', 0)
                    if ma20_angle > 2:  # 从3改为2，更容易触发
                        signal.append('MA20_BUY')
                if use_rsi:
                    rsi = row.get('rsi', 50)
                    if rsi < 40:  # 从35改为40，更容易触发
                        signal.append('RSI_BUY')
                if use_macd:
                    if row.get('macd_diff', 0) > row.get('macd_dea', 0):
                        signal.append('MACD_BUY')
                
                # 卖出信号
                sell_signal = []
                if use_ma20:
                    if row.get('ma20_angle', 0) < -1:  # 从0改为-1
                        sell_signal.append('MA20_SELL')
                if use_rsi:
                    if row.get('rsi', 50) > 65:  # 从70改为65
                        sell_signal.append('RSI_SELL')
                
                # 交易逻辑
                if 'BUY' in signal and position == 0:
                    shares = int(cash / row['close'] * 0.8)
                    cost = shares * row['close']
                    cash -= cost
                    position = 1
                    trades.append({
                        'date': df.index[i],
                        'type': 'BUY',
                        'price': row['close'],
                        'shares': shares
                    })
                
                elif 'SELL' in sell_signal and position == 1:
                    cash += shares * row['close']
                    trades.append({
                        'date': df.index[i],
                        'type': 'SELL',
                        'price': row['close'],
                        'shares': shares
                    })
                    shares = 0
                    position = 0
                
                # 止损止盈
                if position == 1 and len(trades) > 0:
                    last_buy = trades[-1]
                    pnl_pct = (row['close'] - last_buy['price']) / last_buy['price']
                    if pnl_pct <= -stop_loss or pnl_pct >= take_profit:
                        cash += shares * row['close']
                        trades.append({
                            'date': df.index[i],
                            'type': 'SELL',
                            'price': row['close'],
                            'shares': shares
                        })
                        shares = 0
                        position = 0
                
                equity = cash + shares * row['close']
                equity_curve.append({
                    'date': df.index[i],
                    'equity': equity
                })
            
            # 计算回测结果
            final_value = cash + shares * df.iloc[-1]['close']
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            # 夏普比率
            equity_df = pd.DataFrame(equity_curve)
            if len(equity_df) > 1:
                returns = equity_df['equity'].pct_change().dropna()
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe = 0
            
            # 最大回撤
            equity_df['cummax'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = (equity_df['cummax'] - equity_df['equity']) / equity_df['cummax']
            max_drawdown = equity_df['drawdown'].max() * 100
            
            # 胜率
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            wins = 0
            for i in range(len(sell_trades)):
                if i < len(buy_trades):
                    buy_price = buy_trades[i]['price']
                    sell_price = sell_trades[i]['price']
                    if sell_price > buy_price:
                        wins += 1
            win_rate = wins / len(sell_trades) * 100 if sell_trades else 0
            
            # 显示指标
            st.subheader("📊 回测结果")
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("总收益率", f"{total_return:+.2f}%")
            m2.metric("夏普比率", f"{sharpe:.2f}")
            m3.metric("最大回撤", f"{max_drawdown:.2f}%")
            m4.metric("交易次数", f"{len(trades)}")
            
            m5, m6, m7, m8 = st.columns(4)
            m5.metric("最终资金", f"¥{final_value:,.0f}")
            m6.metric("胜率", f"{win_rate:.1f}%")
            m7.metric("买入次数", f"{len(buy_trades)}")
            m8.metric("卖出次数", f"{len(sell_trades)}")
            
            # 收益曲线
            equity_df = pd.DataFrame(equity_curve)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=equity_df['date'],
                y=equity_df['equity'],
                mode='lines',
                name='资金曲线',
                line=dict(color='#2196F3', width=2)
            ))
            
            # 基准线
            fig.add_hline(y=initial_capital, line_dash="dash", line_color="gray", 
                         annotation_text="初始资金")
            
            fig.update_layout(
                title='资金曲线',
                xaxis_title='日期',
                yaxis_title='资金',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 交易记录
            if trades:
                st.subheader("📝 交易记录")
                trades_df = pd.DataFrame(trades)
                trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
                st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("👈 设置参数后点击'运行回测'开始分析")


def show_ml_prediction():
    """ML预测页面"""
    st.markdown('<p class="main-header">🤖 ML预测</p>', unsafe_allow_html=True)
    
    if not SKLEARN_AVAILABLE:
        st.warning("⚠️ 请安装 scikit-learn: `pip install scikit-learn`")
        st.info("📦 安装后即可使用 ML 预测功能")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("ML参数")
        
        symbol = st.text_input("股票代码", value="600519")
        
        model_type = st.selectbox(
            "模型类型",
            ['random_forest', 'gradient_boosting', 'logistic'],
            format_func=lambda x: {
                'random_forest': '🌲 随机森林',
                'gradient_boosting': '📈 梯度提升',
                'logistic': '📊 逻辑回归'
            }.get(x, x)
        )
        
        train_button = st.button("📊 训练模型", type="primary")
        predict_button = st.button("🔮 预测", type="primary")
        
        st.info("""
        **特征说明:**
        - MA5/MA10/MA20: 移动平均线
        - RSI: 相对强弱指标
        - MACD: 趋势变化
        - 动量: 短期走势强度
        - 波动率: 风险水平
        """)
    
    with col2:
        if train_button:
            # 生成训练数据
            df = generate_mock_data(symbol, days=500)
            
            try:
                # 训练模型
                selector = MLSelector(model_type=model_type)
                result = selector.train(df, verbose=True)
                
                if result.get('success'):
                    st.success("✅ 模型训练完成!")
                    st.metric("模型准确率", f"{result['accuracy']:.1%}")
                    
                    # 特征重要性
                    if result.get('feature_importance'):
                        st.subheader("📊 特征重要性")
                        importance_df = pd.DataFrame([
                            {'特征': k, '重要性': v} 
                            for k, v in result['feature_importance'].items()
                        ]).sort_values('重要性', ascending=True)
                        
                        fig = px.barh(
                            importance_df, 
                            x='重要性', 
                            y='特征',
                            title='特征重要性',
                            template='plotly_dark'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"训练失败: {result.get('error')}")
                    
            except Exception as e:
                st.error(f"训练失败: {e}")
        
        if predict_button:
            df = generate_mock_data(symbol, days=200)
            
            try:
                selector = MLSelector(model_type=model_type)
                result = selector.train(df, verbose=False)
                
                if result.get('success'):
                    pred = selector.predict(df)
                    
                    st.subheader("🔮 预测结果")
                    
                    # 信号卡片
                    c1, c2, c3 = st.columns(3)
                    signal_emoji = "📈" if pred.signal == "UP" else "📉" if pred.signal == "DOWN" else "➡️"
                    c1.metric("预测信号", f"{signal_emoji} {pred.signal}")
                    c2.metric("上涨概率", f"{pred.up_probability:.1%}")
                    c3.metric("置信度", f"{pred.confidence:.1%}")
                    
                    # 概率条
                    st.subheader("📊 概率分布")
                    prob_df = pd.DataFrame({
                        '方向': ['上涨 📈', '下跌 📉'],
                        '概率': [pred.up_probability, pred.down_probability]
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='方向',
                        y='概率',
                        color='方向',
                        color_discrete_map={'上涨 📈': '#22c55e', '下跌 📉': '#ef4444'},
                        template='plotly_dark',
                        range_y=[0, 1]
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 特征重要性
                    if pred.feature_importance:
                        st.subheader("📈 特征重要性")
                        imp_df = pd.DataFrame([
                            {'特征': k, '重要性': v}
                            for k, v in sorted(pred.feature_importance.items(), 
                                              key=lambda x: x[1], reverse=True)[:5]
                        ])
                        st.dataframe(imp_df, use_container_width=True)
                else:
                    st.error(f"预测失败: {result.get('error')}")
                    
            except Exception as e:
                st.error(f"预测失败: {e}")
    
    with col2:
        if train_button:
            # 生成训练数据
            df = generate_mock_data(symbol, days=500)
            
            try:
                # 训练模型
                selector = MLSelector(model_type=model_type)
                result = selector.train(df, verbose=True)
                
                st.success("✅ 模型训练完成!")
                
                # 特征重要性
                if result.get('feature_weights'):
                    st.subheader("📊 特征重要性")
                    importance_df = pd.DataFrame([
                        {'特征': k, '重要性': v} 
                        for k, v in result['feature_weights'].items()
                    ]).sort_values('重要性', ascending=True)
                    
                    fig = px.barh(
                        importance_df, 
                        x='重要性', 
                        y='特征',
                        title='特征重要性',
                        template='plotly_dark'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"训练失败: {e}")
        
        if predict_button:
            df = generate_mock_data(symbol, days=200)
            
            try:
                selector = MLSelector(model_type=model_type)
                selector.train(df, verbose=False)
                
                pred = selector.predict(df)
                
                st.subheader("🔮 预测结果")
                
                # 预测信号
                signal = pred['signal']
                confidence = pred['confidence']
                up_prob = pred['up_probability']
                down_prob = pred['down_probability']
                
                # 信号卡片
                c1, c2, c3 = st.columns(3)
                c1.metric("预测信号", signal)
                c2.metric("上涨概率", f"{up_prob:.1%}")
                c3.metric("置信度", f"{confidence:.1%}")
                
                # 概率条
                st.subheader("📈 概率分布")
                prob_df = pd.DataFrame({
                    '方向': ['上涨', '下跌'],
                    '概率': [up_prob, down_prob]
                })
                fig = px.bar(
                    prob_df,
                    x='方向',
                    y='概率',
                    color='方向',
                    color_discrete_map={'上涨': '#4CAF50', '下跌': '#F44336'},
                    title='涨跌概率预测',
                    template='plotly_dark'
                )
                fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
                
                # 特征值
                st.subheader("📊 当前特征值")
                features = pred.get('features', {})
                if features:
                    feat_df = pd.DataFrame([
                        {'特征': k, '值': f"{v:.4f}"} 
                        for k, v in features.items()
                    ])
                    st.dataframe(feat_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"预测失败: {e}")


def show_scoring():
    """评分系统页面"""
    st.markdown('<p class="main-header">⭐ 综合评分系统</p>', unsafe_allow_html=True)
    
    if not PREMIUM_FEATURES:
        st.warning("⚠️ 评分系统模块不可用")
        st.info("💡 付费版专属功能：需要付费版许可证")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("评分参数")
        
        symbol = st.text_input("股票代码", value="600519")
        
        st.info("""
        **评分维度:**
        - 趋势强度 (25%): MA角度、均线位置
        - 动量 (25%): 各周期涨幅
        - 波动率 (15%): 稳定性评估
        - RSI位置 (20%): RSI水平和趋势
        - MACD状态 (15%): 金叉死叉
        """)
        
        score_button = st.button("📊 计算评分", type="primary")
    
    with col2:
        if score_button:
            # 生成数据
            df = generate_mock_data(symbol, days=200)
            
            try:
                # 使用评分系统
                scoring = ScoringSystem()
                result = scoring.calculate(df)
                
                # 显示综合评分
                st.subheader("🎯 综合评分")
                
                score = result.total_score
                signal = result.signal.value
                
                # 评分大卡片
                c1, c2 = st.columns(2)
                
                # 评分环形图
                fig = go.Figure(go.Pie(
                    values=[score, 100-score],
                    hole=0.7,
                    marker=dict(colors=['#4CAF50', '#E0E0E0']),
                    showlegend=False
                ))
                fig.add_annotation(
                    text=f"{score:.0f}",
                    font=dict(size=48, color='#4CAF50'),
                    showarrow=False,
                    x=0.5, y=0.5
                )
                fig.update_layout(
                    title=f'综合评分: {signal}',
                    height=200,
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                c1.plotly_chart(fig, use_container_width=True)
                
                # 信号指示
                c2.markdown(f"""
                <div style="text-align: center; padding: 40px;">
                    <h1 style="color: {'#28a745' if score >= 60 else '#ffc107' if score >= 40 else '#dc3545'};">
                        {signal}
                    </h1>
                    <p>{result.recommendation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 各维度分数
                st.subheader("📊 各维度评分")
                
                scores = result.scores
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("趋势强度", f"{scores.get('trend', 0):.1f}/25")
                m2.metric("动量", f"{scores.get('momentum', 0):.1f}/25")
                m3.metric("波动率", f"{scores.get('volatility', 0):.1f}/15")
                m4.metric("RSI位置", f"{scores.get('rsi', 0):.1f}/20")
                m5.metric("MACD状态", f"{scores.get('macd', 0):.1f}/15")
                
                # 雷达图
                st.subheader("🎯 评分雷达图")
                
                categories = ['趋势', '动量', '波动率', 'RSI', 'MACD']
                values = [
                    scores.get('trend', 0),
                    scores.get('momentum', 0),
                    scores.get('volatility', 0),
                    scores.get('rsi', 0),
                    scores.get('macd', 0)
                ]
                max_vals = [25, 25, 15, 20, 15]
                normalized = [v/m*100 if m > 0 else 0 for v, m in zip(values, max_vals)]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized + [normalized[0]],
                    theta=categories + [categories[0]],
                    fill='toself',
                    name='评分',
                    line_color='#2196F3'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100])
                    ),
                    showlegend=False,
                    height=350
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # 关键指标
                st.subheader("📌 关键指标")
                details = result.details
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MA20角度", f"{details.get('ma20_angle', 0):.2f}°")
                m2.metric("RSI(14)", f"{details.get('rsi', 50):.1f}")
                m3.metric("5日涨幅", f"{details.get('momentum_5', 0):.2%}")
                m4.metric("成交量比", f"{details.get('volume_ratio', 1):.2f}")
                
                # K线图
                st.subheader("📈 K线图")
                fig = plot_candlestick_with_indicators(df, symbol)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"评分失败: {e}")


# ==================== 侧边栏 ====================

def show_sidebar():
    """侧边栏导航"""
    st.sidebar.title("📈 quant_project")
    st.sidebar.markdown("---")
    
    # 功能导航
    page = st.sidebar.radio(
        "功能导航",
        ["选股", "回测", "ML预测", "评分系统"]
    )
    
    st.sidebar.markdown("---")
    
    # 系统信息
    st.sidebar.subheader("ℹ️ 系统信息")
    
    info = {
        "版本": "v1.2.0",
        "状态": "✅ 正常运行",
        "数据": "📊 模拟数据"
    }
    
    for k, v in info.items():
        st.sidebar.text(f"{k}: {v}")
    
    # 快捷链接
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔗 快捷链接")
    
    st.sidebar.markdown("""
    - [项目首页](https://github.com/your-repo)
    - [使用文档](#)
    - [反馈建议](#)
    """)
    
    return page


# ==================== 主函数 ====================

def show_dashboard():
    """仪表盘页面 - 总览"""
    st.markdown('<p class="main-header">📊 仪表盘</p>', unsafe_allow_html=True)
    
    # 快捷统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #22c55e;">0</div>
            <div class="metric-label">今日买入信号</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #ef4444;">0</div>
            <div class="metric-label">今日卖出信号</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #6366f1;">5</div>
            <div class="metric-label">自选股数量</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #f59e0b;">72.5</div>
            <div class="metric-label">综合评分</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 快捷选股区
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("🔍 快速选股")
        
        quick_symbol = st.text_input("股票代码", value="600519", help="输入股票代码快速查看")
        
        if st.button("查询", type="primary"):
            with st.spinner("正在获取数据..."):
                df = generate_mock_data(quick_symbol)
                df = calculate_indicators(df)
                
                if len(df) >= 20:
                    latest = df.iloc[-1]
                    signal, desc = get_signal_from_indicators(latest)
                    
                    # 显示结果
                    st.success(f"信号: {signal}")
                    st.info(f"MA20角度: {latest.get('ma20_angle', 0):.2f}°")
                    st.info(f"RSI: {latest.get('rsi', 50):.1f}")
    
    with col_right:
        st.subheader("📈 市场概览")
        st.info("📊 市场数据加载中...")
    
    st.markdown("---")
    
    # 快捷功能入口
    st.subheader("🚀 快捷功能")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.button("📈 智能选股", help="进入选股页面", use_container_width=True)
    
    with col2:
        st.button("📊 策略回测", help="进入回测页面", use_container_width=True)
    
    with col3:
        st.button("🤖 ML预测", help="进入ML预测页面", use_container_width=True)
    
    with col4:
        st.button("⭐ 评分系统", help="进入评分页面", use_container_width=True)


def main():
    """主函数"""
    # 应用自定义样式
    try:
        from theme import apply_custom_css, get_page_config
        st.markdown(apply_custom_css(), unsafe_allow_html=True)
        
        # 设置页面配置
        st.set_page_config(
            page_title="quant_project - 智能选股系统",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except ImportError:
        pass
    
    # 页面配置
    st.set_page_config(
        page_title="quant_project - 智能选股系统",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 侧边栏导航（使用Tabs）
    with st.sidebar:
        st.title("📈 quant_project")
        st.markdown("---")
        
        page = st.radio(
            "导航",
            ["仪表盘", "选股", "回测", "ML预测", "评分系统"]
        )
        
        st.markdown("---")
        
        # 系统信息
        st.subheader("ℹ️ 系统信息")
        
        info = {
            "版本": "v1.2.0",
            "状态": "✅ 正常运行",
            "数据": "📊 模拟数据"
        }
        
        for k, v in info.items():
            st.text(f"{k}: {v}")
        
        # 快捷链接
        st.markdown("---")
        st.subheader("🔗 快捷链接")
        
        st.markdown("""
        - [项目首页](https://github.com/zhangjc138/quant_project)
        - [使用文档](#)
        - [反馈建议](#)
        """)
    
    # 根据导航显示对应页面
    if page == "仪表盘":
        show_dashboard()
    elif page == "选股":
        show_stock_selector()
    elif page == "回测":
        show_backtest()
    elif page == "ML预测":
        show_ml_prediction()
    elif page == "评分系统":
        show_scoring()
    
    # 页脚
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 12px;'>"
        "quant_project v1.2.0 | 仅供学习和研究使用，不构成投资建议"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
