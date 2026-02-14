#!/usr/bin/env python3
"""
主题配置模块

提供Streamlit应用的主题和样式配置
"""

# 主题配置
THEME_CONFIG = {
    "theme": "dark",
    "primaryColor": "#3b82f6",  # 亮蓝色（更清晰）
    "secondaryColor": "#8b5cf6",  # 紫色
    "accentColor": "#06b6d4",  # 青色
    "successColor": "#22c55e",  # 绿色
    "warningColor": "#f59e0b",  # 橙色
    "errorColor": "#ef4444",  # 红色
    
    # 背景色 - 使用稍浅的深色，提高可读性
    "backgroundColor": "#1e293b",  #  slate-800
    "secondaryBackgroundColor": "#334155",  # slate-700
    "textColor": "#f8fafc",  # slate-50（更白，更清晰）
    "font": "sans-serif",
}

# 股票信号颜色
SIGNAL_COLORS = {
    "BUY": "#22c55e",      # 绿色
    "SELL": "#ef4444",     # 红色
    "HOLD": "#f59e0b",     # 橙色
    "NEUTRAL": "#64748b",  # 灰色
}

# 图表配色方案
CHART_COLORS = {
    "bullish": "#22c55e",   # 上涨
    "bearish": "#ef4444",   # 下跌
    "ma20": "#6366f1",      # MA20
    "ma60": "#8b5cf6",      # MA60
    "rsi": "#f59e0b",       # RSI
    "macd": "#06b6d4",      # MACD
    "volume": "#475569",    # 成交量
}


def apply_custom_css():
    """应用自定义CSS样式 - 优化深色主题可读性"""
    css = """
    <style>
    /* 主标题样式 - 更清晰的颜色 */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #3b82f6;
    }
    
    /* 副标题样式 */
    .sub-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 1rem 0;
    }
    
    /* 指标卡片 - 更亮的背景 */
    .metric-card {
        background-color: #334155;
        border-radius: 0.75rem;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border: 1px solid #475569;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }
    
    /* 侧边栏样式 - 提高对比度 */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* 侧边栏文字更亮 */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
    }
    
    /* 展开框样式 */
    [data-testid="stExpander"] {
        background-color: #334155;
        border-radius: 0.5rem;
        border: 1px solid #475569;
    }
    
    /* 输入框样式 */
    .stTextInput > div > div > input {
        background-color: #1e293b;
        color: #f8fafc;
        border-color: #475569;
    }
    
    /* 选择框样式 */
    .stSelectbox > div > div {
        background-color: #1e293b;
        color: #f8fafc;
        border-color: #475569;
    }
    
    /* 数字输入框样式 */
    .stNumberInput > div > div {
        background-color: #1e293b;
        color: #f8fafc;
        border-color: #475569;
    }
    
    /* 下拉框选项文字 */
    div[data-baseweb="select"] > div {
        background-color: #1e293b;
        color: #f8fafc;
    }
    
    /* 表格样式 */
    [data-testid="stDataFrame"] {
        background-color: #1e293b;
        border-radius: 0.5rem;
        overflow: hidden;
        border: 1px solid #475569;
    }
    
    /* 表格文字 */
    .stDataFrame td, .stDataFrame th {
        color: #e2e8f0 !important;
        background-color: #1e293b !important;
    }
    
    /* 滑块样式 */
    .stSlider [data-testid="stSliderThumb"] {
        background-color: #3b82f6;
    }
    
    /* 提示框样式 */
    .stAlert {
        background-color: #334155;
        border-color: #475569;
        color: #f8fafc;
    }
    
    /* 单选按钮文字 */
    .stRadio label {
        color: #e2e8f0 !important;
    }
    
    /* 复选框文字 */
    .stCheckbox label {
        color: #e2e8f0 !important;
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
    }
    
    /* 标签页文字 */
    .stTabs [data-baseweb="tab-list"] button {
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        color: #f8fafc !important;
        background-color: #3b82f6 !important;
    }
    
    /* 买入信号样式 - 清晰 */
    .signal-buy {
        background-color: rgba(34, 197, 94, 0.25);
        border: 1px solid #22c55e;
        border-radius: 0.5rem;
        padding: 0.5rem;
        color: #4ade80;
        font-weight: 600;
    }
    
    /* 卖出信号样式 */
    .signal-sell {
        background-color: rgba(239, 68, 68, 0.25);
        border: 1px solid #ef4444;
        border-radius: 0.5rem;
        padding: 0.5rem;
        color: #f87171;
        font-weight: 600;
    }
    
    /* 持有信号样式 */
    .signal-hold {
        background-color: rgba(245, 158, 11, 0.25);
        border: 1px solid #f59e0b;
        border-radius: 0.5rem;
        padding: 0.5rem;
        color: #fbbf24;
        font-weight: 600;
    }
    
    /* 涨跌颜色 */
    .price-up {
        color: #4ade80;
    }
    
    .price-down {
        color: #f87171;
    }
    
    /* 股票代码链接 */
    .stock-link {
        color: #60a5fa;
        text-decoration: none;
        font-weight: 600;
    }
    
    .stock-link:hover {
        text-decoration: underline;
    }
    
    /* 行业标签 */
    .industry-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        background-color: #475569;
        color: #e2e8f0;
    }
    
    /* 信号指示器 */
    .signal-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 9999px;
        font-weight: 600;
        color: #f8fafc;
    }
    
    /* 时间戳样式 */
    .timestamp {
        font-size: 0.75rem;
        color: #64748b;
    }
    
    /* 分隔线颜色 */
    hr {
        border-color: #475569;
    }
    
    /* Streamlit markdown文字 */
    .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #e2e8f0 !important;
    }
    
    /* DataFrame表头更亮 */
    [data-testid="stDataFrame"] th {
        background-color: #334155 !important;
        color: #f8fafc !important;
    }
    
    /* DataFrame单元格 */
    [data-testid="stDataFrame"] td {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    </style>
    """
    return css


def get_page_config():
    """获取Streamlit页面配置 - 优化深色主题"""
    return {
        "page_title": "quant_project - 智能选股系统",
        "page_icon": "📈",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
        "menu_items": {
            "Get Help": "https://github.com/zhangjc138/quant_project",
            "Report a bug": "https://github.com/zhangjc138/quant_project/issues",
            "About": "quant_project v1.2.0 - 开源量化选股工具"
        }
    }


def create_metric_card(value: str, label: str, delta: str = None, color: str = "default"):
    """创建指标卡片HTML"""
    color_map = {
        "default": "#f1f5f9",
        "green": "#22c55e",
        "red": "#ef4444",
        "orange": "#f59e0b",
        "blue": "#6366f1",
    }
    
    text_color = color_map.get(color, color_map["default"])
    
    html = f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {text_color};">{value}</div>
        <div class="metric-label">{label}</div>
        {"".join([f'<div class="metric-label">{delta}</div>' if delta else ""])}
    </div>
    """
    return html


def format_signal_html(signal: str) -> str:
    """格式化信号显示"""
    signal_map = {
        "BUY": ("🟢 买入", "signal-buy"),
        "SELL": ("🔴 卖出", "signal-sell"),
        "HOLD": ("🟡 持有", "signal-hold"),
        "NEUTRAL": ("⚪ 中性", "signal-hold"),
    }
    
    emoji, css_class = signal_map.get(signal, ("⚪ 中性", "signal-hold"))
    return f'<span class="{css_class}">{emoji}</span>'


def format_price_change(change_pct: float) -> str:
    """格式化涨跌幅"""
    if change_pct > 0:
        return f'<span class="price-up">+{change_pct:.2f}%</span>'
    elif change_pct < 0:
        return f'<span class="price-down">{change_pct:.2f}%</span>'
    else:
        return f'{change_pct:.2f}%'


def create_signal_indicator(signal: str, confidence: float = None):
    """创建信号指示器"""
    signal_config = {
        "BUY": {"color": "#22c55e", "text": "买入信号"},
        "SELL": {"color": "#ef4444", "text": "卖出信号"},
        "HOLD": {"color": "#f59e0b", "text": "持有"},
        "NEUTRAL": {"color": "#64748b", "text": "中性"},
    }
    
    config = signal_config.get(signal, signal_config["NEUTRAL"])
    confidence_text = f"置信度: {confidence:.0%}" if confidence else ""
    
    html = f'''
    <div class="signal-indicator" style="color: {config['color']}; background-color: {config['color']}20;">
        <span>{config['text']}</span>
        <span style="font-size: 0.75rem; opacity: 0.8;">{confidence_text}</span>
    </div>
    '''
    return html


def create_industry_tag(industry: str) -> str:
    """创建行业标签"""
    colors = [
        "#6366f1", "#8b5cf6", "#06b6d4", "#22c55e",
        "#f59e0b", "#ef4444", "#ec4899", "#14b8a6"
    ]
    
    color = colors[hash(industry) % len(colors)]
    
    html = f'''
    <span class="industry-tag" style="background-color: {color}30; color: {color};">
        {industry}
    </span>
    '''
    return html


def create_score_bar(score: float, max_score: float = 100) -> str:
    """创建评分进度条"""
    percentage = (score / max_score) * 100
    
    # 根据分数确定颜色
    if score >= 80:
        color = "#22c55e"  # 绿色
    elif score >= 60:
        color = "#f59e0b"  # 橙色
    elif score >= 40:
        color = "#6366f1"  # 蓝色
    else:
        color = "#64748b"  # 灰色
    
    html = f'''
    <div style="margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="font-size: 0.875rem; color: #94a3b8;">评分</span>
            <span style="font-size: 0.875rem; font-weight: 600; color: {color};">{score:.0f}</span>
        </div>
        <div style="background-color: #1e293b; border-radius: 9999px; height: 0.5rem; overflow: hidden;">
            <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 9999px;"></div>
        </div>
    </div>
    '''
    return html


def create_indicator_gauge(name: str, value: float, min_val: float, max_val: float, 
                          low_threshold: float = None, high_threshold: float = None) -> str:
    """创建仪表盘样式指标显示"""
    
    # 计算位置百分比
    range_size = max_val - min_val
    position = ((value - min_val) / range_size) * 100
    position = max(0, min(100, position))
    
    # 确定颜色
    if low_threshold and high_threshold:
        if value <= low_threshold:
            color = "#22c55e"  # 超卖/低值 - 好
        elif value >= high_threshold:
            color = "#ef4444"  # 超买/高值 - 坏
        else:
            color = "#f59e0b"  # 中性
    else:
        color = "#6366f1"
    
    html = f'''
    <div style="margin: 0.5rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
            <span style="font-size: 0.875rem; color: #94a3b8;">{name}</span>
            <span style="font-size: 0.875rem; font-weight: 600; color: {color};">{value:.1f}</span>
        </div>
        <div style="position: relative; background-color: #1e293b; height: 0.5rem; border-radius: 9999px;">
            <div style="position: absolute; left: {position}%; top: 50%; transform: translate(-50%, -50%); width: 0.75rem; height: 0.75rem; background: {color}; border-radius: 50%; border: 2px solid #0f172a;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.625rem; color: #64748b; margin-top: 0.125rem;">
            <span>{min_val}</span>
            <span>{max_val}</span>
        </div>
    </div>
    '''
    return html


def create_stock_summary(stock_info: dict) -> str:
    """创建股票摘要卡片"""
    symbol = stock_info.get('symbol', '')
    name = stock_info.get('name', '')
    price = stock_info.get('price', 0)
    change = stock_info.get('change_pct', 0)
    score = stock_info.get('score', 0)
    signal = stock_info.get('signal', 'NEUTRAL')
    industry = stock_info.get('industry', '未知')
    
    html = f'''
    <div style="background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-radius: 1rem; padding: 1.5rem; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; color: #f1f5f9;">
                    <a href="https://quote.eastmoney.com/{symbol}.html" target="_blank" class="stock-link">{symbol}</a> 
                    {name}
                </h3>
                <div style="margin-top: 0.5rem;">
                    {create_industry_tag(industry)}
                </div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2rem; font-weight: 700; color: #f1f5f9;">{price:.2f}</div>
                <div style="font-size: 1rem; color: {'#22c55e' if change > 0 else '#ef4444'};">{'+' if change > 0 else ''}{change:.2f}%</div>
            </div>
        </div>
        <div style="margin-top: 1rem;">
            {create_signal_indicator(signal)}
        </div>
        <div style="margin-top: 1rem;">
            {create_score_bar(score)}
        </div>
    </div>
    '''
    return html


if __name__ == "__main__":
    print("主题配置模块加载成功")
    print("\n可用函数:")
    print("  - apply_custom_css(): 应用自定义CSS")
    print("  - get_page_config(): 获取页面配置")
    print("  - create_metric_card(): 创建指标卡片")
    print("  - format_signal_html(): 格式化信号显示")
    print("  - format_price_change(): 格式化涨跌幅")
    print("  - create_signal_indicator(): 创建信号指示器")
    print("  - create_industry_tag(): 创建行业标签")
    print("  - create_score_bar(): 创建评分进度条")
    print("  - create_indicator_gauge(): 创建仪表盘指标")
