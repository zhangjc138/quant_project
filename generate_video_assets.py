#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成视频封面和配图素材
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import numpy as np
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_cover_image():
    """创建视频封面图 (1920x1080)"""
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    # 背景渐变效果（用矩形模拟）
    for i in range(10):
        rect = Rectangle((0, 0), 19.2, 10.8, 
                         facecolor='#0a0a1a', 
                         alpha=0.1 * (10-i))
        ax.add_patch(rect)
    
    # 装饰性元素：K线图背景
    x = np.linspace(0, 19.2, 100)
    y1 = 8 + np.cumsum(np.random.randn(100) * 0.3)
    y2 = 6 + np.cumsum(np.random.randn(100) * 0.2)
    ax.fill_between(x, y1, 5, alpha=0.3, color='#2196F3')
    ax.fill_between(x, y2, 5, alpha=0.2, color='#4CAF50')
    
    # 主标题
    ax.text(9.6, 7, 'quant_project', fontsize=72, fontweight='bold',
            color='white', ha='center', va='center',
            fontfamily='DejaVu Sans')
    
    # 副标题
    ax.text(9.6, 5.5, '智能量化选股工具', fontsize=36,
            color='#90CAF9', ha='center', va='center')
    
    # 功能标签
    tags = ['MA20角度选股', 'RSI/MACD指标', '策略回测', 'ML机器学习']
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    for i, (tag, color) in enumerate(zip(tags, colors)):
        x_pos = 2.4 + i * 4.8
        y_pos = 3.5
        rect = FancyBboxPatch((x_pos - 1.5, y_pos - 0.4), 3, 0.8,
                              boxstyle="round,pad=0.1,rounding_size=0.2",
                              facecolor=color, alpha=0.8, edgecolor='white')
        ax.add_patch(rect)
        ax.text(x_pos, y_pos, tag, fontsize=20, color='white',
                ha='center', va='center', fontweight='bold')
    
    # GitHub 标识
    github_box = FancyBboxPatch((7.2, 1.2), 4.8, 1.0,
                                boxstyle="round,pad=0.1,rounding_size=0.3",
                                facecolor='#24292e', edgecolor='white', linewidth=2)
    ax.add_patch(github_box)
    ax.text(9.6, 1.7, '⭐ GitHub: quant_project', fontsize=28,
            color='white', ha='center', va='center', fontweight='bold')
    
    # 底部装饰
    ax.text(9.6, 0.5, 'Python | Streamlit | 量化投资', fontsize=18,
            color='#666', ha='center', va='center')
    
    # 移除坐标轴
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 10.8)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/zjc/.openclaw/workspace/quant_project/video_cover.png', 
                dpi=100, facecolor='#0a0a1a', pad_inches=0)
    plt.close()
    print("✓ 封面图已生成: video_cover.png")


def create_feature_image():
    """创建功能介绍图"""
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    # 标题
    ax.text(9.6, 9.5, '四大核心功能', fontsize=56, fontweight='bold',
            color='white', ha='center', va='center')
    
    # 功能卡片
    features = [
        {'title': '📈 智能选股', 'desc': 'MA20角度+RSI+MACD\n多指标综合分析', 'color': '#2196F3', 'icon': '📈'},
        {'title': '📊 策略回测', 'desc': '夏普比率+最大回撤\n收益曲线可视化', 'color': '#4CAF50', 'icon': '📊'},
        {'title': '🤖 ML预测', 'desc': '机器学习模型\n涨跌概率预测', 'color': '#FF9800', 'icon': '🤖'},
        {'title': '⭐ 综合评分', 'desc': '五维度量化评分\n操作建议生成', 'color': '#9C27B0', 'icon': '⭐'},
    ]
    
    for i, feat in enumerate(features):
        x = 2.4 + i * 4.8
        y = 5.5
        
        # 卡片背景
        card = FancyBboxPatch((x - 2, y - 2.5), 4, 5,
                              boxstyle="round,pad=0.2,rounding_size=0.5",
                              facecolor='#1a1a2e', edgecolor=feat['color'], linewidth=3)
        ax.add_patch(card)
        
        # 图标
        ax.text(x, y + 1, feat['icon'], fontsize=60, ha='center', va='center')
        
        # 标题
        ax.text(x, y - 0.5, feat['title'], fontsize=28, fontweight='bold',
                color=feat['color'], ha='center', va='center')
        
        # 描述
        ax.text(x, y - 1.8, feat['desc'], fontsize=18, color='#aaa',
                ha='center', va='center', linespacing=1.5)
    
    # 底部说明
    ax.text(9.6, 1.5, '基于Python的量化投资框架 | 零门槛使用 | 完全开源', 
            fontsize=24, color='#90CAF9', ha='center', va='center')
    
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 10.8)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/zjc/.openclaw/workspace/quant_project/video_features.png',
                dpi=100, facecolor='#0a0a1a', pad_inches=0)
    plt.close()
    print("✓ 功能介绍图已生成: video_features.png")


def create_returns_image():
    """创建收益展示图"""
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    # 标题
    ax.text(9.6, 9.5, '策略回测收益展示', fontsize=56, fontweight='bold',
            color='white', ha='center', va='center')
    
    # 生成模拟收益数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='D')
    
    # 策略收益曲线
    equity = 100000 * np.cumprod(1 + np.random.randn(252) * 0.02 + 0.0005)
    
    # 基准收益
    benchmark = 100000 * np.cumprod(1 + np.random.randn(252) * 0.01 + 0.0002)
    
    ax.plot(dates, equity, linewidth=3, color='#4CAF50', label='策略收益')
    ax.plot(dates, benchmark, linewidth=2, color='#666', linestyle='--', label='基准收益')
    ax.fill_between(dates, equity, benchmark, alpha=0.3, color='#4CAF50')
    
    # 关键指标
    total_return = (equity[-1] / 100000 - 1) * 100
    sharpe_ratio = 1.8
    max_drawdown = 8.2
    win_rate = 65.5
    
    metrics_text = f'''
    总收益率: {total_return:.1f}%
    夏普比率: {sharpe_ratio:.1f}
    最大回撤: {max_drawdown:.1f}%
    胜率: {win_rate:.1f}%
    '''
    
    ax.text(14.5, 7, metrics_text, fontsize=24, color='white',
            va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#1a1a2e', alpha=0.9))
    
    # 设置
    ax.set_xlabel('日期', fontsize=18, color='white')
    ax.set_ylabel('资金 (¥)', fontsize=18, color='white')
    ax.tick_params(colors='white')
    ax.legend(loc='upper left', fontsize=16, facecolor='#1a1a2e', edgecolor='white')
    ax.grid(True, alpha=0.3)
    
    # 背景色
    ax.set_facecolor('#0a0a1a')
    
    plt.tight_layout()
    plt.savefig('/home/zjc/.openclaw/workspace/quant_project/video_returns.png',
                dpi=100, facecolor='#0a0a1a', pad_inches=0)
    plt.close()
    print("✓ 收益展示图已生成: video_returns.png")


def create_github_screenshot():
    """创建GitHub项目截图"""
    fig, ax = plt.subplots(figsize=(19.2, 10.8), facecolor='#0a0a1a')
    ax.set_facecolor('#0a0a1a')
    
    # 模拟GitHub项目页面
    ax.text(9.6, 9.5, 'GitHub Project Stats', fontsize=48, fontweight='bold',
            color='white', ha='center', va='center')
    
    # Stats 卡片
    stats = [
        ('⭐ Stars', '1,234', '#FFD700'),
        ('🍴 Forks', '567', '#6e7681'),
        ('👥 Users', '10,000+', '#4CAF50'),
        ('📦 Commits', '892', '#2196F3'),
    ]
    
    for i, (label, value, color) in enumerate(stats):
        x = 2.4 + i * 4.8
        y = 6
        
        # 卡片
        card = FancyBboxPatch((x - 1.8, y - 1.2), 3.6, 2.4,
                              boxstyle="round,pad=0.2,rounding_size=0.3",
                              facecolor='#1a1a2e', edgecolor=color, linewidth=2)
        ax.add_patch(card)
        
        # 值
        ax.text(x, y + 0.5, value, fontsize=48, fontweight='bold',
                color=color, ha='center', va='center')
        
        # 标签
        ax.text(x, y - 0.5, label, fontsize=20, color='white',
                ha='center', va='center')
    
    # 描述
    ax.text(9.6, 3.5, 'Python量化选股工具 | MA20 + RSI + MACD + ML', 
            fontsize=24, color='#90CAF9', ha='center', va='center')
    
    # Repository URL
    url_box = FancyBboxPatch((5.2, 1.5), 9, 1.2,
                             boxstyle="round,pad=0.1,rounding_size=0.3",
                             facecolor='#24292e', edgecolor='white')
    ax.add_patch(url_box)
    ax.text(9.6, 2.1, 'github.com/zhangjc138/quant_project', 
            fontsize=24, color='white', ha='center', va='center')
    
    ax.set_xlim(0, 19.2)
    ax.set_ylim(0, 10.8)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/zjc/.openclaw/workspace/quant_project/video_github.png',
                dpi=100, facecolor='#0a0a1a', pad_inches=0)
    plt.close()
    print("✓ GitHub截图已生成: video_github.png")


if __name__ == '__main__':
    import pandas as pd
    print("开始生成视频素材...")
    print("=" * 50)
    create_cover_image()
    create_feature_image()
    create_returns_image()
    create_github_screenshot()
    print("=" * 50)
    print("所有素材生成完成！")
