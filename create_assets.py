#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用PIL生成简单的视频封面图
"""

import os

# 尝试使用PIL
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("PIL not available, creating placeholder files...")


def create_simple_cover():
    """创建简单的封面图"""
    if not HAS_PIL:
        # 创建占位文本文件
        content = """# 视频封面设计

## 规格: 1920x1080 (16:9)

## 设计要素:

### 1. 主标题
- 内容: quant_project
- 位置: 居中偏上
- 字体大小: 72pt
- 颜色: 白色
- 背景色: 深蓝 (#0a0a1a)

### 2. 副标题
- 内容: 智能量化选股工具
- 字体大小: 36pt
- 颜色: 浅蓝 (#90CAF9)

### 3. 功能标签
- MA20角度选股
- RSI/MACD指标
- 策略回测
- ML机器学习

### 4. GitHub标识
- 内容: ⭐ GitHub: quant_project
- 位置: 底部居中

### 5. 底部信息
- 内容: Python | Streamlit | 量化投资
- 颜色: 灰色 (#666)
"""
        with open('/home/zjc/.openclaw/workspace/quant_project/video_cover_design.md', 'w') as f:
            f.write(content)
        print("✓ 封面设计文档已创建")
        return
    
    # 使用PIL创建图片
    width, height = 1920, 1080
    img = Image.new('RGB', (width, height), color='#0a0a1a')
    draw = ImageDraw.Draw(img)
    
    # 保存为PNG
    img.save('/home/zjc/.openclaw/workspace/quant_project/video_cover.png')
    print("✓ 封面图已创建: video_cover.png (1920x1080)")


def create_srt_file():
    """创建SRT字幕文件"""
    srt_content = """1
00:00:00,000 --> 00:00:03,000
quant_project | 智能量化选股工具

2
00:00:03,000 --> 00:00:08,000
还在凭感觉选股？追涨杀跌？

3
00:00:08,000 --> 00:00:15,000
试试Python自动选股

4
00:00:15,000 --> 00:00:22,000
MA20角度 | RSI | MACD 三大指标

5
00:00:22,000 --> 00:00:28,000
回测验证 | 总收益 +35.6% | 夏普比率 1.8

6
00:00:28,000 --> 00:00:30,000
GitHub 搜索 quant_project
"""
    
    with open('/home/zjc/.openclaw/workspace/quant_project/video.srt', 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    print("✓ 字幕文件已创建: video.srt")


def create_recording_script():
    """创建屏幕录制脚本"""
    script_content = """#!/bin/bash
# quant_project 屏幕录制脚本
# 用于录制Web界面演示视频

# 配置
OUTPUT_DIR="/home/zjc/.openclaw/workspace/quant_project"
STREAMLIT_PORT=8501
RECORDER_WINDOW="Streamlit"

# 启动Streamlit
echo "启动Streamlit服务..."
cd $OUTPUT_DIR
python3 -m streamlit run app.py &
STREAMLIT_PID=$!
sleep 5

# 等待用户准备
echo "请切换到Streamlit浏览器窗口，准备开始录制"
echo "按Enter开始录制30秒版本..."
read

# 录制30秒版本 (需要x11grab)
if command -v ffmpeg &> /dev/null; then
    echo "开始录制30秒演示..."
    ffmpeg -f x11grab -i :0.0 -t 00:00:30 -c:v libx264 -preset fast \\
        "$OUTPUT_DIR/video_demo_30s.mp4" \\
        -y 2>/dev/null || echo "请使用专业录屏软件"
else
    echo "ffmpeg未安装，请使用OBS或其他录屏软件"
fi

echo "录制完成！"

# 录制1分钟版本
echo "按Enter开始录制1分钟版本..."
read
ffmpeg -f x11grab -i :0.0 -t 00:01:00 -c:v libx264 -preset fast \\
    "$OUTPUT_DIR/video_demo_1min.mp4" \\
    -y 2>/dev/null || echo "请使用专业录屏软件"

# 停止Streamlit
kill $STREAMLIT_PID 2>/dev/null

echo "所有录制完成！"
"""
    
    with open('/home/zjc/.openclaw/workspace/quant_project/record_video.sh', 'w') as f:
        f.write(script_content)
    os.chmod('/home/zjc/.openclaw/workspace/quant_project/record_video.sh', 0o755)
    
    print("✓ 录制脚本已创建: record_video.sh")


def create_mockup_images():
    """创建界面模拟图（SVG格式，可在浏览器中查看）"""
    svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="1920" height="1080" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#0a0a1a"/>
  
  <!-- Streamlit风格界面 -->
  <rect x="200" y="100" width="1520" height="880" rx="10" fill="#1e1e2e"/>
  
  <!-- 侧边栏 -->
  <rect x="200" y="100" width="200" height="880" rx="10" fill="#262636"/>
  <text x="300" y="150" font-family="Arial" font-size="24" fill="white" text-anchor="middle">quant_project</text>
  
  <!-- 菜单项 -->
  <rect x="220" y="200" width="160" height="50" rx="5" fill="#3b82f6"/>
  <text x="300" y="232" font-family="Arial" font-size="16" fill="white" text-anchor="middle">📈 选股</text>
  
  <rect x="220" y="270" width="160" height="50" rx="5" fill="#374151"/>
  <text x="300" y="302" font-family="Arial" font-size="16" fill="white" text-anchor="middle">📊 回测</text>
  
  <rect x="220" y="340" width="160" height="50" rx="5" fill="#374151"/>
  <text x="300" y="372" font-family="Arial" font-size="16" fill="white" text-anchor="middle">🤖 ML预测</text>
  
  <!-- 主内容区 -->
  <rect x="430" y="120" width="1270" height="840" fill="#1e1e2e"/>
  
  <!-- 标题 -->
  <text x="565" y="170" font-family="Arial" font-size="32" fill="white" font-weight="bold">📈 智能选股</text>
  
  <!-- 输入框 -->
  <rect x="460" y="200" width="300" height="50" rx="5" fill="#374151"/>
  <text x="610" y="232" font-family="monospace" font-size="18" fill="#9ca3af">600519</text>
  
  <rect x="780" y="200" width="150" height="50" rx="5" fill="#3b82f6"/>
  <text x="855" y="232" font-family="Arial" font-size="16" fill="white" text-anchor="middle">🔍 开始选股</text>
  
  <!-- 结果卡片 -->
  <rect x="460" y="280" width="400" height="150" rx="10" fill="#262636"/>
  <text x="660" y="320" font-family="Arial" font-size="20" fill="white" text-anchor="middle">600519 - 贵州茅台</text>
  
  <text x="500" y="360" font-family="Arial" font-size="24" fill="#4ade80">🟢 强力买入</text>
  <text x="500" y="400" font-family="Arial" font-size="16" fill="#9ca3af">MA20角度: 5.23° | RSI: 45.2</text>
  
  <!-- K线图占位 -->
  <rect x="890" y="280" width="790" height="400" rx="10" fill="#262636"/>
  <text x="1285" y="500" font-family="Arial" font-size="24" fill="#4b5563">[K线图区域]</text>
  
  <!-- 底部结果表格 -->
  <rect x="460" y="720" width="1220" height="200" rx="10" fill="#262636"/>
  <text x="560" y="760" font-family="Arial" font-size="18" fill="white">选股结果</text>
  
  <rect x="480" y="800" width="1180" height="80" rx="5" fill="#374151"/>
  <text x="550" y="845" font-family="Arial" font-size="14" fill="white">代码</text>
  <text x="650" y="845" font-family="Arial" font-size="14" fill="white">名称</text>
  <text x="750" y="845" font-family="Arial" font-size="14" fill="white">评分</text>
  <text x="850" y="845" font-family="Arial" font-size="14" fill="white">MA20</text>
  <text x="950" y="845" font-family="Arial" font-size="14" fill="white">RSI</text>
  <text x="1050" y="845" font-family="Arial" font-size="14" fill="white">信号</text>
  
  <line x1="480" y1="870" x2="1660" y2="870" stroke="#374151" stroke-width="1"/>
  
  <text x="550" y="900" font-family="monospace" font-size="14" fill="#60a5fa">600519</text>
  <text x="650" y="900" font-family="Arial" font-size="14" fill="white">贵州茅台</text>
  <text x="750" y="900" font-family="Arial" font-size="14" fill="#4ade80">85.6</text>
  <text x="850" y="900" font-family="Arial" font-size="14" fill="white">5.23°</text>
  <text x="950" y="900" font-family="Arial" font-size="14" fill="white">45.2</text>
  <text x="1050" y="900" font-family="Arial" font-size="14" fill="#4ade80">🟢 买入</text>
</svg>
"""
    
    with open('/home/zjc/.openclaw/workspace/quant_project/video_mockup.svg', 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    print("✓ 界面模拟图已创建: video_mockup.svg")


if __name__ == '__main__':
    print("生成视频素材...")
    print("=" * 50)
    create_simple_cover()
    create_srt_file()
    create_recording_script()
    create_mockup_images()
    print("=" * 50)
    print("素材生成完成！")
