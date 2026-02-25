#!/usr/bin/env python3
"""
生成项目LOGO
使用Python PIL库生成简单的量化交易主题LOGO
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_logo():
    """创建项目LOGO"""
    # 创建画布
    width, height = 512, 512
    img = Image.new('RGB', (width, height), color=(15, 23, 42))  # 深蓝色背景
    
    draw = ImageDraw.Draw(img)
    
    # 绘制背景圆形
    cx, cy = width // 2, height // 2
    radius = 200
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], 
                 fill=(30, 41, 59), outline=(59, 130, 246), width=8)
    
    # 绘制K线图形状
    candle_width = 30
    candle_spacing = 50
    start_x = cx - 100
    
    # 模拟K线数据
    candles = [
        (100, 130, 110, 120),  # 阴线
        (115, 150, 120, 145),  # 阳线
        (140, 170, 150, 160),  # 阳线
        (155, 165, 158, 162),  # 十字星
        (160, 190, 170, 185),  # 阳线
    ]
    
    for i, (open_price, high_price, close_price, low_price) in enumerate(candles):
        x = start_x + i * candle_spacing
        
        # 归一化价格到画面坐标
        min_p, max_p = 90, 200
        y_scale = lambda p: cy + 100 - (p - min_p) / (max_p - min_p) * 200
        
        high_y = y_scale(high_price)
        low_y = y_scale(low_price)
        open_y = y_scale(open_price)
        close_y = y_scale(close_price)
        
        # 绘制上下影线
        draw.line([x, high_y, x, low_y], fill=(200, 200, 200), width=2)
        
        # 判断涨跌决定颜色
        if close_price > open_price:
            color = (239, 68, 68)  # 红色 (上涨)
        else:
            color = (34, 197, 94)  # 绿色 (下跌)
        
        # 绘制实体
        top = min(open_y, close_y)
        bottom = max(open_y, close_y)
        draw.rectangle([x - candle_width//2, top, x + candle_width//2, bottom], 
                      fill=color, outline=color)
    
    # 绘制MA20曲线
    ma_points = []
    for i in range(5):
        x = start_x + i * candle_spacing
        # 模拟MA20角度向上
        y = cy + 20 - i * 15
        ma_points.append((x, y))
    
    draw.line(ma_points, fill=(59, 130, 246), width=6)
    
    # 绘制文字 "Q"
    try:
        # 尝试使用系统字体
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 180)
    except:
        # 使用默认字体
        font = ImageFont.load_default()
    
    # 绘制Q字母
    draw.text((cx - 60, cy - 90), "Q", fill=(59, 130, 246), font=font)
    
    # 保存图片
    output_path = os.path.join(os.path.dirname(__file__), 'logo.png')
    img.save(output_path, 'PNG')
    print(f"LOGO已保存到: {output_path}")
    
    return output_path


if __name__ == "__main__":
    create_logo()
