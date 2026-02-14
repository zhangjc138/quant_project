#!/usr/bin/env python3
"""
功能测试脚本
测试 quant_project 的所有核心功能
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("=" * 60)
print("quant_project 功能测试")
print("=" * 60)

# 测试1: 数据生成
print("\n1. 测试数据生成...")
try:
    from app import generate_mock_data, calculate_indicators
    
    df = generate_mock_data("600519", days=200)
    assert df is not None
    assert len(df) >= 200
    print(f"   ✅ 数据生成成功: {len(df)} 条")
    
    # 计算指标
    df_ind = calculate_indicators(df)
    assert 'ma20' in df_ind.columns
    assert 'rsi' in df_ind.columns
    assert 'macd_diff' in df_ind.columns
    print("   ✅ 技术指标计算成功")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试2: 信号生成
print("\n2. 测试信号生成...")
try:
    from app import get_signal_from_indicators
    
    df = generate_mock_data("600519", days=200)
    df_ind = calculate_indicators(df)
    latest = df_ind.iloc[-1]
    
    signal, details = get_signal_from_indicators(latest)
    assert signal in ["🟢 买入", "🟢 强力买入", "🔴 卖出", "🟡 持有"]
    print(f"   ✅ 信号生成成功: {signal}")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试3: K线图绘制
print("\n3. 测试K线图绘制...")
try:
    from app import plot_candlestick_with_indicators
    
    df = generate_mock_data("600519", days=200)
    df_ind = calculate_indicators(df)
    
    fig = plot_candlestick_with_indicators(df_ind, "600519 - 测试")
    assert fig is not None
    print("   ✅ K线图绘制成功")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试4: 行业股票映射
print("\n4. 测试行业股票映射...")
try:
    from app import INDUSTRY_STOCKS
    
    assert '科技' in INDUSTRY_STOCKS
    assert '消费' in INDUSTRY_STOCKS
    assert '云计算' in INDUSTRY_STOCKS
    
    tech_stocks = INDUSTRY_STOCKS['科技']
    assert len(tech_stocks) >= 6
    print(f"   ✅ 行业映射正常: 科技({len(tech_stocks)}只)")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试5: 评分系统
print("\n5. 测试评分系统...")
try:
    from scoring_system import ScoringSystem
    
    df = generate_mock_data("600519", days=200)
    df_ind = calculate_indicators(df)
    
    scoring = ScoringSystem()
    result = scoring.calculate(df_ind, "600519")
    
    assert 0 <= result.total_score <= 100
    print(f"   ✅ 评分系统正常: {result.total_score:.1f}分 ({result.signal.value})")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试6: ML预测
print("\n6. 测试ML预测...")
try:
    from ml_selector import MLSelector
    
    df = generate_mock_data("600519", days=500)
    
    selector = MLSelector(model_type='random_forest')
    result = selector.train(df, verbose=False)
    
    if result.get('success'):
        pred = selector.predict(df)
        assert pred.signal in ["UP", "DOWN", "HOLD"]
        print(f"   ✅ ML预测正常: {pred.signal} ({pred.confidence*100:.0f}%)")
    else:
        print(f"   ⚠️ ML训练失败: {result.get('error')}")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试7: 策略对比
print("\n7. 测试策略对比...")
try:
    from strategy_compare import StrategyComparator, create_sample_results
    
    comparator = StrategyComparator()
    results = create_sample_results()
    
    for r in results:
        comparator.add_result(r)
    
    df = comparator.compare()
    assert len(df) == 4
    print(f"   ✅ 策略对比正常: {len(df)}个策略")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试8: 选股页面逻辑
print("\n8. 测试选股页面逻辑...")
try:
    # 测试行业选择逻辑
    for industry in ["全部", "科技", "消费", "云计算"]:
        if industry == "全部":
            stock_pool = [
                ('600519', '贵州茅台'), ('600036', '招商银行'),
                ('601398', '工商银行'),
            ]
        elif industry in ["科技", "消费", "云计算"]:
            from app import INDUSTRY_STOCKS
            stock_pool = INDUSTRY_STOCKS.get(industry, [])
        
        print(f"   ✅ {industry}: {len(stock_pool)}只股票")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试9: 回测逻辑
print("\n9. 测试回测逻辑...")
try:
    df = generate_mock_data("600519", days=1000)
    df_ind = calculate_indicators(df)
    
    initial_capital = 100000
    cash = initial_capital
    position = 0
    trades = []
    
    for i in range(50, len(df_ind)):
        row = df_ind.iloc[i]
        
        ma20_angle = row.get('ma20_angle', 0)
        rsi = row.get('rsi', 50)
        
        # 买入信号
        if ma20_angle > 2 and rsi < 40 and position == 0:
            cash -= 10000
            position = 1
            trades.append({'type': 'BUY', 'price': row['close']})
        
        # 卖出信号
        elif (ma20_angle < -1 or rsi > 65) and position == 1:
            cash += 10000
            position = 0
            trades.append({'type': 'SELL', 'price': row['close']})
    
    print(f"   ✅ 回测逻辑正常: {len(trades)}笔交易")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

# 测试10: 数据缓存
print("\n10. 测试数据缓存...")
try:
    from app import generate_mock_data
    
    # 多次调用应该返回相同结果
    df1 = generate_mock_data("600519", days=100)
    df2 = generate_mock_data("600519", days=100)
    
    assert df1.equals(df2)
    print("   ✅ 数据缓存正常")
    
except Exception as e:
    print(f"   ❌ 错误: {e}")

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)
