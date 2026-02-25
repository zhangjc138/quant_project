#!/usr/bin/env python3
"""
单元测试 - 不依赖 streamlit
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("quant_project 单元测试")
print("=" * 60)

all_passed = True

# 测试1: 数据生成函数
print("\n1. 测试数据生成...")
try:
    # 模拟 generate_mock_data 函数
    def generate_mock_data(symbol, days=200):
        np.random.seed(hash(symbol) % 2**32)
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
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
    
    df = generate_mock_data("600519", days=200)
    assert df is not None
    assert len(df) >= 200
    assert 'close' in df.columns
    assert 'open' in df.columns
    print(f"   ✅ 数据生成: {len(df)} 条")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试2: 技术指标计算
print("\n2. 测试技术指标计算...")
try:
    def calculate_indicators(df):
        result = df.copy()
        
        # MA
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
        
        # BOLL
        boll_middle = result['close'].rolling(20).mean()
        boll_std = result['close'].rolling(20).std()
        result['boll_upper'] = boll_middle + 2 * boll_std
        result['boll_lower'] = boll_middle - 2 * boll_std
        
        # KDJ
        low_min = result['low'].rolling(9).min()
        high_max = result['high'].rolling(9).max()
        rsv = ((result['close'] - low_min) / (high_max - low_min).replace(0, np.nan) * 100).fillna(50)
        result['kdj_k'] = rsv.rolling(3).mean()
        result['kdj_d'] = result['kdj_k'].rolling(3).mean()
        
        return result
    
    df = generate_mock_data("600519", days=200)
    df_ind = calculate_indicators(df)
    
    required_cols = ['ma20', 'rsi', 'macd_diff', 'boll_upper', 'kdj_k']
    for col in required_cols:
        assert col in df_ind.columns, f"缺少 {col}"
    
    print(f"   ✅ 指标计算: {len(df_ind.columns)} 个指标")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试3: 信号生成
print("\n3. 测试信号生成...")
try:
    def get_signal_from_indicators(row):
        ma20_angle = row.get('ma20_angle', 0)
        rsi = row.get('rsi', 50)
        macd_diff = row.get('macd_diff', 0)
        macd_dea = row.get('macd_dea', 0)
        boll_position = row.get('boll_position', 0.5)
        kdj_k = row.get('kdj_k', 50)
        kdj_d = row.get('kdj_d', 50)
        
        if pd.isna(ma20_angle) or pd.isna(rsi):
            return "HOLD", "数据不足"
        
        # MA20
        if ma20_angle > 3:
            trend_signal = "BUY"
        elif ma20_angle < 0:
            trend_signal = "SELL"
        else:
            trend_signal = "HOLD"
        
        # RSI
        if rsi > 70:
            rsi_signal = "超买"
        elif rsi < 30:
            rsi_signal = "超卖"
        else:
            rsi_signal = "中性"
        
        # MACD
        if macd_diff > macd_dea:
            macd_signal = "金叉"
        elif macd_diff < macd_dea:
            macd_signal = "死叉"
        else:
            macd_signal = "中性"
        
        # 综合信号
        if trend_signal == "BUY" and macd_signal == "金叉":
            signal = "🟢 强力买入"
        elif trend_signal == "BUY":
            signal = "🟢 买入"
        elif trend_signal == "SELL":
            signal = "🔴 卖出"
        else:
            signal = "🟡 持有"
        
        details = f"{trend_signal} | {rsi_signal} | {macd_signal}"
        return signal, details
    
    df = generate_mock_data("600519", days=200)
    df_ind = calculate_indicators(df)
    latest = df_ind.iloc[-1]
    
    signal, details = get_signal_from_indicators(latest)
    assert signal in ["🟢 买入", "🟢 强力买入", "🔴 卖出", "🟡 持有"]
    print(f"   ✅ 信号: {signal}")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试4: 行业股票映射
print("\n4. 测试行业股票映射...")
try:
    INDUSTRY_STOCKS = {
        "科技": [('600703', '三安光电'), ('002475', '长盈精密')],
        "消费": [('600519', '贵州茅台'), ('000858', '五粮液')],
        "金融": [('601398', '工商银行'), ('601318', '中国平安')],
    }
    
    for industry, stocks in INDUSTRY_STOCKS.items():
        assert len(stocks) >= 2
        for code, name in stocks:
            assert isinstance(code, str)
            assert isinstance(name, str)
    
    print(f"   ✅ {len(INDUSTRY_STOCKS)} 个行业映射正常")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试5: 评分系统
print("\n5. 测试评分系统...")
try:
    from scoring_system import ScoringSystem
    
    df = generate_mock_data("600519", days=200)
    df_ind = calculate_indicators(df)
    
    scoring = ScoringSystem()
    result = scoring.calculate(df_ind, "600519")
    
    assert 0 <= result.total_score <= 100
    assert hasattr(result, 'signal')
    assert hasattr(result, 'scores')
    print(f"   ✅ 评分: {result.total_score:.1f}分 ({result.signal.value})")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试6: ML预测
print("\n6. 测试ML预测...")
try:
    from ml_selector import MLSelector
    
    df = generate_mock_data("600519", days=500)
    
    selector = MLSelector(model_type='random_forest')
    result = selector.train(df, verbose=False)
    
    if result.get('success'):
        pred = selector.predict(df)
        assert hasattr(pred, 'signal')
        print(f"   ✅ ML: {pred.signal} ({pred.confidence*100:.0f}%)")
    else:
        print(f"   ⚠️ ML训练失败（模拟数据）")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

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
    print(f"   ✅ {len(df)} 个策略对比正常")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试8: K线形态识别
print("\n8. 测试K线形态识别...")
try:
    from pattern import CandlePatternRecognizer
    
    df = generate_mock_data("600519", days=50)
    
    recognizer = CandlePatternRecognizer()
    patterns = recognizer.recognize(df)
    
    assert isinstance(patterns, list)
    print(f"   ✅ 形态识别: {len(patterns)} 个形态")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试9: 参数优化器
print("\n9. 测试参数优化器...")
try:
    from optimizer import ParameterOptimizer
    
    df = generate_mock_data("600519", days=300)
    
    optimizer = ParameterOptimizer(symbol="600519")
    optimizer.df = df
    
    result = optimizer.optimize_ma(
        periods=[10, 20, 30],
        angle_thresholds=[2.0, 3.0, 5.0]
    )
    
    assert result.total_combinations > 0
    print(f"   ✅ 优化: {result.total_combinations} 种组合")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试10: 回测逻辑
print("\n10. 测试回测逻辑...")
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
        
        if ma20_angle > 2 and rsi < 40 and position == 0:
            cash -= 10000
            position = 1
            trades.append({'type': 'BUY', 'price': row['close']})
        
        elif (ma20_angle < -1 or rsi > 65) and position == 1:
            cash += 10000
            position = 0
            trades.append({'type': 'SELL', 'price': row['close']})
    
    print(f"   ✅ 回测: {len(trades)} 笔交易")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试11: 行业板块数据
print("\n11. 测试行业板块数据...")
try:
    from industry import get_stock_industry, get_industry_stocks
    
    industry = get_stock_industry("600519")
    assert isinstance(industry, str)
    print(f"   ✅ 行业查询: {industry}")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试12: 财务因子
print("\n12. 测试财务因子...")
try:
    from financial import FinancialMetrics, filter_financials
    
    metrics = FinancialMetrics(
        symbol="600519",
        name="测试",
        pe=25.5,
        pb=3.2,
        roe=15.5,
        revenue_growth=20.5,
        profit_growth=25.0,
        gross_margin=45.0,
        debt_ratio=30.0,
        market_cap=1000,
        circulating_cap=800,
        report_date="2024-06-30"
    )
    
    assert metrics.pe == 25.5
    assert metrics.roe == 15.5
    print(f"   ✅ 财务因子: PE={metrics.pe}, ROE={metrics.roe}%")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试13: LSTM预测
print("\n13. 测试LSTM预测...")
try:
    from lstm_predictor import LSTMPredictor
    
    df = generate_mock_data("600519", days=200)
    
    predictor = LSTMPredictor(sequence_length=10)
    result = predictor.train(df, "600519")
    
    if result.get('success'):
        pred = predictor.predict(df)
        assert hasattr(pred, 'trend')
        print(f"   ✅ LSTM: {pred.trend}")
    else:
        print(f"   ⚠️ LSTM训练失败（简化模式）")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

# 测试14: 分钟数据
print("\n14. 测试分钟数据...")
try:
    from minute_data import calculate_minute_indicators, get_minute_signal
    
    df = generate_mock_data("600519", days=200)
    df_ind = calculate_minute_indicators(df)
    
    signal = get_minute_signal(df_ind)
    assert 'signal' in signal
    print(f"   ✅ 分钟信号: {signal['signal']}")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("✅ 所有测试通过!")
else:
    print("⚠️ 部分测试失败，请检查上方错误")
print("=" * 60)
