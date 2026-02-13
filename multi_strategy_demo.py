#!/usr/bin/env python3
"""
多策略组合演示
演示 MA20 + RSI + MACD 组合使用
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from stock_strategy import StockSelector, TechnicalIndicator, StockSignal
from stock_backtest import Backtester, BacktestResult
import akshare as ak


def demo_single_indicators():
    """演示单个指标的计算"""
    print("=" * 60)
    print("演示 1: 单个技术指标计算")
    print("=" * 60)
    
    selector = StockSelector()
    symbol = "600000"  # 浦发银行
    
    # 加载数据
    df = selector.load_stock_data(symbol, days=250)
    if df is None:
        print("数据加载失败")
        return
    
    # 计算技术指标
    df = selector.calculate_indicators(df)
    
    # 获取最新值
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    
    print(f"\n股票: {selector.watchlist.get(symbol, {}).get('name', symbol)} ({symbol})")
    print(f"当前价格: {latest['close']:.2f}")
    print(f"MA20: {latest['MA20']:.2f}")
    print(f"MA20 角度: {latest.get('MA20_angle', 0):.2f}°")
    print(f"RSI(14): {latest['RSI']:.2f}")
    print(f"MACD(DIF): {latest['DIF']:.4f}")
    print(f"MACD(DEA): {latest['DEA']:.4f}")
    print(f"MACD(Histogram): {latest['MACD']:.4f}")
    
    # RSI 信号
    rsi_signal = TechnicalIndicator.detect_rsi_signal(latest['RSI'])
    print(f"\nRSI 信号: {rsi_signal}")
    if rsi_signal == "OVERBOUGHT":
        print("  → 建议: 股价可能过热，考虑减仓或观望")
    elif rsi_signal == "OVERSOLD":
        print("  → 建议: 股价可能超卖，可能存在反弹机会")
    else:
        print("  → 建议: RSI 处于中性区域")
    
    # MACD 信号
    macd_signal = TechnicalIndicator.detect_macd_signal(
        latest['DIF'], prev['DIF'],
        latest['DEA'], prev['DEA']
    )
    print(f"\nMACD 信号: {macd_signal}")
    if macd_signal == "GOLD_CROSS":
        print("  → 建议: MACD 金叉，短期可能转强")
    elif macd_signal == "DEAD_CROSS":
        print("  → 建议: MACD 死叉，短期可能转弱")
    else:
        print("  → 建议: MACD 无明显交叉信号")
    
    print()


def demo_strategy_combination():
    """演示策略组合"""
    print("=" * 60)
    print("演示 2: 多策略组合信号")
    print("=" * 60)
    
    selector = StockSelector()
    symbols = ["600000", "600036", "600016", "600012"]
    
    print("\n扫描股票池...")
    signals = []
    for symbol in symbols:
        signal = selector.get_signal(symbol)
        if signal:
            signals.append(signal)
            print(f"  {signal.name} ({signal.symbol}): {signal.signal} - {signal.signal_desc}")
    
    if signals:
        print(f"\n生成组合报告...")
        report = selector.format_report(signals)
        print(report)
    
    print()


def demo_backtest_comparison():
    """演示不同策略组合的回测对比"""
    print("=" * 60)
    print("演示 3: 多策略回测对比")
    print("=" * 60)
    
    symbol = "600000"
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    print(f"\n回测股票: {symbol} (浦发银行)")
    print(f"回测区间: {start_date} ~ {end_date}")
    print()
    
    # 不同策略组合
    strategies = [
        {
            "name": "MA20 角度策略",
            "use_ma20": True,
            "use_rsi": False,
            "use_macd": False,
        },
        {
            "name": "MA20 + RSI 组合",
            "use_ma20": True,
            "use_rsi": True,
            "use_macd": False,
        },
        {
            "name": "MA20 + MACD 组合",
            "use_ma20": True,
            "use_rsi": False,
            "use_macd": True,
        },
        {
            "name": "MA20 + RSI + MACD 完整组合",
            "use_ma20": True,
            "use_rsi": True,
            "use_macd": True,
        },
    ]
    
    results = []
    for strategy in strategies:
        print(f"回测中: {strategy['name']}...")
        result = run_multi_strategy_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            use_ma20=strategy["use_ma20"],
            use_rsi=strategy["use_rsi"],
            use_macd=strategy["use_macd"]
        )
        results.append((strategy["name"], result))
        
        print(f"  总收益: {result.total_return:+.2f}%")
        print(f"  年化收益: {result.annual_return:+.2f}%")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  胜率: {result.win_rate:.1f}%")
        print(f"  最大回撤: {result.max_drawdown_pct:.2f}%")
        print(f"  交易次数: {result.total_trades}")
        print()
    
    # 生成对比报告
    print("=" * 60)
    print("策略对比汇总")
    print("=" * 60)
    print(f"\n{'策略名称':<25} {'总收益':>10} {'年化收益':>10} {'夏普比率':>10} {'胜率':>8} {'最大回撤':>10}")
    print("-" * 75)
    
    for name, result in results:
        sharpe_emoji = "🟢" if result.sharpe_ratio >= 1 else "🟡" if result.sharpe_ratio >= 0 else "🔴"
        print(f"{name:<25} {result.total_return:>+9.2f}% {result.annual_return:>+9.2f}% {sharpe_emoji} {result.sharpe_ratio:>8.2f} {result.win_rate:>7.1f}% {result.max_drawdown_pct:>9.2f}%")
    
    print()


def demo_batch_backtest():
    """演示批量回测"""
    print("=" * 60)
    print("演示 4: 批量回测多只股票")
    print("=" * 60)
    
    symbols = {
        "600000": "浦发银行",
        "600036": "招商银行",
        "600016": "民生银行",
        "600012": "皖通高速",
        "600009": "上海机场",
    }
    
    start_date = "2023-01-01"
    end_date = "2024-12-31"
    
    print(f"\n回测区间: {start_date} ~ {end_date}")
    print(f"回测策略: MA20 + RSI + MACD 组合")
    print()
    
    backtester = Backtester()
    results = backtester.run_batch(list(symbols.keys()), start_date, end_date)
    
    # 生成对比报告
    print("\n批量回测结果:")
    report = backtester.compare_results(results)
    print(report)


def demo_trading_signals():
    """演示实时交易信号"""
    print("=" * 60)
    print("演示 5: 实时交易信号生成")
    print("=" * 60)
    
    print("\n根据技术指标生成交易信号:\n")
    print("买入条件 (BUY):")
    print("  1. MA20 角度 > 3° (强势上涨趋势)")
    print("  2. RSI ≤ 30 (超卖区域)")
    print("  3. MACD 金叉 (DIF 上穿 DEA)")
    print("  → 同时满足以上条件时产生买入信号")
    print()
    print("卖出条件 (SELL):")
    print("  1. MA20 角度 < 0° (下跌趋势)")
    print("  2. RSI ≥ 70 (超买区域)")
    print("  3. MACD 死叉 (DIF 下穿 DEA)")
    print("  → 任一条件满足时产生卖出信号")
    print()
    print("观望条件 (HOLD):")
    print("  - 不满足买入或卖出条件")
    print("  - 建议等待更明确的信号")
    print()


def demo_rsi_strategy():
    """演示 RSI 专项策略"""
    print("=" * 60)
    print("演示 6: RSI 专项策略")
    print("=" * 60)
    
    selector = StockSelector()
    symbol = "600000"
    
    df = selector.load_stock_data(symbol, days=250)
    if df is None:
        print("数据加载失败")
        return
    
    df = selector.calculate_indicators(df)
    
    # 统计 RSI 信号
    rsi_values = df['RSI'].dropna()
    
    print(f"\nRSI 统计 (周期: 14):")
    print(f"  当前值: {rsi_values.iloc[-1]:.2f}")
    print(f"  平均值: {rsi_values.mean():.2f}")
    print(f"  最小值: {rsi_values.min():.2f}")
    print(f"  最大值: {rsi_values.max():.2f}")
    print()
    
    # 超买超卖统计
    oversold_days = (rsi_values <= 30).sum()
    overbought_days = (rsi_values >= 70).sum()
    neutral_days = len(rsi_values) - oversold_days - overbought_days
    
    print(f"RSI 区域分布:")
    print(f"  超卖 (≤30): {oversold_days} 天 ({oversold_days/len(rsi_values)*100:.1f}%)")
    print(f"  中性 (30-70): {neutral_days} 天 ({neutral_days/len(rsi_values)*100:.1f}%)")
    print(f"  超买 (≥70): {overbought_days} 天 ({overbought_days/len(rsi_values)*100:.1f}%)")
    print()


def demo_macd_strategy():
    """演示 MACD 专项策略"""
    print("=" * 60)
    print("演示 7: MACD 专项策略")
    print("=" * 60)
    
    selector = StockSelector()
    symbol = "600000"
    
    df = selector.load_stock_data(symbol, days=250)
    if df is None:
        print("数据加载失败")
        return
    
    df = selector.calculate_indicators(df)
    
    # 计算 MACD 信号
    df['macd_signal'] = 'NEUTRAL'
    for i in range(1, len(df)):
        dif = df.iloc[i]['DIF']
        dea = df.iloc[i]['DEA']
        dif_prev = df.iloc[i-1]['DIF']
        dea_prev = df.iloc[i-1]['DEA']
        
        if pd.isna(dif) or pd.isna(dea) or pd.isna(dif_prev) or pd.isna(dea_prev):
            continue
            
        if dif_prev <= dea_prev and dif > dea:
            df.iloc[i, df.columns.get_loc('macd_signal')] = 'GOLD_CROSS'
        elif dif_prev >= dea_prev and dif < dea:
            df.iloc[i, df.columns.get_loc('macd_signal')] = 'DEAD_CROSS'
    
    # 统计 MACD 信号
    golden_crosses = (df['macd_signal'] == 'GOLD_CROSS').sum()
    dead_crosses = (df['macd_signal'] == 'DEAD_CROSS').sum()
    
    print(f"\nMACD 信号统计:")
    print(f"  金叉次数: {golden_crosses}")
    print(f"  死叉次数: {dead_crosses}")
    print()
    
    # 最近的金叉死叉
    print("最近的 MACD 信号:")
    recent_signals = df[df['macd_signal'] != 'NEUTRAL'].tail(5)
    for _, row in recent_signals.iterrows():
        signal_emoji = "🟢" if row['macd_signal'] == 'GOLD_CROSS' else "🔴"
        print(f"  {row.name.strftime('%Y-%m-%d')}: {signal_emoji} {row['macd_signal']}")
    
    print()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("  多策略组合演示 - MA20 + RSI + MACD")
    print("=" * 60)
    
    # 演示 1: 单个指标计算
    demo_single_indicators()
    
    # 演示 2: 策略组合
    demo_strategy_combination()
    
    # 演示 3: 回测对比
    demo_backtest_comparison()
    
    # 演示 4: 批量回测
    demo_batch_backtest()
    
    # 演示 5: 交易信号
    demo_trading_signals()
    
    # 演示 6: RSI 专项
    demo_rsi_strategy()
    
    # 演示 7: MACD 专项
    demo_macd_strategy()
    
    print("=" * 60)
    print("演示完成!")
    print("=" * 60)
    print("\n提示:")
    print("- 可以修改策略参数进行个性化回测")
    print("- 结合多个指标可以提高信号可靠性")
    print("- 建议在不同市场环境下调整策略参数")


if __name__ == "__main__":
    main()
