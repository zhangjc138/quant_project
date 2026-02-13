#!/usr/bin/env python3
"""
MA20 量化选股工具 - 主程序
支持股票扫描、信号生成、回测等功能
"""

import argparse
import sys
import os
from datetime import datetime
from stock_strategy import StockSelector, StockSignal
from stock_backtest import Backtester
from data_manager import fetch_stock_daily, get_realtime_price


def setup_parser() -> argparse.ArgumentParser:
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="MA20 量化选股工具 - 基于均线角度的趋势选股",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 扫描监控股票池
  python main.py --scan watchlist
  
  # 扫描全部 A 股（限100只）
  python main.py --scan all --limit 100
  
  # 回测单只股票
  python main.py --backtest --symbol 600000 --start 2024-01-01
  
  # 批量回测
  python main.py --batch-backtest --symbols 600000,600036 --start 2024-01-01
  
  # 获取单只股票信号
  python main.py --signal 600000
  
  # 查看实时行情
  python main.py --realtime 600000
        """
    )
    
    # 扫描模式
    parser.add_argument('--scan', choices=['watchlist', 'all'], 
                        help='扫描模式: watchlist=股票池, all=全部A股')
    parser.add_argument('--limit', type=int, default=100,
                        help='扫描数量限制 (默认: 100)')
    parser.add_argument('--output', type=str, default='report.md',
                        help='输出报告文件名')
    
    # 回测模式
    parser.add_argument('--backtest', action='store_true',
                        help='启用回测模式')
    parser.add_argument('--batch-backtest', action='store_true',
                        help='批量回测模式')
    parser.add_argument('--symbol', type=str,
                        help='股票代码 (支持 600000 格式)')
    parser.add_argument('--symbols', type=str,
                        help='股票代码列表 (逗号分隔)')
    parser.add_argument('--start', type=str, default='2024-01-01',
                        help='回测开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str,
                        help='回测结束日期 (YYYY-MM-DD)')
    
    # 信号模式
    parser.add_argument('--signal', type=str,
                        help='获取单个股票信号')
    
    # 实时行情
    parser.add_argument('--realtime', type=str,
                        help='获取实时行情')
    
    # 配置参数
    parser.add_argument('--angle-buy', type=float, default=3.0,
                        help='买入角度阈值 (默认: 3.0°)')
    parser.add_argument('--angle-sell', type=float, default=0.0,
                        help='卖出角度阈值 (默认: 0.0°)')
    parser.add_argument('--capital', type=float, default=100000,
                        help='回测初始资金 (默认: 100000)')
    
    return parser


def cmd_scan_watchlist(args):
    """扫描监控股票池"""
    print("=== 扫描监控股票池 ===\n")
    
    selector = StockSelector()
    selector.config['angle_threshold_buy'] = args.angle_buy
    selector.config['angle_threshold_sell'] = args.angle_sell
    
    signals = selector.scan_watchlist()
    
    if not signals:
        print("未扫描到任何信号")
        return
    
    # 打印简要统计
    buy = len([s for s in signals if s.signal == "BUY"])
    sell = len([s for s in signals if s.signal == "SELL"])
    hold = len([s for s in signals if s.signal == "HOLD"])
    
    print(f"扫描结果: {len(signals)} 只")
    print(f"  🟢 买入信号: {buy} 只")
    print(f"  🔴 卖出信号: {sell} 只")
    print(f"  🟡 观望: {hold} 只")
    print()
    
    # 打印报告
    report = selector.format_report(signals)
    print(report)
    
    # 保存报告
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ 报告已保存: {args.output}")


def cmd_scan_all(args):
    """扫描全部 A 股"""
    print(f"=== 扫描全部 A 股 (限 {args.limit} 只) ===\n")
    
    selector = StockSelector()
    selector.config['angle_threshold_buy'] = args.angle_buy
    selector.config['angle_threshold_sell'] = args.angle_sell
    
    signals = selector.scan_all_a_shares(limit=args.limit)
    
    if not signals:
        print("未扫描到任何信号")
        return
    
    # 打印简要统计
    buy = len([s for s in signals if s.signal == "BUY"])
    sell = len([s for s in signals if s.signal == "SELL"])
    hold = len([s for s in signals if s.signal == "HOLD"])
    
    print(f"扫描结果: {len(signals)} 只")
    print(f"  🟢 买入信号: {buy} 只")
    print(f"  🔴 卖出信号: {sell} 只")
    print(f"  🟡 观望: {hold} 只")
    print()
    
    # 打印 TOP 10 买入信号
    print("=== TOP 10 买入信号 ===")
    for s in signals[:10]:
        print(f"  {s.symbol} {s.name}: {s.ma20_angle:.2f}° @ {s.price:.2f}")
    print()
    
    # 生成完整报告
    report = selector.format_report(signals)
    
    # 保存报告
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ 完整报告已保存: {args.output}")


def cmd_backtest(args):
    """回测单只股票"""
    symbol = args.symbol
    if not symbol:
        print("错误: 请指定 --symbol 参数")
        return
    
    # 格式化股票代码
    if not symbol.startswith(('6', '0', '3')):
        symbol = symbol.zfill(6)
    
    print(f"=== 回测 {symbol} ===")
    print(f"回测期间: {args.start} ~ {args.end or '至今'}")
    print(f"初始资金: ¥{args.capital:,.0f}\n")
    
    # 创建回测器
    backtester = Backtester({
        'initial_capital': args.capital,
        'angle_threshold_buy': args.angle_buy,
        'angle_threshold_sell': args.angle_sell,
    })
    
    # 运行回测
    result = backtester.run(symbol, args.start, args.end)
    
    # 打印报告
    report = backtester.format_result(result)
    print(report)
    
    # 保存报告
    output_file = f"backtest_{symbol}.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ 回测报告已保存: {output_file}")


def cmd_batch_backtest(args):
    """批量回测"""
    symbols_str = args.symbols or args.symbol
    if not symbols_str:
        print("错误: 请指定 --symbols 参数")
        return
    
    symbols = [s.strip().zfill(6) for s in symbols_str.split(',')]
    
    print(f"=== 批量回测 {len(symbols)} 只股票 ===")
    print(f"回测期间: {args.start} ~ {args.end or '至今'}")
    print(f"初始资金: ¥{args.capital:,.0f}\n")
    
    # 创建回测器
    backtester = Backtester({
        'initial_capital': args.capital,
        'angle_threshold_buy': args.angle_buy,
        'angle_threshold_sell': args.angle_sell,
    })
    
    # 批量回测
    results = backtester.run_batch(symbols, args.start, args.end)
    
    # 打印对比报告
    report = backtester.compare_results(results)
    print(report)
    
    # 保存报告
    output_file = "batch_backtest_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✅ 批量回测报告已保存: {output_file}")


def cmd_get_signal(args):
    """获取单个股票信号"""
    symbol = args.signal
    if not symbol:
        print("错误: 请指定 --signal 参数")
        return
    
    # 格式化股票代码
    if not symbol.startswith(('6', '0', '3')):
        symbol = symbol.zfill(6)
    
    print(f"=== {symbol} MA20 角度信号 ===\n")
    
    selector = StockSelector()
    selector.config['angle_threshold_buy'] = args.angle_buy
    selector.config['angle_threshold_sell'] = args.angle_sell
    
    result = selector.get_signal(symbol)
    
    if result is None:
        print("获取信号失败")
        return
    
    # 打印信号信息
    print(f"股票名称: {result.name}")
    print(f"当前价格: {result.price:.2f}")
    print(f"涨跌幅: {result.change_pct:+.2f}%")
    print(f"MA20: {result.ma20:.2f}")
    print(f"MA20角度: {result.ma20_angle:.2f}°")
    print(f"信号: {result.signal} - {result.signal_desc}")
    print(f"更新时间: {result.update_time}")


def cmd_realtime(args):
    """获取实时行情"""
    symbol = args.realtime
    if not symbol:
        print("错误: 请指定 --realtime 参数")
        return
    
    # 格式化股票代码
    if not symbol.startswith(('6', '0', '3')):
        symbol = symbol.zfill(6)
    
    print(f"=== {symbol} 实时行情 ===\n")
    
    realtime = get_realtime_price(symbol)
    
    if realtime is None:
        print("获取实时行情失败")
        return
    
    # 打印行情信息
    print(f"股票名称: {realtime.get('name', symbol)}")
    print(f"当前价格: {realtime['price']:.2f}")
    print(f"涨跌: {realtime['change_pct']:+.2f}%")
    print(f"今开: {realtime['open']:.2f}")
    print(f"最高: {realtime['high']:.2f}")
    print(f"最低: {realtime['low']:.2f}")
    print(f"成交量: {realtime['volume']:,.0f}")


def main():
    """主函数"""
    parser = setup_parser()
    args = parser.parse_args()
    
    # 检查是否指定了操作
    if not any([args.scan, args.backtest, args.batch_backtest, args.signal, args.realtime]):
        parser.print_help()
        print("\n错误: 请指定操作参数 (--scan, --backtest, --signal, --realtime)")
        sys.exit(1)
    
    # 根据参数执行对应操作
    if args.scan == 'watchlist':
        cmd_scan_watchlist(args)
    elif args.scan == 'all':
        cmd_scan_all(args)
    elif args.backtest:
        cmd_backtest(args)
    elif args.batch_backtest:
        cmd_batch_backtest(args)
    elif args.signal:
        cmd_get_signal(args)
    elif args.realtime:
        cmd_realtime(args)


if __name__ == "__main__":
    main()
